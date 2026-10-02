"""CLI.  python -m jevbot <command>

  check-env        verify .env has what's needed (prints names only, never values)
  fetch-history    download >= 2 years of closed bars from Alpaca into data/history.csv
  backtest         run the gate on data/history.csv, write strategy.md if it passes
  run              start the live paper loop
  nightly          Opus 5.5 review -> harness backtest -> accept or reject
  rollback         restore the previous strategy.json
  report           print and send today's report
  go-live          run the final check; write the go-live record only if all clean + you confirm
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import ROOT, HardLimits, Settings

log = logging.getLogger("jevbot")
HISTORY_CSV = ROOT / "data" / "history.csv"
STRATEGY_MD = ROOT / "strategy.md"
RECAL_JSON = ROOT / "data" / "recalibration.json"
REQUIRED = ["TYPESAFE_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
OPTIONAL = ["ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "DASHBOARD_TOKEN"]


def load_history() -> pd.DataFrame:
    df = pd.read_csv(HISTORY_CSV)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def cmd_check_env(_s: Settings) -> int:
    missing = [k for k in REQUIRED if not os.getenv(k)]
    for k in REQUIRED + OPTIONAL:
        print(f"{k:22s} {'set' if os.getenv(k) else 'MISSING' if k in REQUIRED else 'not set'}")
    print("hard limits:", HardLimits.from_env())
    return 1 if missing else 0


def cmd_fetch_history(s: Settings, days: int = 800) -> int:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = CryptoHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    end = datetime.now(timezone.utc)
    df = client.get_crypto_bars(CryptoBarsRequest(
        symbol_or_symbols=s.symbol, timeframe=TimeFrame(s.timeframe_min, TimeFrameUnit.Minute),
        start=end - timedelta(days=days), end=end)).df.reset_index()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True) + pd.Timedelta(minutes=s.timeframe_min)
    df = df[df["ts"] <= end][["ts", "open", "high", "low", "close", "volume"]]
    HISTORY_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(HISTORY_CSV, index=False)
    print(f"saved {len(df)} bars, {df.ts.min()} -> {df.ts.max()}")
    return 0


def cmd_backtest(s: Settings) -> int:
    from .backtest import run_backtest
    from .jev_layer import ProxyScorer
    from .strategy import StrategyParams

    p = StrategyParams.load(s.strategy_json)
    res = run_backtest(load_history(), p, ProxyScorer())
    print("in-sample     ", res.in_sample.as_dict())
    print("out-of-sample ", res.out_of_sample.as_dict())
    for k, v in res.by_regime.items():
        print(f"  regime {k:18s}", v.as_dict())
    if res.passed:
        p.save(s.strategy_json)
        STRATEGY_MD.write_text(p.to_markdown(res.out_of_sample.as_dict()))
        print("PASSED. Wrote strategy.md")
        return 0
    print("REJECTED:\n  " + "\n  ".join(res.gate_failures))
    return 2


def cmd_run(s: Settings) -> int:
    from . import alerts
    from .broker import AlpacaBroker
    from .calibration import Recalibrator
    from .engine import Engine
    from .jev_layer import TypeSafeJevScorer
    from .outcomes import calibration_reports, calibration_verified, resolve_pending
    from .report import daily_report
    from .risk import RiskManager
    from .store import Store
    from .strategy import StrategyParams

    limits, p = HardLimits.from_env(), StrategyParams.load(s.strategy_json)
    store = Store(s.db_path)
    broker = AlpacaBroker(s.symbol, s.timeframe_min)
    risk = RiskManager(limits, s.kill_file, broker.equity())
    # Restarts must not reset drawdown or today's loss: restore them from the equity log.
    hist = store.query("SELECT max(equity) peak FROM equity")[0]["peak"]
    if hist:
        risk.peak_equity = max(risk.peak_equity, hist)
    today0 = store.query("SELECT equity FROM equity WHERE ts >= ? ORDER BY ts LIMIT 1",
                         (datetime.now(timezone.utc).date().isoformat(),))
    if today0:
        risk.day, risk.day_start_equity = datetime.now(timezone.utc).date(), today0[0]["equity"]
    eng = Engine(broker=broker, scorer=TypeSafeJevScorer(p), params=p, limits=limits, risk=risk,
                 store=store, recal=Recalibrator.load(RECAL_JSON), live=not broker.paper,
                 symbol=s.symbol)
    if broker.position_usd() > 0:  # adopt a position left open by a previous run
        from .engine import OpenTrade
        e0 = broker.avg_entry()
        last = store.query("SELECT id FROM signals WHERE action='long' ORDER BY id DESC LIMIT 1")
        eng.open = OpenTrade(last[0]["id"] if last else 0, e0, e0 * (1 - p.stop_bps / 1e4),
                             e0 * (1 + p.take_profit_bps / 1e4))
        store.event("warn", "start", f"adopted open position, entry {e0}")
    store.event("info", "start", f"mode={broker.mode_reason} strategy={p.name}")
    alerts.send("escalation", f"Bot started: {s.symbol} {broker.mode_reason}, strategy {p.name}")
    last_report = datetime.now(timezone.utc).date()
    period = s.timeframe_min * 60
    while True:
        wait = period - (time.time() % period) + 5  # 5s after each candle closes
        time.sleep(wait)
        try:
            bars = broker.closed_bars()
            out = eng.on_candle(bars, broker.book())
            log.info("%s", out)
            resolve_pending(store, bars, p)
            reps = calibration_reports(store)
            eng.cal_ok = calibration_verified(reps)
            eng.recal = Recalibrator.from_reports(reps)
            eng.recal.save(RECAL_JSON)
            today = datetime.now(timezone.utc).date()
            if today != last_report:
                alerts.send("report", daily_report(store))
                last_report = today
        except Exception as e:  # never crash silently; the supervisor restarts us
            store.event("error", "loop", f"{type(e).__name__}: {e}")
            alerts.send("error", f"{type(e).__name__}: {e}")
            raise


def cmd_nightly(s: Settings) -> int:
    from . import alerts
    from .nightly import run_nightly
    from .store import Store

    out = run_nightly(Store(s.db_path), load_history(), s.strategy_json, STRATEGY_MD)
    print(json.dumps(out, indent=2, default=str))
    alerts.send("report", f"Nightly: {'ACCEPTED' if out['accepted'] else 'rejected'}: {out['reason']}")
    return 0


def cmd_rollback(s: Settings) -> int:
    from .nightly import rollback

    f = rollback(s.strategy_json)
    print(f"restored {f}" if f else "no archived strategy to roll back to")
    return 0 if f else 1


def cmd_report(s: Settings) -> int:
    from . import alerts
    from .report import daily_report
    from .store import Store

    r = daily_report(Store(s.db_path))
    print(r)
    alerts.send("report", r)
    return 0


def cmd_go_live(s: Settings) -> int:
    from .backtest import run_backtest
    from .broker import GO_LIVE_FILE
    from .golive import run_all
    from .jev_layer import ProxyScorer
    from .store import Store
    from .strategy import StrategyParams

    bt = run_backtest(load_history(), StrategyParams.load(s.strategy_json), ProxyScorer())
    clean, text = run_all(Store(s.db_path), bt)
    print(text)
    if not clean:
        return 3
    if input("\nType I ACCEPT THE RISK to sign the go-live record: ").strip() != "I ACCEPT THE RISK":
        print("not signed")
        return 3
    GO_LIVE_FILE.write_text(json.dumps({"all_clean": True, "human_confirmed": True,
                                        "ts": datetime.now(timezone.utc).isoformat()}))
    print(f"signed {GO_LIVE_FILE}. Set LIVE_TRADING=1 in .env and restart to trade live.")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmds = {"check-env": cmd_check_env, "fetch-history": cmd_fetch_history, "backtest": cmd_backtest,
            "run": cmd_run, "nightly": cmd_nightly, "rollback": cmd_rollback, "report": cmd_report,
            "go-live": cmd_go_live}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        return 1
    return cmds[sys.argv[1]](Settings.from_env())


if __name__ == "__main__":
    sys.exit(main())

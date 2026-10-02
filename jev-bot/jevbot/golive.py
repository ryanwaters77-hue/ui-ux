"""Final check before any live money. Every answer must be clean or live stays refused."""
from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass

from .backtest import Result
from .config import ROOT
from .nightly import TUNABLE
from .outcomes import calibration_reports, calibration_verified
from .store import Store

MODEL_MODULES = {"jevbot.jev_layer", "jevbot.nightly", "anthropic", "typesafe_sdk",
                 ".jev_layer", ".nightly"}
HARD_LIMIT_MODULES = ("config.py", "risk.py", "sizing.py")
HARD_LIMIT_KEYS = {"max_position_usd", "daily_loss_limit_usd", "max_drawdown_pct",
                   "manual_approval_above_usd", "kelly_cap"}
MIN_PAPER_TRADES = 50


@dataclass
class Check:
    question: str
    clean: bool
    detail: str


def _imports(path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            out.add(("." * n.level) + (n.module or ""))
    return out


def check_limits_in_code() -> Check:
    leaks = [f"{f} imports {m}" for f in HARD_LIMIT_MODULES
             for m in _imports(ROOT / "jevbot" / f) if m in MODEL_MODULES]
    tunable_leak = HARD_LIMIT_KEYS & set(TUNABLE)
    ok = not leaks and not tunable_leak
    return Check("Is any hard limit delegated to a model instead of code?", ok,
                 "No: limits live in config/risk/sizing, which import no model code, and the "
                 "nightly loop cannot tune them." if ok else f"YES: {leaks} {tunable_leak}")


def check_kill_switch_tested() -> Check:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_engine.py", "-k", "kill"],
                       cwd=ROOT, capture_output=True, text=True)
    return Check("Did the kill switch fire in testing?", r.returncode == 0,
                 r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-300:])


def check_kill_drill(store: Store) -> Check:
    n = store.query("SELECT count(*) n FROM events WHERE kind='kill'")[0]["n"]
    return Check("Did the kill switch fire on the paper account?", n > 0,
                 f"{n} kill event(s) logged" if n else
                 "never fired on paper: run `touch KILL` once while a paper position is open")


def check_paper_vs_backtest(store: Store, bt: Result) -> Check:
    rows = store.query("SELECT result_bps FROM signals WHERE result_bps IS NOT NULL")
    r = [x["result_bps"] for x in rows]
    if len(r) < MIN_PAPER_TRADES:
        return Check("Does paper match the backtest?", False,
                     f"only {len(r)} closed paper trades; need {MIN_PAPER_TRADES}")
    hit = sum(x > 0 for x in r) / len(r)
    avg = sum(r) / len(r)
    m = bt.out_of_sample
    ok = abs(hit - m.hit_rate) <= 0.10 and avg > 0 and avg >= 0.5 * m.avg_trade_bps
    return Check("Does paper match the backtest?", ok,
                 f"paper hit {hit:.1%} / {avg:.1f} bps vs backtest {m.hit_rate:.1%} / {m.avg_trade_bps:.1f} bps")


def check_calibration(store: Store) -> Check:
    reps = calibration_reports(store)
    ok = calibration_verified(reps)
    detail = ", ".join(f"{r.question}: n={r.n} ECE={r.ece:.3f}" for r in reps) or "no resolved predictions"
    return Check("Is Jev's confidence calibrated on my own fills?", ok, detail)


def check_regimes(bt: Result) -> Check:
    bad = {k: v.avg_trade_bps for k, v in bt.by_regime.items() if v.avg_trade_bps <= 0}
    return Check("What market regime would break this?", bt.passed and not bad,
                 (f"losing OOS regimes: {bad}" if bad else "no losing OOS regime in the sample; ")
                 + "untested: exchange outage, flash crash, weekend liquidity gaps, regimes absent from the data")


BLOW_UP = """WHAT COULD BLOW UP THIS ACCOUNT?
- A gap through the stop. Stops are market exits, and a 10% candle fills far past 60 bps.
  The daily loss limit fires only after the loss has already happened.
- Venue or data outage while in a position. If the bot can't see prices, it can't exit.
  Keep positions small enough to survive an unmanaged night.
- Regime the backtest never saw: the 2-year window may not include a crash or a mania.
- Calibration drift: Jev was calibrated on its training data. If its probabilities drift on
  this venue, Kelly sizing becomes overbetting. Recheck the reliability curve weekly.
- Overfitting via the nightly loop: every accepted revision spends out-of-sample data.
  Refresh the OOS window and keep the rejection log.
- Fees: Alpaca crypto taker fees (~25 bps/side at the lowest tier) are larger than most edges.
- Operational: leaked API keys (keep trade-only keys, IP allowlist where offered), a host
  that dies silently (watch for missing Telegram heartbeats), clock skew on the server.
"""


def run_all(store: Store, bt: Result) -> tuple[bool, str]:
    checks = [check_paper_vs_backtest(store, bt), check_kill_switch_tested(), check_kill_drill(store),
              check_limits_in_code(), check_calibration(store), check_regimes(bt)]
    lines = [f"[{'CLEAN' if c.clean else 'NOT CLEAN'}] {c.question}\n    {c.detail}" for c in checks]
    all_clean = all(c.clean for c in checks)
    lines += ["", BLOW_UP, "VERDICT: " + ("all clean; human confirmation still required" if all_clean
                                          else "REFUSING TO GO LIVE until every answer is clean")]
    return all_clean, "\n".join(lines)

"""The per-candle loop: snapshot -> Jev -> decide -> size -> risk -> order -> log.

Ordering matters: risk limits are checked first and last. A Jev error, a timeout, or any
exception in the decision path results in NO trade, never a default trade.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from . import alerts
from .calibration import Recalibrator
from .config import HardLimits
from .decision import decide
from .jev_layer import Scorer
from .risk import RiskManager
from .sizing import size_trade
from .state_engine import BookTop, Snapshot, build_snapshot
from .store import Store
from .strategy import StrategyParams

log = logging.getLogger(__name__)
EPS = pd.Timedelta(microseconds=1)


@dataclass
class OpenTrade:
    signal_id: int
    entry: float
    stop: float
    tp: float
    bars_held: int = 0


class Engine:
    def __init__(self, *, broker, scorer: Scorer, params: StrategyParams, limits: HardLimits,
                 risk: RiskManager, store: Store, recal: Recalibrator | None = None,
                 calibration_verified: bool = False, live: bool = False, symbol: str = ""):
        self.broker, self.scorer, self.p, self.limits = broker, scorer, params, limits
        self.risk, self.store, self.recal = risk, store, recal or Recalibrator()
        self.cal_ok, self.live, self.symbol = calibration_verified, live, symbol
        self.open: OpenTrade | None = None
        self._kill_handled = False

    # ------------------------------------------------------------------
    def on_candle(self, candles: pd.DataFrame, book: BookTop | None = None) -> dict:
        """Call once per closed candle. Returns a summary for logging and tests."""
        decision_time = candles["ts"].iloc[-1] + EPS
        price = float(candles["close"].iloc[-1])
        if hasattr(self.broker, "set_mark"):
            self.broker.set_mark(price)

        # 1. hard limits on current equity, kill switch
        equity = self.broker.equity()
        self.store.mark_equity(equity)
        tripped = self.risk.update_equity(equity)
        if self.risk.killed:
            return self._handle_kill(tripped or "kill file present")

        # 2. manage the open position (exits are always allowed)
        snap = build_snapshot(candles, decision_time, book)
        if self.open:
            return self._manage_open(price, snap)

        # 3. ask Jev. Any failure means no trade.
        try:
            scores = self.scorer.score(snap.to_state())
        except Exception as e:
            self.store.event("error", "jev", f"{type(e).__name__}: {e}")
            alerts.send("error", f"Jev call failed, standing aside: {type(e).__name__}")
            return {"action": "flat", "reason": "jev_error"}

        d = decide(scores, self.p, self.recal)
        size = size_trade(p_win=d.p_win, confidence=d.confidence, payoff_ratio=self.p.payoff_ratio,
                          stop_bps=self.p.stop_bps, equity=equity, limits=self.limits,
                          p_cutoff=self.p.p_cutoff, calibration_verified=self.cal_ok, live=self.live)
        verdict_txt, notional = "no signal", 0.0
        if d.fired and d.action == "short":
            verdict_txt = "short signal ignored: venue is long-only"
        elif d.fired and size.notional_usd > 0:
            v = self.risk.check_order(notional_usd=size.notional_usd,
                                      current_position_usd=self.broker.position_usd(),
                                      reduces_risk=False)
            verdict_txt = v.reason
            if v.needs_approval:
                alerts.send("escalation", f"{self.symbol} long ${size.notional_usd:.0f} p={d.p_win:.2f} "
                                          f"needs manual approval; not placed.")
            if v.allowed:
                notional = size.notional_usd
        elif d.fired:
            verdict_txt = size.reason

        sid = self.store.log_signal(
            ts=snap.ts, symbol=self.symbol, price=price, snapshot=snap.to_state(),
            scores=scores.as_dict(), action=d.action if notional else "flat", p_win=d.p_win,
            confidence=d.confidence, gates=d.gates, size_usd=notional, verdict=verdict_txt,
            latency_ms=scores.latency_ms)
        self._log_predictions(sid, scores, d, snap, candles)

        if notional > 0:
            fill = self.broker.buy(notional)
            self.store.log_fill(symbol=self.symbol, side="buy", qty=fill.get("qty", 0.0),
                                price=fill.get("price", price), notional=notional, reason="entry")
            self.open = OpenTrade(sid, price, price * (1 - self.p.stop_bps / 1e4),
                                  price * (1 + self.p.take_profit_bps / 1e4))
            alerts.send("fill", f"BUY {self.symbol} ${notional:.0f} @~{price:.2f} p_win={d.p_win:.2f}")
        return {"action": "long" if notional else "flat", "p_win": d.p_win, "verdict": verdict_txt,
                "size": notional, "signal_id": sid}

    # ------------------------------------------------------------------
    def _manage_open(self, price: float, snap: Snapshot) -> dict:
        t = self.open
        t.bars_held += 1
        why = None
        if price <= t.stop:
            why = "stop"
        elif price >= t.tp:
            why = "take_profit"
        elif t.bars_held >= self.p.max_hold_bars:
            why = "time"
        elif snap.trend_bps < 0:
            why = "invalidation"
        if not why:
            return {"action": "hold"}
        fill = self.broker.sell_all(why)
        pnl_bps = fill.get("pnl_bps", (price / t.entry - 1) * 1e4)
        self.store.log_fill(symbol=self.symbol, side="sell", qty=fill.get("qty", 0.0),
                            price=fill.get("price", price), notional=0.0, reason=why)
        self.store.set_signal_result(t.signal_id, pnl_bps)
        self.store.event("info", "exit", f"{why} {pnl_bps:.1f} bps")
        alerts.send("fill", f"SELL {self.symbol} ({why}) {pnl_bps:+.1f} bps")
        self.open = None
        return {"action": "exit", "reason": why, "pnl_bps": pnl_bps}

    def _handle_kill(self, reason: str) -> dict:
        if not self._kill_handled:
            self.broker.flatten_everything()
            self.open = None
            self.store.event("critical", "kill", reason)
            alerts.send("kill", f"Kill switch: {reason}. Flattened and halted. Delete KILL file to resume.")
            self._kill_handled = True
        return {"action": "halted", "reason": reason}

    def _log_predictions(self, sid, scores, d, snap, candles) -> None:
        tf = candles["ts"].iloc[-1] - candles["ts"].iloc[-2]
        resolve = (candles["ts"].iloc[-1] + tf * self.p.horizon_bars).isoformat()
        for label, prob in scores.direction.items():
            self.store.log_prediction(sid, f"direction:{label}", prob, resolve)
        self.store.log_prediction(sid, "buying_pressure", scores.buying_pressure, resolve)
        self.store.log_prediction(sid, "p_win", d.p_win, resolve)

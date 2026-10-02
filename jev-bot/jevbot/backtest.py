"""Event-driven backtest with costs, slippage, out-of-sample split and the acceptance gate.

Opus writes and revises strategies, but it never grades them. This module grades them. A
strategy is accepted only if its OUT-OF-SAMPLE numbers clear every gate.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .decision import decide
from .jev_layer import Scorer
from .state_engine import MIN_BARS, build_snapshot
from .strategy import StrategyParams

EPS = pd.Timedelta(microseconds=1)


@dataclass(frozen=True)
class Costs:
    fee_bps: float = 25.0       # per side; Alpaca crypto taker at the lowest volume tier
    slippage_bps: float = 5.0   # per side


@dataclass(frozen=True)
class Gate:
    min_sharpe: float = 1.5
    max_drawdown_pct: float = 15.0
    min_hit_rate: float = 0.55
    min_t_stat: float = 2.0
    min_trades: int = 30
    min_span_days: int = 730


@dataclass
class Trade:
    entry_ts: str
    exit_ts: str
    side: str
    entry: float
    exit: float
    net_bps: float
    reason: str
    p_win: float
    regime: str


@dataclass
class Metrics:
    trades: int
    sharpe: float
    max_drawdown_pct: float
    hit_rate: float
    t_stat: float
    total_return_pct: float
    avg_trade_bps: float

    def as_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


@dataclass
class Result:
    in_sample: Metrics
    out_of_sample: Metrics
    by_regime: dict[str, Metrics]
    span_days: float
    trades: list[Trade] = field(default_factory=list)
    gate_failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.gate_failures


def _metrics(trades: list[Trade], bar_rets: np.ndarray, bars_per_year: float) -> Metrics:
    r = np.array([t.net_bps for t in trades], float)
    eq = np.cumprod(1 + bar_rets) if len(bar_rets) else np.array([1.0])
    peak = np.maximum.accumulate(eq)
    dd = float(((peak - eq) / peak).max() * 100) if len(eq) else 0.0
    sd = bar_rets.std(ddof=1) if len(bar_rets) > 1 else 0.0
    sharpe = float(bar_rets.mean() / sd * math.sqrt(bars_per_year)) if sd > 0 else 0.0
    t = float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))) if len(r) > 1 and r.std(ddof=1) > 0 else 0.0
    return Metrics(len(r), sharpe, dd, float((r > 0).mean()) if len(r) else 0.0, t,
                   float((eq[-1] - 1) * 100), float(r.mean()) if len(r) else 0.0)


def _regime_label(snap) -> str:
    vol = "high_vol" if snap.vol_ratio > 1.3 else "normal_vol"
    trend = "up" if snap.trend_bps > 0 else "down"
    return f"{trend}/{vol}"


def run_backtest(
    candles: pd.DataFrame,
    params: StrategyParams,
    scorer: Scorer,
    costs: Costs = Costs(),
    gate: Gate = Gate(),
    oos_fraction: float = 0.3,
) -> Result:
    c = candles.reset_index(drop=True)
    n = len(c)
    bars_per_year = 365 * 24 * 60 / params.timeframe_min
    cost = (costs.fee_bps + costs.slippage_bps) / 1e4
    split = int(n * (1 - oos_fraction))

    bar_ret = np.zeros(n)
    trades: list[Trade] = []
    pos = None  # dict(side, entry, entry_i, stop, tp, p_win, regime)
    pending = None  # decision made at close of bar i -> enter at open of i+1
    exit_next_open = False

    for i in range(MIN_BARS, n):
        row = c.iloc[i]
        o, h, l, cl = row["open"], row["high"], row["low"], row["close"]

        # 1. fills at this bar's open
        if pos and exit_next_open:
            px = o
            sgn = 1 if pos["side"] == "long" else -1
            bar_ret[i] += sgn * (px / prev_close - 1) - cost
            trades.append(_close(pos, c, i, px, "invalidation", cost))
            pos, exit_next_open = None, False
        if pending and pos is None:
            side, p_win, regime = pending
            entry = o
            sgn = 1 if side == "long" else -1
            pos = dict(side=side, entry=entry, entry_i=i, p_win=p_win, regime=regime,
                       stop=entry * (1 - sgn * params.stop_bps / 1e4),
                       tp=entry * (1 + sgn * params.take_profit_bps / 1e4))
            bar_ret[i] -= cost
            prev_close = entry
        pending = None

        # 2. intrabar exits (if stop and target both touched, assume the stop: conservative)
        if pos:
            sgn = 1 if pos["side"] == "long" else -1
            hit_stop = l <= pos["stop"] if sgn > 0 else h >= pos["stop"]
            hit_tp = h >= pos["tp"] if sgn > 0 else l <= pos["tp"]
            timed_out = i - pos["entry_i"] >= params.max_hold_bars
            px, why = None, None
            if hit_stop:
                px, why = pos["stop"], "stop"
            elif hit_tp:
                px, why = pos["tp"], "take_profit"
            elif timed_out:
                px, why = cl, "time"
            if px is not None:
                bar_ret[i] += sgn * (px / prev_close - 1) - cost
                trades.append(_close(pos, c, i, px, why, cost))
                pos = None
            else:
                bar_ret[i] += sgn * (cl / prev_close - 1)
                prev_close = cl

        # 3. decide at this bar's close, using only bars <= i
        if i + 1 >= n:
            break
        snap = build_snapshot(c.iloc[max(0, i - 199): i + 1], c.at[i, "ts"] + EPS)
        if pos:
            against = snap.trend_bps < 0 if pos["side"] == "long" else snap.trend_bps > 0
            exit_next_open = against
            continue
        d = decide(scorer.score(snap.to_state()), params)
        if d.fired:
            pending = (d.action, d.p_win, _regime_label(snap))

    span = (c["ts"].iloc[-1] - c["ts"].iloc[0]).total_seconds() / 86400
    split_ts = str(c.at[split, "ts"])
    is_tr = [t for t in trades if t.entry_ts < split_ts]
    oos_tr = [t for t in trades if t.entry_ts >= split_ts]
    res = Result(
        in_sample=_metrics(is_tr, bar_ret[:split], bars_per_year),
        out_of_sample=_metrics(oos_tr, bar_ret[split:], bars_per_year),
        by_regime={}, span_days=span, trades=trades,
    )
    for reg in sorted({t.regime for t in oos_tr}):
        tr = [t for t in oos_tr if t.regime == reg]
        res.by_regime[reg] = _metrics(tr, np.array([t.net_bps / 1e4 for t in tr]), len(tr) or 1)

    m, g = res.out_of_sample, gate
    checks = [
        (span >= g.min_span_days, f"data span {span:.0f}d < {g.min_span_days}d"),
        (m.trades >= g.min_trades, f"OOS trades {m.trades} < {g.min_trades}"),
        (m.sharpe > g.min_sharpe, f"OOS Sharpe {m.sharpe:.2f} <= {g.min_sharpe}"),
        (m.max_drawdown_pct < g.max_drawdown_pct, f"OOS max DD {m.max_drawdown_pct:.1f}% >= {g.max_drawdown_pct}%"),
        (m.hit_rate > g.min_hit_rate, f"OOS hit rate {m.hit_rate:.2%} <= {g.min_hit_rate:.0%}"),
        (m.t_stat > g.min_t_stat, f"OOS t-stat {m.t_stat:.2f} <= {g.min_t_stat}"),
        (len(res.by_regime) >= 2, f"OOS covers {len(res.by_regime)} regime(s); need >= 2"),
    ]
    for reg, rm in res.by_regime.items():
        if rm.trades >= 10:
            checks.append((rm.avg_trade_bps > 0, f"regime {reg} loses money ({rm.avg_trade_bps:.1f} bps/trade)"))
    res.gate_failures = [msg for ok, msg in checks if not ok]
    return res


def _close(pos: dict, c: pd.DataFrame, i: int, px: float, why: str, cost: float) -> Trade:
    sgn = 1 if pos["side"] == "long" else -1
    net = (sgn * (px / pos["entry"] - 1) - 2 * cost) * 1e4
    return Trade(str(c.at[pos["entry_i"], "ts"]), str(c.at[i, "ts"]), pos["side"],
                 pos["entry"], px, net, why, pos["p_win"], pos["regime"])

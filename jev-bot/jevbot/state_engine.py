"""Deterministic state engine.

On every candle it packs the market into one compact numeric snapshot. It uses only data
timestamped strictly before the decision time; anything else raises LookaheadError so the
backtest can't quietly cheat.

Candle convention: `ts` is the candle's CLOSE time. A candle closing at 10:15 is usable for a
decision at 10:15:00.000001 and later, never at 10:15 itself.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

MIN_BARS = 64


class LookaheadError(RuntimeError):
    pass


@dataclass(frozen=True)
class BookTop:
    ts: pd.Timestamp
    bid: float
    ask: float
    bid_size: float
    ask_size: float


@dataclass(frozen=True)
class Snapshot:
    ts: str
    price: float
    spread_bps: float
    book_imbalance: float     # [-1, 1], + = more resting bids
    realized_vol_pct: float   # annualized, from last 32 bars
    vol_ratio: float          # short vol / long vol
    trend_bps: float          # (EMA12 - EMA48) / price
    ret_1_bps: float
    ret_16_bps: float
    zscore_48: float          # close vs 48-bar mean, in std units
    order_flow: float         # [-1, 1] signed volume share over last 8 bars
    volume_z: float

    def to_state(self) -> dict:
        """The exact dict Jev sees. Numbers only (plus the timestamp)."""
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _ema(x: np.ndarray, span: int) -> float:
    return float(pd.Series(x).ewm(span=span, adjust=False).mean().iloc[-1])


def build_snapshot(
    candles: pd.DataFrame,
    decision_time: pd.Timestamp,
    book: BookTop | None = None,
    bars_per_year: float = 365 * 24 * 4,
    fallback_spread_bps: float = 2.0,
) -> Snapshot:
    """Build the snapshot from candles closed strictly before `decision_time`.

    `candles` must have columns ts, open, high, low, close, volume and optionally
    buy_volume, sell_volume. Rows at or after `decision_time` are an error, not a filter:
    the caller must never hand the engine future data.
    """
    decision_time = pd.Timestamp(decision_time)
    if len(candles) and candles["ts"].max() >= decision_time:
        raise LookaheadError(f"candle at/after decision time {decision_time}")
    if book is not None and book.ts >= decision_time:
        raise LookaheadError(f"book at/after decision time {decision_time}")
    if len(candles) < MIN_BARS:
        raise ValueError(f"need >= {MIN_BARS} closed bars, got {len(candles)}")

    c = candles.iloc[-200:]
    close = c["close"].to_numpy(float)
    vol = c["volume"].to_numpy(float)
    price = close[-1]
    lr = np.diff(np.log(close))

    short_sd = lr[-8:].std(ddof=1)
    long_sd = lr[-32:].std(ddof=1)
    realized = long_sd * math.sqrt(bars_per_year) * 100

    if "buy_volume" in c and "sell_volume" in c:
        b = c["buy_volume"].to_numpy(float)[-8:].sum()
        s = c["sell_volume"].to_numpy(float)[-8:].sum()
    else:  # tick rule on bars: up-bar volume counts as buying
        sign = np.sign(close[-8:] - c["open"].to_numpy(float)[-8:])
        b = vol[-8:][sign > 0].sum()
        s = vol[-8:][sign < 0].sum()
    flow = (b - s) / (b + s) if (b + s) > 0 else 0.0

    if book is not None and book.ask > book.bid > 0:
        mid = (book.ask + book.bid) / 2
        spread = (book.ask - book.bid) / mid * 1e4
        depth = book.bid_size + book.ask_size
        imb = (book.bid_size - book.ask_size) / depth if depth > 0 else 0.0
    else:
        spread, imb = fallback_spread_bps, 0.0

    m48, s48 = close[-48:].mean(), close[-48:].std(ddof=1)
    v_mu, v_sd = vol[-48:].mean(), vol[-48:].std(ddof=1)

    return Snapshot(
        ts=decision_time.isoformat(),
        price=float(price),
        spread_bps=float(spread),
        book_imbalance=float(imb),
        realized_vol_pct=float(realized),
        vol_ratio=float(short_sd / long_sd) if long_sd > 0 else 1.0,
        trend_bps=float((_ema(close, 12) - _ema(close, 48)) / price * 1e4),
        ret_1_bps=float((close[-1] / close[-2] - 1) * 1e4),
        ret_16_bps=float((close[-1] / close[-17] - 1) * 1e4),
        zscore_48=float((price - m48) / s48) if s48 > 0 else 0.0,
        order_flow=float(flow),
        volume_z=float((vol[-1] - v_mu) / v_sd) if v_sd > 0 else 0.0,
    )

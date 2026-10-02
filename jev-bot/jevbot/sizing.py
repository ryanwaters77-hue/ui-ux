"""Capped fractional Kelly sizing from Jev's (calibrated) probability.

Kelly for a bet that wins `b` units per unit risked with probability p:  f* = p - (1-p)/b.
We use at most a quarter of that, scaled 50-100% by confidence, and zero below the cutoff.
Kelly is only meaningful if p is calibrated, so live sizing requires verified calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import HardLimits


@dataclass(frozen=True)
class SizeResult:
    notional_usd: float
    kelly_full: float
    kelly_used: float
    reason: str


def kelly_fraction(p: float, payoff_ratio: float) -> float:
    if payoff_ratio <= 0:
        return 0.0
    return p - (1 - p) / payoff_ratio


def size_trade(
    *,
    p_win: float,
    confidence: float,
    payoff_ratio: float,
    stop_bps: float,
    equity: float,
    limits: HardLimits,
    p_cutoff: float,
    calibration_verified: bool,
    live: bool,
) -> SizeResult:
    if live and not calibration_verified:
        return SizeResult(0.0, 0.0, 0.0, "live sizing blocked: calibration not verified on own fills")
    if p_win < p_cutoff:
        return SizeResult(0.0, 0.0, 0.0, f"p_win {p_win:.3f} below cutoff {p_cutoff}")
    f_full = kelly_fraction(p_win, payoff_ratio)
    if f_full <= 0:
        return SizeResult(0.0, f_full, 0.0, "no edge at this payoff ratio")
    scale = 0.5 + 0.5 * min(max((confidence - 0.5) / 0.4, 0.0), 1.0)  # 0.5 at conf<=.5, 1.0 at >=.9
    f_used = min(f_full * limits.kelly_cap * scale, limits.kelly_cap)
    risk_usd = equity * f_used                      # capital lost if the stop hits
    notional = risk_usd / (stop_bps / 1e4)
    notional = min(notional, limits.max_position_usd, equity)
    return SizeResult(round(notional, 2), f_full, f_used, "ok")

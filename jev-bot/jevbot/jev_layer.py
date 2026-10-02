"""Jev: the fast, calibrated decision layer.

Each strategy rule is compiled into a fixed-outcome question. One call answers all of them
in parallel and returns a probability per outcome. Jev never sees anything but the numeric
snapshot: no headlines, no free text, so there's nothing to inject instructions through.

`ProxyScorer` is a deterministic stand-in with the same interface. The backtest uses it,
and so do the tests. It is NOT Jev, and the calibration you measure on it says nothing
about Jev.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

from .strategy import StrategyParams

REGIMES = ("trending_up", "trending_down", "ranging", "volatile_chop")
DIRECTIONS = ("up", "down", "flat")
RISK_STATES = ("normal", "elevated", "extreme")
SETUP_LEVELS = (
    "no setup: conflicting or absent signals",
    "weak: one supporting factor",
    "fair: trend and flow agree, nothing else",
    "good: trend, flow and book all agree, normal volatility",
    "excellent: textbook pullback in a clean trend with strong flow",
)

STATE_LEGEND = {
    "price": "last closed price",
    "spread_bps": "bid-ask spread in basis points",
    "book_imbalance": "(bid size - ask size) / total, -1..1",
    "realized_vol_pct": "annualized realized volatility, last 32 bars",
    "vol_ratio": "8-bar vol / 32-bar vol; >1.5 means volatility is expanding",
    "trend_bps": "(EMA12 - EMA48) / price in bps; sign is trend direction",
    "ret_1_bps": "last bar return, bps",
    "ret_16_bps": "16-bar return, bps",
    "zscore_48": "price vs 48-bar mean in std units",
    "order_flow": "signed volume share over last 8 bars, -1..1",
    "volume_z": "last bar volume z-score",
}


def build_questions(p: StrategyParams) -> dict:
    """Question dicts in the typesafe-sdk wire format."""
    qs = _base_questions(p)
    for name, text in p.question_text.items():
        if name in qs:  # only rewrites instructions; outcomes stay fixed
            qs[name]["instructions"] = text
    return qs


def _base_questions(p: StrategyParams) -> dict:
    return {
        "regime": {
            "type": "choice",
            "instructions": "Which market regime describes the current state?",
            "criteria": {
                "trending_up": "persistent upward trend, pullbacks shallow",
                "trending_down": "persistent downward trend, rallies shallow",
                "ranging": "mean-reverting, no persistent trend",
                "volatile_chop": "large moves in both directions, no edge",
            },
        },
        "direction": {
            "type": "choice",
            "instructions": f"Over the next {p.horizon_bars} bars of {p.timeframe_min} minutes, "
                            f"where will price close relative to now?",
            "criteria": {
                "up": f"up more than {p.flat_band_bps} bps",
                "down": f"down more than {p.flat_band_bps} bps",
                "flat": f"within {p.flat_band_bps} bps",
            },
        },
        "buying_pressure": {
            "type": "noul",
            "instructions": "Is the current buying pressure real (sustained net demand), "
                            "not a one-bar spike or thin-book artifact?",
        },
        "setup_quality": {
            "type": "score",
            "instructions": "Rate the quality of a long trend-following setup right now.",
            "criteria": list(SETUP_LEVELS),
        },
        "risk_state": {
            "type": "choice",
            "instructions": "What is the current risk state of the market?",
            "criteria": {
                "normal": "volatility and spreads normal",
                "elevated": "volatility or spreads noticeably above normal",
                "extreme": "disorderly: gaps, blowout spreads, volatility spike",
            },
        },
    }


@dataclass(frozen=True)
class JevScores:
    regime: dict[str, float]
    direction: dict[str, float]
    buying_pressure: float
    setup_probs: tuple[float, ...]   # P(level i), i = 0..4
    risk_state: dict[str, float]
    latency_ms: float = 0.0
    input_tokens: int = 0

    @property
    def setup_quality(self) -> float:
        """Expected setup score normalized to [0, 1]."""
        n = len(self.setup_probs) - 1
        return sum(i * p for i, p in enumerate(self.setup_probs)) / n

    def as_dict(self) -> dict:
        return {
            "regime": self.regime, "direction": self.direction,
            "buying_pressure": self.buying_pressure, "setup_quality": self.setup_quality,
            "risk_state": self.risk_state, "latency_ms": self.latency_ms,
        }


class Scorer(Protocol):
    def score(self, state: dict) -> JevScores: ...


class TypeSafeJevScorer:
    """Real Jev via typesafe-sdk. Reads TYPESAFE_API_KEY from the environment."""

    def __init__(self, params: StrategyParams, model: str = "jev-latest", timeout: float = 2.0):
        from typesafe_sdk import TypeSafeClient  # imported lazily so tests don't need it

        self._client = TypeSafeClient(model=model, timeout=timeout)
        self._questions = build_questions(params)

    def score(self, state: dict) -> JevScores:
        t0 = time.perf_counter()
        resp = self._client.system_one(
            state={"legend": STATE_LEGEND, "snapshot": state}, questions=self._questions
        )
        latency = (time.perf_counter() - t0) * 1000
        a = {k: v.root if hasattr(v, "root") else v for k, v in resp.answers.items()}
        setup = a["setup_quality"].probabilities
        # Score probabilities are keyed by level; order them 0..n
        setup_probs = tuple(setup[k] for k in sorted(setup, key=int))
        return JevScores(
            regime=dict(a["regime"].probabilities),
            direction=dict(a["direction"].probabilities),
            buying_pressure=float(a["buying_pressure"].noul),
            setup_probs=setup_probs,
            risk_state=dict(a["risk_state"].probabilities),
            latency_ms=latency,
            input_tokens=resp.usage.input_tokens,
        )


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _softmax(d: dict[str, float]) -> dict[str, float]:
    m = max(d.values())
    e = {k: math.exp(v - m) for k, v in d.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


class ProxyScorer:
    """Deterministic feature->probability map with Jev's output shape. Backtests and tests only."""

    def score(self, s: dict) -> JevScores:
        trend = s["trend_bps"] / 30
        flow = s["order_flow"] * 2
        chop = max(0.0, s["vol_ratio"] - 1.2) * 3
        regime = _softmax({"trending_up": trend, "trending_down": -trend,
                           "ranging": 0.5 - abs(trend), "volatile_chop": chop - 1})
        pull = -s["zscore_48"] * 0.3  # pullbacks in an uptrend are better entries
        direction = _softmax({"up": 0.6 * trend + 0.4 * flow + pull,
                              "down": -0.6 * trend - 0.4 * flow - pull, "flat": 0.3})
        bp = _sig(flow + 0.5 * s["book_imbalance"])
        q = _sig(trend + flow - chop)  # 0..1 quality
        centers = [0, 0.25, 0.5, 0.75, 1.0]
        w = [math.exp(-((q - c) ** 2) / 0.03) for c in centers]
        setup = tuple(x / sum(w) for x in w)
        risk = _softmax({"normal": 1.5 - chop, "elevated": 0.0 + chop * 0.5,
                         "extreme": chop - 2 + max(0, s["spread_bps"] - 20) / 10})
        return JevScores(regime, direction, bp, setup, risk)

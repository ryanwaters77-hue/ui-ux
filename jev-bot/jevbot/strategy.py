"""Strategy parameters: the only thing the nightly Opus loop is allowed to rewrite.

strategy.json holds the machine-readable params; strategy.md is rendered from it for humans.
Thresholds here can only make trading *stricter*; HardLimits in config.py are separate and
untouchable.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class StrategyParams:
    name: str = "trend-pullback-v1"
    timeframe_min: int = 15
    horizon_bars: int = 4            # how far ahead Jev's direction question looks
    flat_band_bps: float = 10.0      # |move| below this counts as "flat"
    allow_short: bool = False        # Alpaca crypto is long-only

    # Gates: every probability must clear its threshold or nothing fires.
    min_p_regime: float = 0.45       # p(regime that supports the side)
    min_p_direction: float = 0.55
    min_p_buying_pressure: float = 0.55
    min_setup_quality: float = 0.55  # expected score normalized to [0, 1]
    max_p_risk_extreme: float = 0.15

    # Explicit weights for the combined win probability.
    w_direction: float = 0.5
    w_pressure: float = 0.3
    w_setup: float = 0.2

    # Exits
    stop_bps: float = 60.0
    take_profit_bps: float = 120.0
    max_hold_bars: int = 16
    invalidation: str = "exit if trend_bps changes sign against the position"

    # Sizing
    p_cutoff: float = 0.55           # zero size below this combined probability

    # Plain-language rules learned from losses (appended nightly).
    rules: list[str] = field(default_factory=list)
    # Optional rewrites of Jev question instructions, keyed by question name.
    question_text: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        w = self.w_direction + self.w_pressure + self.w_setup
        if abs(w - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1, got {w}")
        if self.stop_bps <= 0 or self.take_profit_bps <= 0:
            raise ValueError("stop and take profit must be positive")
        for k in ("min_p_regime", "min_p_direction", "min_p_buying_pressure",
                  "min_setup_quality", "max_p_risk_extreme", "p_cutoff"):
            if not 0.0 <= getattr(self, k) <= 1.0:
                raise ValueError(f"{k} must be a probability")

    @property
    def payoff_ratio(self) -> float:
        return self.take_profit_bps / self.stop_bps

    @classmethod
    def load(cls, path: Path) -> "StrategyParams":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    def to_markdown(self, metrics: dict | None = None) -> str:
        side = "long or short" if self.allow_short else "long only"
        lines = [
            f"# Strategy: {self.name}",
            "",
            f"Timeframe: {self.timeframe_min}-minute candles, {side}.",
            "",
            "## Entry",
            f"Fire only when every Jev probability clears its gate on the closed candle:",
            f"- p(regime supports side) >= {self.min_p_regime}",
            f"- p(direction = side over next {self.horizon_bars} bars) >= {self.min_p_direction}",
            f"- p(buying pressure is real) >= {self.min_p_buying_pressure} (long) / <= {1 - self.min_p_buying_pressure:.2f} (short)",
            f"- setup quality (normalized) >= {self.min_setup_quality}",
            f"- p(risk state = extreme) <= {self.max_p_risk_extreme}",
            f"- combined p_win = {self.w_direction}*p_dir + {self.w_pressure}*p_pressure + {self.w_setup}*setup >= {self.p_cutoff}",
            "Order goes in at the next bar's open.",
            "",
            "## Exit",
            f"- Stop: {self.stop_bps} bps against entry",
            f"- Take profit: {self.take_profit_bps} bps in favor",
            f"- Time stop: {self.max_hold_bars} bars",
            f"- Invalidation: {self.invalidation}",
            "",
            "## Rules learned from losses",
            *([f"- {r}" for r in self.rules] or ["- (none yet)"]),
        ]
        if metrics:
            lines += ["", "## Backtest (out of sample)", "",
                      *[f"- {k}: {v}" for k, v in metrics.items()]]
        return "\n".join(lines) + "\n"

"""Deterministic combination of Jev's probabilities into an action.

Jev advises; this code decides. Every probability must clear its gate, and the gates and
weights come from strategy.json, never from a model's output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .calibration import Recalibrator
from .jev_layer import JevScores
from .strategy import StrategyParams


@dataclass
class Decision:
    action: str                 # "long" | "short" | "flat"
    p_win: float                # combined, calibrated
    confidence: float           # Jev's own confidence in the direction choice
    gates: dict[str, bool] = field(default_factory=dict)
    probs: dict[str, float] = field(default_factory=dict)

    @property
    def fired(self) -> bool:
        return self.action != "flat"


def decide(
    s: JevScores, p: StrategyParams, recal: Recalibrator | None = None
) -> Decision:
    cal = recal.apply if recal else (lambda _q, x: x)
    best: Decision | None = None
    sides = ("long", "short") if p.allow_short else ("long",)
    for side in sides:
        regime_key = "trending_up" if side == "long" else "trending_down"
        dir_key = "up" if side == "long" else "down"
        p_regime = s.regime.get(regime_key, 0.0)
        p_dir = cal(f"direction:{dir_key}", s.direction.get(dir_key, 0.0))
        bp = cal("buying_pressure", s.buying_pressure)
        p_pressure = bp if side == "long" else 1 - bp
        setup = s.setup_quality
        p_extreme = s.risk_state.get("extreme", 0.0)

        gates = {
            "regime": p_regime >= p.min_p_regime,
            "direction": p_dir >= p.min_p_direction,
            "buying_pressure": p_pressure >= p.min_p_buying_pressure,
            "setup_quality": setup >= p.min_setup_quality,
            "risk_state": p_extreme <= p.max_p_risk_extreme,
        }
        # Components are recalibrated individually (fit on raw Jev output vs our outcomes);
        # the combination itself is a fixed, explicit weighting.
        p_win = p.w_direction * p_dir + p.w_pressure * p_pressure + p.w_setup * setup
        gates["p_win"] = p_win >= p.p_cutoff
        d = Decision(
            action=side if all(gates.values()) else "flat",
            p_win=p_win,
            confidence=max(s.direction.values()) if s.direction else 0.0,
            gates=gates,
            probs={"regime": p_regime, "direction": p_dir, "pressure": p_pressure,
                   "setup": setup, "risk_extreme": p_extreme},
        )
        if best is None or (d.fired and (not best.fired or d.p_win > best.p_win)):
            best = d
    assert best is not None
    return best

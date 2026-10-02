import numpy as np
import pytest

from jevbot.calibration import Recalibrator, ece, evaluate, fit_platt
from jevbot.decision import decide
from jevbot.jev_layer import JevScores, build_questions
from jevbot.strategy import StrategyParams

P = StrategyParams()


def scores(**kw):
    base = dict(regime={"trending_up": .7, "trending_down": .1, "ranging": .1, "volatile_chop": .1},
                direction={"up": .7, "down": .1, "flat": .2}, buying_pressure=.75,
                setup_probs=(0, 0, .1, .4, .5), risk_state={"normal": .9, "elevated": .08, "extreme": .02})
    base.update(kw)
    return JevScores(**base)


def test_fires_when_every_gate_clears():
    d = decide(scores(), P)
    assert d.action == "long" and all(d.gates.values())


@pytest.mark.parametrize("override,gate", [
    ({"direction": {"up": .4, "down": .3, "flat": .3}}, "direction"),
    ({"buying_pressure": .4}, "buying_pressure"),
    ({"risk_state": {"normal": .5, "elevated": .2, "extreme": .3}}, "risk_state"),
    ({"setup_probs": (.6, .3, .1, 0, 0)}, "setup_quality"),
    ({"regime": {"trending_up": .2, "trending_down": .2, "ranging": .5, "volatile_chop": .1}}, "regime"),
])
def test_any_single_failed_gate_blocks(override, gate):
    d = decide(scores(**override), P)
    assert d.action == "flat" and not d.gates[gate]


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        StrategyParams(w_direction=.9)


def test_questions_cover_all_factors():
    q = build_questions(P)
    assert {k: v["type"] for k, v in q.items()} == {
        "regime": "choice", "direction": "choice", "buying_pressure": "noul",
        "setup_quality": "score", "risk_state": "choice"}


def test_platt_fixes_overconfidence():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(.2, .8, 5000)
    y = (rng.uniform(size=5000) < true_p).astype(float)
    logit = np.log(true_p / (1 - true_p))
    overconf = 1 / (1 + np.exp(-2.5 * logit))      # a model that's too sure of itself
    rep = evaluate("q", overconf, y)
    assert rep.ece > .05 and rep.ece_after < .02
    a, _ = fit_platt(overconf, y)
    assert a == pytest.approx(0.4, abs=0.05)
    rc = Recalibrator.from_reports([rep])
    assert "q" in rc.maps and rc.apply("q", .9) < .9


def test_calibrated_input_is_left_alone():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=4000)
    y = (rng.uniform(size=4000) < p).astype(float)
    rep = evaluate("q", p, y)
    assert rep.calibrated and ece(p, y) < .05
    assert Recalibrator.from_reports([rep]).maps == {}

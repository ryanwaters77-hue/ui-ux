"""Parses a real typesafe-sdk response object (no network) into JevScores."""
from typesafe_sdk import SystemOneResponse

from jevbot.jev_layer import TypeSafeJevScorer, build_questions
from jevbot.strategy import StrategyParams

RAW = {
    "model": "jev-latest", "usage": {"input_tokens": 812, "output_tokens": 0},
    "answers": {
        "regime": {"type": "choice", "choice": "trending_up", "confidence": .6,
                   "probabilities": {"trending_up": .6, "trending_down": .1, "ranging": .2, "volatile_chop": .1}},
        "direction": {"type": "choice", "choice": "up", "confidence": .58,
                      "probabilities": {"up": .58, "down": .17, "flat": .25}},
        "buying_pressure": {"type": "noul", "noul": .71},
        "setup_quality": {"type": "score", "score": 2.9, "confidence": .5,
                          "legend": {i: "x" for i in range(5)},
                          "probabilities": {0: .05, 1: .05, 2: .1, 3: .5, 4: .3}},
        "risk_state": {"type": "choice", "choice": "normal", "confidence": .8,
                       "probabilities": {"normal": .8, "elevated": .15, "extreme": .05}},
    },
}


class FakeClient:
    def __init__(self): self.sent = None
    def system_one(self, state, questions):
        self.sent = (state, questions)
        return SystemOneResponse.model_validate(RAW)


def test_parses_sdk_response():
    s = TypeSafeJevScorer.__new__(TypeSafeJevScorer)
    s._client, s._questions = FakeClient(), build_questions(StrategyParams())
    out = s.score({"price": 1.0})
    assert out.direction["up"] == .58 and out.buying_pressure == .71
    assert abs(out.setup_quality - (0.05 * 1 + .1 * 2 + .5 * 3 + .3 * 4) / 4) < 1e-9
    assert out.input_tokens == 812
    state, questions = s._client.sent
    assert state["snapshot"] == {"price": 1.0} and set(questions) == set(RAW["answers"])

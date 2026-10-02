import pytest

from jevbot import golive, nightly
from jevbot.backtest import run_backtest
from jevbot.jev_layer import ProxyScorer, build_questions
from jevbot.store import Store
from jevbot.strategy import StrategyParams
from tests.conftest import make_candles
from tests.test_backtest import P


def test_proposal_cannot_touch_hard_limits():
    with pytest.raises(ValueError):
        nightly.apply_proposal(StrategyParams(), {"param_changes": {"max_position_usd": 1e9}})
    assert not (golive.HARD_LIMIT_KEYS & set(nightly.TUNABLE))


def test_proposal_is_clamped_and_weights_renormalized():
    p = nightly.apply_proposal(StrategyParams(), {
        "param_changes": {"min_p_direction": 0.01, "w_direction": 2, "w_pressure": 1, "w_setup": 1},
        "new_rules": ["flatten before scheduled macro"], "question_text": {"regime": "New wording?"}})
    assert p.min_p_direction == 0.4
    assert p.w_direction + p.w_pressure + p.w_setup == pytest.approx(1)
    assert "flatten before scheduled macro" in p.rules
    assert build_questions(p)["regime"]["instructions"] == "New wording?"
    assert set(build_questions(p)["regime"]["criteria"]) == {"trending_up", "trending_down", "ranging", "volatile_chop"}


def test_nightly_rejects_candidate_that_fails_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly, "HISTORY", tmp_path / "hist")
    sp, md = tmp_path / "strategy.json", tmp_path / "strategy.md"
    P.save(sp)
    before = sp.read_text()
    out = nightly.run_nightly(Store(tmp_path / "db"), make_candles(1500, tf_min=240), sp, md,
                              proposer=lambda *_: {"param_changes": {"stop_bps": 30},
                                                   "new_rules": ["x"], "question_text": {}})
    assert not out["accepted"] and sp.read_text() == before and not md.exists()


def test_nightly_survives_garbage_proposal(tmp_path):
    sp = tmp_path / "s.json"
    out = nightly.run_nightly(Store(tmp_path / "db"), make_candles(600, tf_min=240), sp, tmp_path / "m",
                              proposer=lambda *_: {"param_changes": {"kelly_cap": 1}})
    assert not out["accepted"] and "not tunable" in out["reason"]


def test_go_live_refuses_on_fresh_install(tmp_path):
    bt = run_backtest(make_candles(1500, tf_min=240), P, ProxyScorer())
    clean, text = golive.run_all(Store(tmp_path / "db"), bt)
    assert not clean
    assert "WHAT COULD BLOW UP THIS ACCOUNT?" in text and "REFUSING TO GO LIVE" in text


def test_limits_not_delegated_to_models():
    assert golive.check_limits_in_code().clean

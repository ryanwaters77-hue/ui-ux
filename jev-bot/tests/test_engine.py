import pytest

from jevbot.broker import SimBroker
from jevbot.config import HardLimits
from jevbot.engine import Engine
from jevbot.outcomes import calibration_reports, resolve_pending
from jevbot.risk import RiskManager
from jevbot.store import Store
from jevbot.strategy import StrategyParams
from tests.conftest import make_candles
from tests.test_decision_calibration import scores


class Fixed:
    def __init__(self, s): self.s, self.calls = s, 0
    def score(self, state):
        self.calls += 1
        assert set(state) >= {"price", "trend_bps"} and "headline" not in state
        return self.s


class Boom:
    def score(self, state): raise TimeoutError("jev slow")


def make(tmp_path, scorer, limits=HardLimits(max_position_usd=400, manual_approval_above_usd=1000), cash=10_000):
    b = SimBroker(cash)
    rm = RiskManager(limits, tmp_path / "KILL", cash)
    e = Engine(broker=b, scorer=scorer, params=StrategyParams(), limits=limits, risk=rm,
               store=Store(tmp_path / "t.db"), symbol="TEST")
    return e, b, rm


def test_enters_and_logs(tmp_path):
    e, b, _ = make(tmp_path, Fixed(scores()))
    out = e.on_candle(make_candles(200))
    assert out["action"] == "long" and 0 < out["size"] <= 400
    assert b.qty > 0
    assert e.store.query("SELECT count(*) n FROM signals")[0]["n"] == 1
    assert e.store.query("SELECT count(*) n FROM predictions")[0]["n"] == 5


def test_jev_failure_means_no_trade(tmp_path):
    e, b, _ = make(tmp_path, Boom())
    assert e.on_candle(make_candles(200))["reason"] == "jev_error"
    assert b.qty == 0


def test_kill_file_flattens_and_halts(tmp_path):
    e, b, rm = make(tmp_path, Fixed(scores()))
    c = make_candles(300)
    e.on_candle(c.iloc[:200])
    assert b.qty > 0
    rm.kill("manual")
    out = e.on_candle(c.iloc[:201])
    assert out["action"] == "halted" and b.qty == 0
    calls = e.scorer.calls
    e.on_candle(c.iloc[:202])
    assert e.scorer.calls == calls  # Jev isn't even consulted while halted


def test_big_order_escalates_instead_of_trading(tmp_path):
    lim = HardLimits(max_position_usd=5000, manual_approval_above_usd=100)
    e, b, _ = make(tmp_path, Fixed(scores()), limits=lim)
    out = e.on_candle(make_candles(200))
    assert out["action"] == "flat" and "approval" in out["verdict"] and b.qty == 0


def test_daily_loss_limit_kills(tmp_path):
    lim = HardLimits(daily_loss_limit_usd=50, max_position_usd=400, manual_approval_above_usd=1000)
    e, b, rm = make(tmp_path, Fixed(scores()), limits=lim)
    c = make_candles(300)
    e.on_candle(c.iloc[:200])
    b.cash -= 100  # simulate a loss
    assert e.on_candle(c.iloc[:201])["action"] == "halted"
    assert rm.killed and b.qty == 0


def test_outcomes_resolve_for_calibration(tmp_path):
    e, _, _ = make(tmp_path, Fixed(scores()))
    c = make_candles(300)
    for i in range(200, 260):
        e.on_candle(c.iloc[:i])
    assert resolve_pending(e.store, c, e.p) > 0
    reps = calibration_reports(e.store)
    assert {r.question for r in reps} >= {"buying_pressure", "direction:up"}
    assert all(0 <= r.brier <= 1 for r in reps)

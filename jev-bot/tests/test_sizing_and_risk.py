from datetime import date

import pytest

from jevbot.config import HardLimits
from jevbot.risk import RiskManager
from jevbot.sizing import kelly_fraction, size_trade

L = HardLimits(max_position_usd=1000, daily_loss_limit_usd=100, max_drawdown_pct=10,
               manual_approval_above_usd=500, kelly_cap=0.25)
BASE = dict(payoff_ratio=2.0, stop_bps=60, equity=10_000, limits=L, p_cutoff=0.55,
            calibration_verified=True, live=False)


def test_kelly_formula():
    assert kelly_fraction(0.5, 2.0) == pytest.approx(0.25)
    assert kelly_fraction(0.3, 2.0) < 0


def test_zero_below_cutoff():
    assert size_trade(p_win=0.54, confidence=0.9, **BASE).notional_usd == 0


def test_never_exceeds_quarter_kelly_or_max_position():
    r = size_trade(p_win=0.99, confidence=0.99, **BASE)
    assert r.kelly_used <= 0.25
    assert r.notional_usd <= L.max_position_usd


def test_higher_confidence_sizes_larger():
    lo = size_trade(p_win=0.6, confidence=0.5, **{**BASE, "equity": 1e6, "stop_bps": 1000, "limits": HardLimits(max_position_usd=1e9)})
    hi = size_trade(p_win=0.6, confidence=0.95, **{**BASE, "equity": 1e6, "stop_bps": 1000, "limits": HardLimits(max_position_usd=1e9)})
    assert hi.notional_usd > lo.notional_usd > 0


def test_live_blocked_without_calibration():
    r = size_trade(p_win=0.8, confidence=0.9, **{**BASE, "live": True, "calibration_verified": False})
    assert r.notional_usd == 0 and "calibration" in r.reason


def test_limits_validate():
    with pytest.raises(ValueError):
        HardLimits(kelly_cap=0.5)


def test_daily_loss_trips_kill(tmp_path):
    rm = RiskManager(L, tmp_path / "KILL", 10_000)
    assert rm.update_equity(9_950, date(2026, 1, 1)) is None
    assert "daily loss" in rm.update_equity(9_899, date(2026, 1, 1))
    assert rm.killed
    assert not rm.check_order(notional_usd=10, current_position_usd=0, reduces_risk=False).allowed
    assert rm.check_order(notional_usd=10, current_position_usd=0, reduces_risk=True).allowed


def test_drawdown_trips_kill(tmp_path):
    rm = RiskManager(HardLimits(daily_loss_limit_usd=1e9), tmp_path / "KILL", 10_000)
    rm.update_equity(12_000, date(2026, 1, 1))
    rm.update_equity(11_000, date(2026, 1, 2))
    assert rm.update_equity(10_700, date(2026, 1, 3)).startswith("drawdown")


def test_order_checks(tmp_path):
    rm = RiskManager(L, tmp_path / "KILL", 10_000)
    assert rm.check_order(notional_usd=400, current_position_usd=0, reduces_risk=False).allowed
    v = rm.check_order(notional_usd=600, current_position_usd=0, reduces_risk=False)
    assert not v.allowed and v.needs_approval
    assert not rm.check_order(notional_usd=400, current_position_usd=700, reduces_risk=False).allowed

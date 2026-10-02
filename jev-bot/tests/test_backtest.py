from jevbot.backtest import Costs, run_backtest
from jevbot.jev_layer import ProxyScorer
from jevbot.strategy import StrategyParams
from tests.conftest import make_candles

P = StrategyParams(timeframe_min=240, min_p_regime=.3, min_p_direction=.4, min_p_buying_pressure=.5,
                   min_setup_quality=.3, max_p_risk_extreme=.5, p_cutoff=.4)


def test_random_walk_fails_the_gate():
    c = make_candles(n=6600, tf_min=240, seed=7)   # ~3 years of 4h bars, no edge
    r = run_backtest(c, P, ProxyScorer())
    assert r.span_days > 730
    assert r.in_sample.trades > 0
    assert not r.passed and r.gate_failures


def test_costs_are_charged():
    c = make_candles(n=1500, tf_min=240, seed=3)
    free = run_backtest(c, P, ProxyScorer(), costs=Costs(0, 0))
    paid = run_backtest(c, P, ProxyScorer(), costs=Costs(25, 5))
    assert len(free.trades) == len(paid.trades) > 0
    for f, p in zip(free.trades, paid.trades):
        assert p.net_bps == __import__("pytest").approx(f.net_bps - 60, abs=1e-6)


def test_short_data_fails_span_gate():
    c = make_candles(n=600, tf_min=240, seed=1)
    r = run_backtest(c, P, ProxyScorer())
    assert any("data span" in m for m in r.gate_failures)

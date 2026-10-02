import pandas as pd
import pytest

from jevbot.state_engine import BookTop, LookaheadError, build_snapshot
from tests.conftest import make_candles

EPS = pd.Timedelta(microseconds=1)


def test_snapshot_is_numeric_and_compact(candles):
    s = build_snapshot(candles, candles.ts.iloc[-1] + EPS).to_state()
    assert set(s) >= {"price", "spread_bps", "book_imbalance", "realized_vol_pct", "trend_bps", "order_flow"}
    assert all(isinstance(v, float) for k, v in s.items() if k != "ts")


def test_rejects_candle_at_decision_time(candles):
    with pytest.raises(LookaheadError):
        build_snapshot(candles, candles.ts.iloc[-1])


def test_rejects_future_book(candles):
    t = candles.ts.iloc[-1] + EPS
    with pytest.raises(LookaheadError):
        build_snapshot(candles, t, BookTop(t, 99, 101, 1, 1))


def test_future_data_cannot_change_snapshot():
    a = make_candles(400, seed=1)
    b = a.copy()
    b.loc[300:, "close"] *= 3  # mutate the "future"
    t = a.ts.iloc[299] + EPS
    assert build_snapshot(a.iloc[:300], t) == build_snapshot(b.iloc[:300], t)


def test_book_features():
    c = make_candles(100)
    t = c.ts.iloc[-1] + EPS
    s = build_snapshot(c, t, BookTop(c.ts.iloc[-1], 99.99, 100.01, 30, 10))
    assert s.spread_bps == pytest.approx(2.0, rel=1e-3)
    assert s.book_imbalance == pytest.approx(0.5)


def test_needs_history():
    c = make_candles(20)
    with pytest.raises(ValueError):
        build_snapshot(c, c.ts.iloc[-1] + EPS)

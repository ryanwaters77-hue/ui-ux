import numpy as np
import pandas as pd
import pytest


def make_candles(n=400, drift=0.0, vol=0.002, seed=0, start="2024-01-01", tf_min=15):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[100, close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    ts = pd.date_range(start, periods=n, freq=f"{tf_min}min", tz="UTC") + pd.Timedelta(minutes=tf_min)
    return pd.DataFrame({"ts": ts, "open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(50, 150, n)})


@pytest.fixture
def candles():
    return make_candles()

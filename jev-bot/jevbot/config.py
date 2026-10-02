"""Hard limits and settings.

Every limit here is enforced by deterministic code. Models (Jev, Opus) never read or
write this module's values; strategy.json can tighten thresholds but cannot loosen these.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _f(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


@dataclass(frozen=True)
class HardLimits:
    max_position_usd: float = 1000.0
    daily_loss_limit_usd: float = 100.0
    max_drawdown_pct: float = 10.0
    manual_approval_above_usd: float = 500.0
    kelly_cap: float = 0.25  # never more than quarter Kelly

    def __post_init__(self) -> None:
        if not 0 < self.kelly_cap <= 0.25:
            raise ValueError("kelly_cap must be in (0, 0.25]")
        for k in ("max_position_usd", "daily_loss_limit_usd", "max_drawdown_pct"):
            if getattr(self, k) <= 0:
                raise ValueError(f"{k} must be positive")

    @classmethod
    def from_env(cls) -> "HardLimits":
        return cls(
            max_position_usd=_f("MAX_POSITION_USD", 1000.0),
            daily_loss_limit_usd=_f("DAILY_LOSS_LIMIT_USD", 100.0),
            max_drawdown_pct=_f("MAX_DRAWDOWN_PCT", 10.0),
            manual_approval_above_usd=_f("MANUAL_APPROVAL_ABOVE_USD", 500.0),
            kelly_cap=min(_f("KELLY_CAP", 0.25), 0.25),
        )


@dataclass(frozen=True)
class Settings:
    symbol: str = "BTC/USD"
    timeframe_min: int = 15
    live_trading: bool = False
    data_dir: Path = ROOT / "data"
    db_path: Path = ROOT / "data" / "bot.db"
    kill_file: Path = ROOT / "KILL"
    strategy_json: Path = ROOT / "strategy.json"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            symbol=os.getenv("SYMBOL", "BTC/USD"),
            timeframe_min=int(os.getenv("TIMEFRAME_MIN", "15")),
            live_trading=os.getenv("LIVE_TRADING", "0") == "1",
        )

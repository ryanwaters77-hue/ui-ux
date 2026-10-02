"""Exchange connectivity. Alpaca paper by default; live only behind the go-live gate.

API keys come from .env and need trade permission only. Alpaca API keys can't withdraw
funds. This module never asks for, stores or types passwords or 2FA codes.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd

from .config import ROOT
from .state_engine import BookTop

GO_LIVE_FILE = ROOT / "data" / "go_live_approved.json"


class Broker(Protocol):
    def equity(self) -> float: ...
    def position_usd(self) -> float: ...
    def buy(self, notional_usd: float) -> dict: ...
    def sell_all(self, reason: str) -> dict: ...
    def flatten_everything(self) -> None: ...


def live_allowed() -> tuple[bool, str]:
    if os.getenv("LIVE_TRADING", "0") != "1":
        return False, "LIVE_TRADING != 1"
    if not GO_LIVE_FILE.exists():
        return False, "go-live check has not been passed and signed (run scripts/go_live_check.py)"
    rec = json.loads(GO_LIVE_FILE.read_text())
    if not rec.get("all_clean") or not rec.get("human_confirmed"):
        return False, "go-live record is not clean and human-confirmed"
    return True, "ok"


class AlpacaBroker:
    def __init__(self, symbol: str, timeframe_min: int):
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.trading.client import TradingClient

        live, why = live_allowed()
        self.paper = not live
        key, secret = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
        self.trading = TradingClient(key, secret, paper=self.paper)
        self.data = CryptoHistoricalDataClient(key, secret)
        self.symbol = symbol
        self.pos_symbol = symbol.replace("/", "")
        self.tf = timeframe_min
        self.mode_reason = "paper" if self.paper else "LIVE"
        if self.paper and os.getenv("LIVE_TRADING") == "1":
            self.mode_reason = f"paper (live refused: {why})"

    # -- data ---------------------------------------------------------------
    def closed_bars(self, lookback: int = 300) -> pd.DataFrame:
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        now = datetime.now(timezone.utc)
        req = CryptoBarsRequest(symbol_or_symbols=self.symbol,
                                timeframe=TimeFrame(self.tf, TimeFrameUnit.Minute),
                                start=now - timedelta(minutes=self.tf * (lookback + 5)))
        df = self.data.get_crypto_bars(req).df.reset_index()
        # Alpaca stamps bars with their START; convert to close time and drop the open bar.
        df["ts"] = pd.to_datetime(df["timestamp"], utc=True) + pd.Timedelta(minutes=self.tf)
        df = df[df["ts"] <= now]
        return df[["ts", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    def book(self) -> BookTop | None:
        from alpaca.data.requests import CryptoLatestOrderbookRequest

        ob = self.data.get_crypto_latest_orderbook(
            CryptoLatestOrderbookRequest(symbol_or_symbols=self.symbol))[self.symbol]
        if not ob.bids or not ob.asks:
            return None
        return BookTop(pd.Timestamp(ob.timestamp), ob.bids[0].p, ob.asks[0].p,
                       sum(b.s for b in ob.bids[:10]), sum(a.s for a in ob.asks[:10]))

    # -- account --------------------------------------------------------------
    def equity(self) -> float:
        return float(self.trading.get_account().equity)

    def position_usd(self) -> float:
        try:
            return float(self.trading.get_open_position(self.pos_symbol).market_value)
        except Exception:
            return 0.0

    def avg_entry(self) -> float:
        try:
            return float(self.trading.get_open_position(self.pos_symbol).avg_entry_price)
        except Exception:
            return 0.0

    def position_qty(self) -> float:
        try:
            return float(self.trading.get_open_position(self.pos_symbol).qty)
        except Exception:
            return 0.0

    # -- orders ---------------------------------------------------------------
    def buy(self, notional_usd: float) -> dict:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        o = self.trading.submit_order(MarketOrderRequest(
            symbol=self.symbol, notional=round(notional_usd, 2),
            side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
        return {"id": str(o.id), "status": str(o.status)}

    def sell_all(self, reason: str) -> dict:
        try:
            o = self.trading.close_position(self.pos_symbol)
            return {"id": str(getattr(o, "id", "")), "status": "closing", "reason": reason}
        except Exception as e:  # no position
            return {"status": "noop", "reason": f"{reason}: {e}"}

    def flatten_everything(self) -> None:
        self.trading.cancel_orders()
        self.trading.close_all_positions(cancel_orders=True)


class SimBroker:
    """In-process paper broker for dry runs and tests. Fills at the given mark plus costs."""

    def __init__(self, cash: float = 10_000.0, cost_bps: float = 30.0):
        self.cash, self.qty, self.mark, self.cost = cash, 0.0, 0.0, cost_bps / 1e4
        self.entry = 0.0

    def set_mark(self, price: float) -> None:
        self.mark = price

    def equity(self) -> float:
        return self.cash + self.qty * self.mark

    def position_usd(self) -> float:
        return self.qty * self.mark

    def buy(self, notional_usd: float) -> dict:
        px = self.mark * (1 + self.cost)
        q = notional_usd / px
        self.entry = (self.entry * self.qty + px * q) / (self.qty + q)
        self.qty += q
        self.cash -= notional_usd
        return {"status": "filled", "price": px, "qty": q}

    def sell_all(self, reason: str) -> dict:
        if self.qty == 0:
            return {"status": "noop"}
        px = self.mark * (1 - self.cost)
        q, self.qty = self.qty, 0.0
        self.cash += q * px
        return {"status": "filled", "price": px, "qty": q,
                "pnl_bps": (px / self.entry - 1) * 1e4, "reason": reason}

    def flatten_everything(self) -> None:
        self.sell_all("flatten")

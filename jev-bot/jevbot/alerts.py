"""Telegram alerts via a BotFather bot. Failures to alert never stop trading logic."""
from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger(__name__)


def send(kind: str, text: str) -> bool:
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log.info("[alert:%s] %s", kind, text)
        return False
    icon = {"fill": "🟢", "error": "🔴", "escalation": "🟠", "kill": "🛑", "report": "📊"}.get(kind, "•")
    try:
        r = httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                       json={"chat_id": chat, "text": f"{icon} {kind.upper()}\n{text}"}, timeout=5)
        return r.status_code == 200
    except Exception as e:  # never log the token-bearing URL
        log.warning("telegram send failed: %s", type(e).__name__)
        return False

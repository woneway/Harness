"""TelegramNotifier — 通过 Telegram Bot API 发送通知。"""

from __future__ import annotations

import httpx

from harness.notifier.base import AbstractNotifier


class TelegramNotifier(AbstractNotifier):
    """使用 httpx 异步发送 Telegram 消息。

    成功时消息前缀 ✅，失败时前缀 ❌。

    Args:
        bot_token: Telegram Bot Token。
        chat_id: 目标 Chat ID（用户或群组）。
    """

    _API_BASE = "https://api.telegram.org"

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id

    async def notify(
        self,
        title: str,
        body: str,
        *,
        success: bool,
    ) -> None:
        """发送 Telegram 消息。"""
        icon = "✅" if success else "❌"
        text = f"{icon} *{title}*\n\n{body}"

        url = f"{self._API_BASE}/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10.0)
            response.raise_for_status()

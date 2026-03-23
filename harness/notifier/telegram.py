"""TelegramNotifier — 通过 Telegram Bot API 发送通知。"""

from __future__ import annotations

import html

import httpx

from harness.notifier.base import AbstractNotifier


class TelegramNotifier(AbstractNotifier):
    """使用 httpx 异步发送 Telegram 消息。

    成功时消息前缀 ✅，失败时前缀 ❌。
    使用 HTML parse_mode 以避免 Markdown 特殊字符导致解析错误。

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
        safe_title = html.escape(title)
        safe_body = html.escape(body)
        text = f"{icon} <b>{safe_title}</b>\n\n{safe_body}"

        url = f"{self._API_BASE}/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10.0)
            response.raise_for_status()

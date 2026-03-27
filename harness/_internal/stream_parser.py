"""StreamParser — 逐行解析 Claude Code CLI stream-json 输出。"""

from __future__ import annotations

import json
from typing import Callable


class StreamParser:
    """逐行解析 Claude Code CLI 的 stream-json 输出。

    - ``result`` event → 提取最终文本和 tokens_used
    - partial message text → 回调 stream_callback 或 raw_stream_callback
    """

    def __init__(
        self,
        stream_callback: Callable[[str], None] | None = None,
        raw_stream_callback: Callable[[dict], None] | None = None,
    ) -> None:
        self._stream_callback = stream_callback
        self._raw_stream_callback = raw_stream_callback

        self.final_text: str | None = None
        self.tokens_used: int = 0
        self.session_id: str | None = None

        # 兜底：追踪最后一个 assistant 消息的文本，
        # 当 result 事件 text 为空时（如 Claude 使用工具后未产生纯文本 result）用作回退。
        self._last_assistant_text: str | None = None

    def feed(self, line: str) -> None:
        """处理一行 JSON 输入。"""
        line = line.strip()
        if not line:
            return

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # 非 JSON 行（如启动信息）忽略
            return

        if not isinstance(event, dict):
            return

        event_type = event.get("type")

        # 原始回调（返回所有 event）
        if self._raw_stream_callback is not None:
            try:
                self._raw_stream_callback(event)
            except Exception:
                pass

        # result event：提取最终文本和 token 用量
        if event_type == "result":
            self.final_text = event.get("result", "")
            usage = event.get("usage") or {}
            self.tokens_used = usage.get("output_tokens", 0)
            if session_id := event.get("session_id"):
                self.session_id = session_id
            return

        # stream_event 格式（Claude CLI --verbose 模式）
        if event_type == "stream_event":
            inner = event.get("event", {})
            inner_type = inner.get("type")

            if inner_type == "content_block_delta":
                delta = inner.get("delta", {})
                if delta.get("type") == "text":
                    text = delta.get("text", "")
                    if text and self._stream_callback is not None:
                        try:
                            self._stream_callback(text)
                        except Exception:
                            pass

            elif inner_type == "message_start":
                msg = inner.get("message", {})
                if session_id := msg.get("session"):
                    self.session_id = session_id

            return

        # assistant message 中的文本片段（partial messages）
        if event_type == "assistant":
            message = event.get("message", {})
            content = message.get("content", [])
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        text_parts.append(text)
                        if self._stream_callback is not None:
                            try:
                                self._stream_callback(text)
                            except Exception:
                                pass
            # 记录最后一个 assistant 消息的完整文本（兜底用）
            if text_parts:
                self._last_assistant_text = "".join(text_parts)

        # system init event：提取 session_id
        if event_type == "system" and event.get("subtype") == "init":
            if session_id := event.get("session_id"):
                self.session_id = session_id

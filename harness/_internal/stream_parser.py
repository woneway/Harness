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

        # assistant message 中的文本片段（partial messages）
        if event_type == "assistant":
            message = event.get("message", {})
            content = message.get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text and self._stream_callback is not None:
                        try:
                            self._stream_callback(text)
                        except Exception:
                            pass

        # system init event：提取 session_id
        if event_type == "system" and event.get("subtype") == "init":
            if session_id := event.get("session_id"):
                self.session_id = session_id

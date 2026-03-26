"""HTTP runner 共享工具函数。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx


def safe_schema_name(title: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in title)


def raise_for_status(resp: httpx.Response, service: str) -> None:
    """检查响应状态码，对流式响应也能正常工作。"""
    if resp.status_code >= 400:
        # 对于流式响应，.text 访问未读取的响应体会抛 ResponseNotRead，
        # 用 try/except 兜底，确保始终抛出 RuntimeError。
        try:
            detail = resp.text
        except Exception:
            detail = f"HTTP {resp.status_code}"
        raise RuntimeError(f"{service} API error {resp.status_code}: {detail}")


async def iter_sse_events(lines: AsyncIterator[str]) -> AsyncIterator[dict]:
    """从 SSE 行流中解析 JSON 事件，跳过非 data 行和 [DONE] 标记。

    用于 OpenAI / Anthropic 等 SSE 流式 API 的共享解析逻辑。
    """
    async for line in lines:
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            yield json.loads(data_str)
        except json.JSONDecodeError:
            continue

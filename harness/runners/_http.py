"""HTTP runner 共享工具函数。"""

from __future__ import annotations

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

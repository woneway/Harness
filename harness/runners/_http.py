"""HTTP runner 共享工具函数。"""

from __future__ import annotations

import httpx


def safe_schema_name(title: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in title)


def raise_for_status(resp: httpx.Response, service: str) -> None:
    if resp.status_code >= 400:
        raise RuntimeError(f"{service} API error {resp.status_code}: {resp.text}")

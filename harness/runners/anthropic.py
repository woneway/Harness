"""AnthropicRunner — 直接调用 Anthropic Messages API 的 runner。

不走 Claude CLI 子进程，适合需要精确控制模型、token、计费的场景。

依赖：httpx（已在 harness 依赖中）

配置优先级（高 → 低）：
  1. 构造函数直接传值（api_key=、model=）
  2. 系统环境变量：ANTHROPIC_API_KEY、ANTHROPIC_MODEL
  3. 内置默认值（model="claude-opus-4-6"）
"""

from __future__ import annotations

import json
import os
from typing import Callable

import httpx

from harness.runners.base import AbstractRunner, RunnerResult

_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MODEL = "claude-opus-4-6"
_DEFAULT_MAX_TOKENS = 8192
_HTTP_TIMEOUT = 300.0


class AnthropicRunner(AbstractRunner):
    """直接调用 Anthropic Messages API（非 Claude CLI）。

    通过 ``*_env`` 参数指定读取哪个环境变量。

    Args:
        api_key: API 密钥，直接传值优先于环境变量。
        model: 模型 ID，直接传值优先于环境变量。
        max_tokens: 最大输出 token 数，默认 8192。
        api_key_env: 读取 API 密钥的环境变量名，默认 ``ANTHROPIC_API_KEY``。
        model_env: 读取模型名的环境变量名，默认 ``ANTHROPIC_MODEL``。

    Examples:
        # 读 ANTHROPIC_API_KEY
        runner = AnthropicRunner()

        # 自定义变量名
        runner = AnthropicRunner(api_key_env="MY_CLAUDE_KEY", model_env="MY_CLAUDE_MODEL")
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        api_key_env: str = "ANTHROPIC_API_KEY",
        model_env: str = "ANTHROPIC_MODEL",
    ) -> None:
        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key:
            raise ValueError(
                f"Anthropic API key is required. "
                f"Pass api_key= or set the {api_key_env!r} environment variable."
            )
        self.api_key = resolved_key
        self.model = model or os.environ.get(model_env) or _DEFAULT_MODEL
        self.max_tokens = max_tokens

    async def execute(
        self,
        prompt: str,
        *,
        system_prompt: str,
        session_id: str | None,
        output_schema_json: str | None = None,
        stream_callback: Callable[[str], None] | None = None,
        raw_stream_callback: Callable[[dict], None] | None = None,
        **kwargs: object,
    ) -> RunnerResult:
        """调用 Anthropic Messages API。

        - 有 output_schema_json：使用 tool_use 强制结构化输出（非流式）。
        - 无 output_schema_json 且有 stream_callback：流式返回文本。
        - 其他：非流式文本返回。

        session_id 被忽略（REST API 无状态）。
        """
        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        if output_schema_json is not None:
            return await self._complete_with_schema(payload, headers, output_schema_json)

        if stream_callback is not None:
            return await self._stream(payload, headers, stream_callback)

        return await self._complete(payload, headers)

    async def _complete(self, payload: dict, headers: dict) -> RunnerResult:
        """非流式纯文本调用。"""
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(_API_URL, json=payload, headers=headers)
            _raise_for_status(resp)
            data = resp.json()

        text = _extract_text(data)
        tokens = _count_tokens(data)
        return RunnerResult(text=text, tokens_used=tokens, session_id=None)

    async def _complete_with_schema(
        self,
        payload: dict,
        headers: dict,
        output_schema_json: str,
    ) -> RunnerResult:
        """使用 tool_use 强制返回符合 JSON schema 的结构化输出。"""
        schema = json.loads(output_schema_json)
        tool_name = _safe_name(schema.get("title", "response"))

        schema_payload = {
            **payload,
            "tools": [
                {
                    "name": tool_name,
                    "description": "Respond with structured data matching the schema.",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": tool_name},
        }

        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(_API_URL, json=schema_payload, headers=headers)
            _raise_for_status(resp)
            data = resp.json()

        # 从 tool_use content block 提取 JSON
        for block in data.get("content", []):
            if block.get("type") == "tool_use":
                text = json.dumps(block.get("input", {}))
                tokens = _count_tokens(data)
                return RunnerResult(text=text, tokens_used=tokens, session_id=None)

        # Fallback：未返回 tool_use（不应发生，但做保底）
        text = _extract_text(data)
        return RunnerResult(text=text, tokens_used=_count_tokens(data), session_id=None)

    async def _stream(
        self,
        payload: dict,
        headers: dict,
        stream_callback: Callable[[str], None],
    ) -> RunnerResult:
        """流式调用，逐 text_delta 回调。"""
        stream_payload = {**payload, "stream": True}
        full_text = ""
        input_tokens = 0
        output_tokens = 0

        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            async with client.stream("POST", _API_URL, json=stream_payload, headers=headers) as resp:
                _raise_for_status(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")

                    if etype == "message_start":
                        usage = event.get("message", {}).get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)

                    elif etype == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                full_text += text
                                try:
                                    stream_callback(text)
                                except Exception:
                                    pass

                    elif etype == "message_delta":
                        usage = event.get("usage", {})
                        output_tokens = usage.get("output_tokens", 0)

        return RunnerResult(
            text=full_text,
            tokens_used=input_tokens + output_tokens,
            session_id=None,
        )


def _extract_text(data: dict) -> str:
    """从非流式响应中提取文本内容。"""
    return "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    )


def _count_tokens(data: dict) -> int:
    usage = data.get("usage", {})
    return usage.get("input_tokens", 0) + usage.get("output_tokens", 0)


def _safe_name(title: str) -> str:
    """将 schema title 转为合法 tool name（仅字母数字下划线）。"""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in title)


def _raise_for_status(resp: httpx.Response) -> None:
    """抛出包含响应体的可读错误。"""
    if resp.status_code >= 400:
        try:
            body = resp.text
        except Exception:
            body = "<unreadable>"
        raise RuntimeError(
            f"Anthropic API error {resp.status_code}: {body}"
        )

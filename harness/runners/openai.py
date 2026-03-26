"""OpenAIRunner — 调用 OpenAI-compatible API 的 runner。

支持任何兼容 OpenAI Chat Completions API 的服务（OpenAI、MiniMax、DeepSeek 等）。

依赖：httpx（已在 harness 依赖中）

配置优先级（高 → 低）：
  1. 构造函数直接传值（api_key=、model=、base_url=）
  2. *_env 参数指向的环境变量（默认 OPENAI_API_KEY、OPENAI_MODEL、OPENAI_BASE_URL）
  3. 内置默认值（model="gpt-4o"、base_url=OpenAI 官方）
"""

from __future__ import annotations

import json
import os
from typing import Callable

import httpx

from harness.runners._http import iter_sse_events, raise_for_status, safe_schema_name
from harness.runners.base import AbstractRunner, RunnerResult

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o"
_HTTP_TIMEOUT = 300.0


class OpenAIRunner(AbstractRunner):
    """调用 OpenAI-compatible Chat Completions API。

    通过 ``*_env`` 参数指定读取哪个环境变量，不同项目可用不同变量名互不干扰。

    Args:
        api_key: API 密钥，直接传值优先于环境变量。
        model: 模型名称，直接传值优先于环境变量。
        base_url: API 基础 URL，直接传值优先于环境变量。
        api_key_env: 读取 API 密钥的环境变量名，默认 ``OPENAI_API_KEY``。
        model_env: 读取模型名的环境变量名，默认 ``OPENAI_MODEL``。
        base_url_env: 读取 base URL 的环境变量名，默认 ``OPENAI_BASE_URL``。

    Examples:
        # OpenAI（读 OPENAI_API_KEY）
        runner = OpenAIRunner()

        # MiniMax（读 MINIMAX_API_KEY，与 OpenAI key 互不干扰）
        runner = OpenAIRunner(
            api_key_env="MINIMAX_API_KEY",
            base_url_env="MINIMAX_BASE_URL",
            model_env="MINIMAX_MODEL",
        )
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        model_env: str = "OPENAI_MODEL",
        base_url_env: str = "OPENAI_BASE_URL",
    ) -> None:
        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key:
            raise ValueError(
                f"OpenAI API key is required. "
                f"Pass api_key= or set the {api_key_env!r} environment variable."
            )
        self.api_key = resolved_key
        self.model = model or os.environ.get(model_env) or _DEFAULT_MODEL
        raw_url = base_url or os.environ.get(base_url_env) or _DEFAULT_BASE_URL
        self.base_url = raw_url.rstrip("/")

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
        """调用 OpenAI-compatible API。

        - 有 output_schema_json：非流式，使用 response_format structured output。
        - 无 output_schema_json 且有 stream_callback：流式返回。
        - 其他：非流式。

        session_id 被忽略（REST API 无状态）。
        """
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {"model": self.model, "messages": messages}

        if output_schema_json is not None:
            schema = json.loads(output_schema_json)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": safe_schema_name(schema.get("title", "response")),
                    "schema": schema,
                    "strict": True,
                },
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        use_stream = stream_callback is not None and output_schema_json is None

        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            if use_stream:
                return await self._stream(client, url, headers, payload, stream_callback)
            return await self._complete(client, url, headers, payload)

    async def _complete(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict,
        payload: dict,
    ) -> RunnerResult:
        resp = await client.post(url, json=payload, headers=headers)
        raise_for_status(resp, "OpenAI")
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return RunnerResult(text=text, tokens_used=tokens, session_id=None)

    async def _stream(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict,
        payload: dict,
        stream_callback: Callable[[str], None],
    ) -> RunnerResult:
        stream_payload = {
            **payload,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        full_text = ""
        tokens_used = 0

        async with client.stream("POST", url, json=stream_payload, headers=headers) as resp:
            raise_for_status(resp, "OpenAI")
            async for chunk in iter_sse_events(resp.aiter_lines()):
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content") or ""
                    if content:
                        full_text += content
                        stream_callback(content)
                if chunk.get("usage"):
                    tokens_used = chunk["usage"].get("total_tokens", 0)

        return RunnerResult(text=full_text, tokens_used=tokens_used, session_id=None)



"""tests/unit/test_openai_runner.py — OpenAIRunner 单元测试（mock httpx）。"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.runners.openai import OpenAIRunner, _safe_name


# ---------------------------------------------------------------------------
# 测试：env var 配置
# ---------------------------------------------------------------------------


def test_reads_api_key_from_default_env():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}):
        runner = OpenAIRunner()
    assert runner.api_key == "sk-from-env"


def test_reads_model_from_default_env():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-x", "OPENAI_MODEL": "gpt-4-turbo"}):
        runner = OpenAIRunner()
    assert runner.model == "gpt-4-turbo"


def test_reads_base_url_from_default_env():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-x", "OPENAI_BASE_URL": "https://custom.api/v1"}):
        runner = OpenAIRunner()
    assert runner.base_url == "https://custom.api/v1"


def test_custom_env_var_names():
    """不同项目可用不同变量名，避免冲突。"""
    env = {"MINIMAX_API_KEY": "sk-minimax", "MINIMAX_BASE_URL": "https://api.minimax.chat/v1"}
    with patch.dict(os.environ, env):
        runner = OpenAIRunner(api_key_env="MINIMAX_API_KEY", base_url_env="MINIMAX_BASE_URL")
    assert runner.api_key == "sk-minimax"
    assert runner.base_url == "https://api.minimax.chat/v1"


def test_direct_value_overrides_env():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}):
        runner = OpenAIRunner(api_key="sk-direct")
    assert runner.api_key == "sk-direct"


def test_missing_api_key_raises():
    env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIRunner()


def test_missing_api_key_custom_env_name_in_error():
    env_without_key = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
            OpenAIRunner(api_key_env="MINIMAX_API_KEY")


# ---------------------------------------------------------------------------
# 辅助：构造 mock httpx response
# ---------------------------------------------------------------------------


def _make_response(body: dict, status_code: int = 200) -> MagicMock:
    """非流式 response mock。"""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


def _make_stream_lines(chunks: list[str], usage_tokens: int = 15) -> list[str]:
    """构造 OpenAI SSE 流式行。"""
    lines = []
    for content in chunks:
        data = {
            "choices": [{"delta": {"content": content}, "index": 0}],
            "usage": None,
        }
        lines.append(f"data: {json.dumps(data)}")
    # 最后一个 chunk 带 usage
    usage_chunk = {"choices": [], "usage": {"total_tokens": usage_tokens}}
    lines.append(f"data: {json.dumps(usage_chunk)}")
    lines.append("data: [DONE]")
    return lines


# ---------------------------------------------------------------------------
# 测试：非流式完成
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_basic():
    runner = OpenAIRunner(api_key="sk-test", model="gpt-4o")
    body = {
        "choices": [{"message": {"content": "Hello world"}}],
        "usage": {"total_tokens": 20},
    }

    with patch("harness.runners.openai.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_make_response(body))

        result = await runner.execute(
            "Say hello",
            system_prompt="",
            session_id=None,
        )

    assert result.text == "Hello world"
    assert result.tokens_used == 20
    assert result.session_id is None


@pytest.mark.asyncio
async def test_complete_with_system_prompt():
    runner = OpenAIRunner(api_key="sk-test")
    body = {
        "choices": [{"message": {"content": "response"}}],
        "usage": {"total_tokens": 10},
    }

    with patch("harness.runners.openai.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_make_response(body))

        await runner.execute("prompt", system_prompt="You are helpful.", session_id=None)

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are helpful."
        assert payload["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_complete_with_output_schema():
    runner = OpenAIRunner(api_key="sk-test")
    schema = {"title": "MyResult", "type": "object", "properties": {"value": {"type": "string"}}}
    body = {
        "choices": [{"message": {"content": '{"value": "42"}'}}],
        "usage": {"total_tokens": 30},
    }

    with patch("harness.runners.openai.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_make_response(body))

        result = await runner.execute(
            "compute",
            system_prompt="",
            session_id=None,
            output_schema_json=json.dumps(schema),
        )

        call_payload = mock_client.post.call_args[1]["json"]
        assert call_payload["response_format"]["type"] == "json_schema"
        assert call_payload["response_format"]["json_schema"]["name"] == "MyResult"

    assert result.text == '{"value": "42"}'


@pytest.mark.asyncio
async def test_session_id_ignored():
    """session_id 对 REST API 无意义，结果应返回 None。"""
    runner = OpenAIRunner(api_key="sk-test")
    body = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"total_tokens": 5},
    }

    with patch("harness.runners.openai.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_make_response(body))

        result = await runner.execute("x", system_prompt="", session_id="some-session")

    assert result.session_id is None


# ---------------------------------------------------------------------------
# 测试：流式
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_calls_callback():
    runner = OpenAIRunner(api_key="sk-test")
    lines = _make_stream_lines(["Hello", " ", "world"], usage_tokens=12)
    received: list[str] = []

    mock_stream_resp = AsyncMock()
    mock_stream_resp.status_code = 200
    mock_stream_resp.text = ""

    async def _aiter_lines():
        for line in lines:
            yield line

    mock_stream_resp.aiter_lines = _aiter_lines

    with patch("harness.runners.openai.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=stream_ctx)

        result = await runner.execute(
            "say hello",
            system_prompt="",
            session_id=None,
            stream_callback=received.append,
        )

    assert result.text == "Hello world"
    assert received == ["Hello", " ", "world"]
    assert result.tokens_used == 12


@pytest.mark.asyncio
async def test_stream_not_used_when_schema_set():
    """有 output_schema_json 时应走非流式路径，不调用 stream。"""
    runner = OpenAIRunner(api_key="sk-test")
    body = {
        "choices": [{"message": {"content": '{"v": 1}'}}],
        "usage": {"total_tokens": 8},
    }
    received: list[str] = []

    with patch("harness.runners.openai.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_make_response(body))

        await runner.execute(
            "compute",
            system_prompt="",
            session_id=None,
            output_schema_json='{"title": "R", "type": "object"}',
            stream_callback=received.append,
        )

        mock_client.stream.assert_not_called()


# ---------------------------------------------------------------------------
# 测试：错误处理
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_error_raises():
    runner = OpenAIRunner(api_key="sk-test")
    error_body = {"error": {"message": "Invalid API key"}}

    with patch("harness.runners.openai.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_make_response(error_body, status_code=401))

        with pytest.raises(RuntimeError, match="401"):
            await runner.execute("x", system_prompt="", session_id=None)


# ---------------------------------------------------------------------------
# 测试：custom base_url
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_base_url():
    runner = OpenAIRunner(
        api_key="sk-minimax",
        model="abab6.5s-chat",
        base_url="https://api.minimax.chat/v1",
    )
    body = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"total_tokens": 5},
    }

    with patch("harness.runners.openai.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_make_response(body))

        await runner.execute("x", system_prompt="", session_id=None)

        call_args = mock_client.post.call_args
        assert "minimax" in call_args[0][0]


# ---------------------------------------------------------------------------
# 测试：_safe_name
# ---------------------------------------------------------------------------


def test_safe_name_basic():
    assert _safe_name("MyResult") == "MyResult"


def test_safe_name_spaces():
    assert _safe_name("My Result") == "My_Result"


def test_safe_name_special_chars():
    assert _safe_name("my-result.v2") == "my_result_v2"

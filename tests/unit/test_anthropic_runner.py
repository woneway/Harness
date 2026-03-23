"""tests/unit/test_anthropic_runner.py — AnthropicRunner 单元测试（mock httpx）。"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.runners.anthropic import AnthropicRunner


# ---------------------------------------------------------------------------
# 辅助：构造 mock httpx response
# ---------------------------------------------------------------------------


def _make_response(body: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


def _text_response(text: str, input_tokens: int = 10, output_tokens: int = 5) -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _tool_response(tool_name: str, input_data: dict, input_tokens: int = 10, output_tokens: int = 5) -> dict:
    return {
        "content": [{"type": "tool_use", "name": tool_name, "input": input_data}],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _stream_events(chunks: list[str], input_tokens: int = 10, output_tokens: int = 5) -> list[str]:
    """构造 Anthropic SSE 流式事件行。"""
    events = [
        f"data: {json.dumps({'type': 'message_start', 'message': {'usage': {'input_tokens': input_tokens}}})}",
        f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}",
    ]
    for chunk in chunks:
        events.append(
            f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': chunk}})}"
        )
    events.append(f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}")
    events.append(
        f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': output_tokens}})}"
    )
    events.append(f"data: {json.dumps({'type': 'message_stop'})}")
    return events


# ---------------------------------------------------------------------------
# 测试：非流式纯文本
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_basic():
    runner = AnthropicRunner(api_key="sk-ant-test")
    body = _text_response("Hello Claude", input_tokens=10, output_tokens=5)

    with patch("harness.runners.anthropic.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_make_response(body))

        result = await runner.execute("Say hello", system_prompt="", session_id=None)

    assert result.text == "Hello Claude"
    assert result.tokens_used == 15
    assert result.session_id is None


@pytest.mark.asyncio
async def test_complete_with_system_prompt():
    runner = AnthropicRunner(api_key="sk-ant-test")
    body = _text_response("ok")

    with patch("harness.runners.anthropic.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_make_response(body))

        await runner.execute("prompt", system_prompt="Be concise.", session_id=None)

        payload = mock_client.post.call_args[1]["json"]
        assert payload["system"] == "Be concise."


@pytest.mark.asyncio
async def test_no_system_prompt_omitted():
    runner = AnthropicRunner(api_key="sk-ant-test")
    body = _text_response("ok")

    with patch("harness.runners.anthropic.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_make_response(body))

        await runner.execute("prompt", system_prompt="", session_id=None)

        payload = mock_client.post.call_args[1]["json"]
        assert "system" not in payload


@pytest.mark.asyncio
async def test_session_id_ignored():
    runner = AnthropicRunner(api_key="sk-ant-test")
    body = _text_response("ok")

    with patch("harness.runners.anthropic.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_make_response(body))

        result = await runner.execute("x", system_prompt="", session_id="some-session")

    assert result.session_id is None


# ---------------------------------------------------------------------------
# 测试：output_schema（tool_use）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schema_uses_tool_use():
    runner = AnthropicRunner(api_key="sk-ant-test")
    schema = {
        "title": "VideoScript",
        "type": "object",
        "properties": {"scenes": {"type": "array", "items": {"type": "string"}}},
    }
    tool_input = {"scenes": ["scene1", "scene2"]}
    body = _tool_response("VideoScript", tool_input)

    with patch("harness.runners.anthropic.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_make_response(body))

        result = await runner.execute(
            "Generate script",
            system_prompt="",
            session_id=None,
            output_schema_json=json.dumps(schema),
        )

        payload = mock_client.post.call_args[1]["json"]
        assert payload["tool_choice"] == {"type": "tool", "name": "VideoScript"}
        assert payload["tools"][0]["name"] == "VideoScript"

    assert json.loads(result.text) == tool_input


@pytest.mark.asyncio
async def test_schema_no_stream_even_with_callback():
    """有 output_schema 时应走 tool_use 路径，不调用 stream。"""
    runner = AnthropicRunner(api_key="sk-ant-test")
    body = _tool_response("R", {"v": 1})
    received: list[str] = []

    with patch("harness.runners.anthropic.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_make_response(body))

        await runner.execute(
            "x",
            system_prompt="",
            session_id=None,
            output_schema_json='{"title": "R", "type": "object"}',
            stream_callback=received.append,
        )

        mock_client.stream.assert_not_called()


# ---------------------------------------------------------------------------
# 测试：流式
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_calls_callback():
    runner = AnthropicRunner(api_key="sk-ant-test")
    lines = _stream_events(["Hello", " ", "world"], input_tokens=8, output_tokens=4)
    received: list[str] = []

    mock_stream_resp = AsyncMock()
    mock_stream_resp.status_code = 200
    mock_stream_resp.text = ""

    async def _aiter_lines():
        for line in lines:
            yield line

    mock_stream_resp.aiter_lines = _aiter_lines

    with patch("harness.runners.anthropic.httpx.AsyncClient") as mock_client_cls:
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
async def test_stream_payload_has_stream_true():
    runner = AnthropicRunner(api_key="sk-ant-test")
    lines = _stream_events(["ok"])

    mock_stream_resp = AsyncMock()
    mock_stream_resp.status_code = 200
    mock_stream_resp.text = ""

    async def _aiter_lines():
        for line in lines:
            yield line

    mock_stream_resp.aiter_lines = _aiter_lines

    with patch("harness.runners.anthropic.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=stream_ctx)

        await runner.execute("x", system_prompt="", session_id=None, stream_callback=lambda t: None)

        call_args = mock_client.stream.call_args
        payload = call_args[1]["json"]
        assert payload["stream"] is True


# ---------------------------------------------------------------------------
# 测试：错误处理
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_error_raises():
    runner = AnthropicRunner(api_key="sk-ant-test")
    error_body = {"error": {"type": "authentication_error", "message": "Invalid API key"}}

    with patch("harness.runners.anthropic.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_make_response(error_body, status_code=401))

        with pytest.raises(RuntimeError, match="401"):
            await runner.execute("x", system_prompt="", session_id=None)



"""tests/unit/test_stream_parser.py — StreamParser 解析逻辑验证。"""

import json

import pytest

from harness._internal.stream_parser import StreamParser


def _make_event(**kwargs: object) -> str:
    return json.dumps(kwargs)


class TestStreamParserBasic:
    def test_empty_line_ignored(self) -> None:
        parser = StreamParser()
        parser.feed("")
        assert parser.final_text is None
        assert parser.tokens_used == 0

    def test_non_json_line_ignored(self) -> None:
        parser = StreamParser()
        parser.feed("not json at all")
        assert parser.final_text is None

    def test_result_event_extracts_text_and_tokens(self) -> None:
        parser = StreamParser()
        parser.feed(
            _make_event(type="result", result="hello world", usage={"output_tokens": 42})
        )
        assert parser.final_text == "hello world"
        assert parser.tokens_used == 42

    def test_result_event_without_usage(self) -> None:
        parser = StreamParser()
        parser.feed(_make_event(type="result", result="hi"))
        assert parser.tokens_used == 0

    def test_result_event_stores_session_id(self) -> None:
        parser = StreamParser()
        parser.feed(_make_event(type="result", result="x", session_id="sess-123"))
        assert parser.session_id == "sess-123"

    def test_system_init_stores_session_id(self) -> None:
        parser = StreamParser()
        parser.feed(_make_event(type="system", subtype="init", session_id="sess-abc"))
        assert parser.session_id == "sess-abc"


class TestStreamParserCallbacks:
    def test_stream_callback_receives_text(self) -> None:
        received: list[str] = []
        parser = StreamParser(stream_callback=received.append)

        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": " world"},
                ]
            },
        }
        parser.feed(json.dumps(event))
        assert received == ["Hello", " world"]

    def test_raw_stream_callback_receives_all_events(self) -> None:
        received: list[dict] = []
        parser = StreamParser(raw_stream_callback=received.append)

        parser.feed(_make_event(type="result", result="done"))
        parser.feed(_make_event(type="system", subtype="init"))
        assert len(received) == 2

    def test_stream_callback_not_called_for_result_event(self) -> None:
        received: list[str] = []
        parser = StreamParser(stream_callback=received.append)
        parser.feed(_make_event(type="result", result="final"))
        # result event 不应触发 stream_callback
        assert received == []

    def test_callback_exception_does_not_crash_parser(self) -> None:
        def bad_callback(text: str) -> None:
            raise RuntimeError("callback error")

        parser = StreamParser(stream_callback=bad_callback)
        event = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "hi"}]},
        }
        # 不应抛出异常
        parser.feed(json.dumps(event))

    def test_multiple_lines(self) -> None:
        received: list[str] = []
        parser = StreamParser(stream_callback=received.append)

        for text in ["A", "B", "C"]:
            event = {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": text}]},
            }
            parser.feed(json.dumps(event))

        assert received == ["A", "B", "C"]


class TestStreamEventFormat:
    """Claude CLI --verbose 模式下的 stream_event 格式。"""

    def test_content_block_delta_text(self) -> None:
        """content_block_delta.delta.text 触发 stream_callback。"""
        received: list[str] = []
        parser = StreamParser(stream_callback=received.append)

        parser.feed('{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text","text":"AI "}}}')
        parser.feed('{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text","text":"agent "}}}')

        assert received == ["AI ", "agent "]

    def test_multiple_indices(self) -> None:
        """多 index 的 delta 分别触发回调。"""
        received: list[str] = []
        parser = StreamParser(stream_callback=received.append)

        parser.feed('{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text","text":"第一句。"}}}')
        parser.feed('{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"text","text":"第二句。"}}}')

        assert received == ["第一句。", "第二句。"]

    def test_raw_callback_receives_stream_events(self) -> None:
        """raw_stream_callback 也能收到 stream_event。"""
        received: list[dict] = []
        parser = StreamParser(raw_stream_callback=received.append)

        parser.feed('{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text","text":"hi"}}}')

        assert len(received) == 1
        assert received[0]["type"] == "stream_event"

    def test_message_start_sets_session(self) -> None:
        """message_start event 含 session_id。"""
        parser = StreamParser()
        parser.feed('{"type":"stream_event","event":{"type":"message_start","message":{"id":"msg-1","session":"sess-verbose-1"}}}')

        assert parser.session_id == "sess-verbose-1"

    def test_callback_exception_does_not_crash(self) -> None:
        """stream_event 回调异常不中断解析。"""
        def bad(text: str) -> None:
            raise RuntimeError("boom")

        parser = StreamParser(stream_callback=bad)
        parser.feed('{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text","text":"x"}}}')
        # 不应抛出

    def test_non_text_delta_ignored(self) -> None:
        """非 text 类型的 delta 不触发 stream_callback。"""
        received: list[str] = []
        parser = StreamParser(stream_callback=received.append)

        parser.feed('{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{"}}}')

        assert received == []

    def test_mixed_formats(self) -> None:
        """stream_event 和 assistant 格式混合出现。"""
        received: list[str] = []
        parser = StreamParser(stream_callback=received.append)

        parser.feed('{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text","text":"A"}}}')
        parser.feed(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "B"}]}}))

        assert received == ["A", "B"]

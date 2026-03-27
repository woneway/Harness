"""tests/unit/test_state.py — State 类测试。"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from harness.state import State
from harness.tasks.result import Result


def _make_result(output: str = "ok", task_index: str = "0") -> Result:
    return Result(
        task_index=task_index,
        task_type="llm",
        output=output,
        raw_text=output,
        tokens_used=5,
        duration_seconds=1.0,
        success=True,
        error=None,
    )


class TestStateBasics:
    def test_default_state(self) -> None:
        s = State()
        assert s._results == []

    def test_append_result(self) -> None:
        s = State()
        r = _make_result("hello")
        s._append_result(r)
        assert len(s._results) == 1
        assert s._results[0].output == "hello"

    def test_set_output(self) -> None:
        s = State()
        s._set_output("analysis", "good code")
        assert s.analysis == "good code"  # type: ignore[attr-defined]

    def test_set_output_overwrite(self) -> None:
        s = State()
        s._set_output("key", "v1")
        s._set_output("key", "v2")
        assert s.key == "v2"  # type: ignore[attr-defined]

    def test_snapshot(self) -> None:
        s = State()
        s._append_result(_make_result("a"))
        s._set_output("x", 42)
        snap = s._snapshot()
        assert snap["_results_count"] == 1
        assert snap["x"] == 42

    def test_extra_fields_via_init(self) -> None:
        """State(extra='allow') 允许任意初始化字段。"""
        s = State(foo="bar")  # type: ignore[call-arg]
        assert s.foo == "bar"  # type: ignore[attr-defined]


class TestCustomState:
    def test_subclass(self) -> None:
        class MyState(State):
            score: float = 0.0
            analysis: str = ""

        s = MyState()
        assert s.score == 0.0
        s._set_output("score", 0.95)
        assert s.score == 0.95

    def test_subclass_with_results(self) -> None:
        class MyState(State):
            count: int = 0

        s = MyState(count=10)
        s._append_result(_make_result())
        assert s.count == 10
        assert len(s._results) == 1

    def test_snapshot_includes_custom_fields(self) -> None:
        class MyState(State):
            name: str = "test"

        s = MyState(name="hello")
        snap = s._snapshot()
        assert snap["name"] == "hello"


class TestSetOutputCoercion:
    """_set_output 自动类型转换测试（Issue #6）。"""

    def test_str_to_list(self) -> None:
        """str → list[dict] 自动 json.loads 转换。"""
        class MyState(State):
            competitors: list[dict] = []

        s = MyState()
        s._set_output("competitors", '[{"name": "a"}, {"name": "b"}]')
        assert isinstance(s.competitors, list)
        assert s.competitors[0]["name"] == "a"

    def test_str_to_dict(self) -> None:
        """str → dict 自动 json.loads 转换。"""
        class MyState(State):
            config: dict = {}

        s = MyState()
        s._set_output("config", '{"key": "value"}')
        assert isinstance(s.config, dict)
        assert s.config["key"] == "value"

    def test_non_json_str_stays_str(self, caplog: pytest.LogCaptureFixture) -> None:
        """非 JSON str 保持原值，输出 warning。"""
        import logging

        class MyState(State):
            items: list = []

        s = MyState()
        with caplog.at_level(logging.WARNING, logger="harness.state"):
            s._set_output("items", "not json")
        assert s.items == "not json"
        assert "json.loads 转换失败" in caplog.text

    def test_str_field_no_conversion(self) -> None:
        """目标字段是 str 时不尝试转换。"""
        class MyState(State):
            text: str = ""

        s = MyState()
        s._set_output("text", '{"key": "value"}')
        assert s.text == '{"key": "value"}'

    def test_extra_field_no_conversion(self) -> None:
        """extra 字段（未声明）不做转换。"""
        s = State()
        s._set_output("dynamic", '[1, 2, 3]')
        assert s.dynamic == '[1, 2, 3]'  # type: ignore[attr-defined]

    def test_non_str_value_no_conversion(self) -> None:
        """非 str 值直接写入，不做转换。"""
        class MyState(State):
            items: list = []

        s = MyState()
        s._set_output("items", [1, 2, 3])
        assert s.items == [1, 2, 3]

    def test_optional_list_conversion(self) -> None:
        """list | None 字段也能自动转换。"""
        class MyState(State):
            items: list[dict] | None = None

        s = MyState()
        s._set_output("items", '[{"name": "a"}]')
        assert isinstance(s.items, list)
        assert s.items[0]["name"] == "a"

    def test_optional_dict_conversion(self) -> None:
        """dict | None 字段也能自动转换。"""
        from typing import Optional

        class MyState(State):
            config: Optional[dict] = None

        s = MyState()
        s._set_output("config", '{"k": "v"}')
        assert isinstance(s.config, dict)
        assert s.config["k"] == "v"

    def test_json_with_markdown_fence(self) -> None:
        """带 markdown code fences 的 JSON 也能正确转换。"""
        class MyState(State):
            competitors: list[dict] = []

        s = MyState()
        raw = '```json\n[{"name": "a", "url": "u"}]\n```'
        s._set_output("competitors", raw)
        assert isinstance(s.competitors, list)
        assert s.competitors[0]["name"] == "a"

    def test_json_with_markdown_fence_no_lang(self) -> None:
        """无语言标记的 markdown fences 也能正确转换。"""
        class MyState(State):
            items: list[dict] = []

        s = MyState()
        raw = '```\n[{"k": "v"}]\n```'
        s._set_output("items", raw)
        assert isinstance(s.items, list)
        assert s.items[0]["k"] == "v"


class TestStateImport:
    def test_import_from_harness(self) -> None:
        from harness import State as S
        assert S is State

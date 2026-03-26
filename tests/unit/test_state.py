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


class TestStateImport:
    def test_import_from_harness(self) -> None:
        from harness import State as S
        assert S is State

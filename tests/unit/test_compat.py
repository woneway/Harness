"""tests/unit/test_compat.py — v1/v2 callable 双模式检测测试。"""

from __future__ import annotations

import pytest

from harness._internal.compat import call_with_compat, detect_callable_mode
from harness.state import State
from harness.tasks.result import Result


def _make_result(output: str = "ok") -> Result:
    return Result(
        task_index="0",
        task_type="llm",
        output=output,
        raw_text=output,
        tokens_used=5,
        duration_seconds=1.0,
        success=True,
        error=None,
    )


# ---------------------------------------------------------------------------
# detect_callable_mode
# ---------------------------------------------------------------------------


class TestDetectMode:
    def test_v1_lambda(self) -> None:
        assert detect_callable_mode(lambda results: "hi") == "results"

    def test_v1_named_results(self) -> None:
        def fn(results: list) -> str:
            return "v1"
        assert detect_callable_mode(fn) == "results"

    def test_v2_state_annotation(self) -> None:
        def fn(state: State) -> str:
            return "v2"
        assert detect_callable_mode(fn) == "state"

    def test_v2_state_subclass_annotation(self) -> None:
        class MyState(State):
            x: int = 0

        def fn(s: MyState) -> str:
            return "v2"
        assert detect_callable_mode(fn) == "state"

    def test_v2_state_name_no_annotation(self) -> None:
        def fn(state):
            return "v2"
        assert detect_callable_mode(fn) == "state"

    def test_v1_lambda_r(self) -> None:
        """lambda r: ... 默认 v1（不是 'state' 名）。"""
        assert detect_callable_mode(lambda r: r) == "results"

    def test_no_params(self) -> None:
        def fn():
            return "no params"
        assert detect_callable_mode(fn) == "results"


# ---------------------------------------------------------------------------
# call_with_compat
# ---------------------------------------------------------------------------


class TestCallWithCompat:
    def test_v1_receives_results(self) -> None:
        s = State()
        s._append_result(_make_result("hello"))

        def fn(results):
            return len(results)

        assert call_with_compat(fn, s) == 1

    def test_v2_receives_state(self) -> None:
        s = State()
        s._set_output("x", 42)

        def fn(state: State):
            return state.x  # type: ignore[attr-defined]

        assert call_with_compat(fn, s) == 42

    def test_v2_by_name(self) -> None:
        s = State()
        s._append_result(_make_result("hi"))

        def fn(state):
            return len(state._results)

        assert call_with_compat(fn, s) == 1

    def test_v1_lambda_with_state(self) -> None:
        """lambda results: ... 走 v1 路径，收到 list[Result]。"""
        s = State()
        s._append_result(_make_result())
        result = call_with_compat(lambda results: type(results).__name__, s)
        assert result == "list"

    def test_v1_warns_in_state_pipeline(self, caplog: pytest.LogCaptureFixture) -> None:
        """in_state_pipeline=True 且检测为 v1 模式时输出 warning。"""
        import logging

        s = State()
        s._append_result(_make_result())
        with caplog.at_level(logging.WARNING, logger="harness._internal.compat"):
            call_with_compat(lambda s: len(s), s, in_state_pipeline=True)
        assert "v1 模式" in caplog.text

    def test_v2_no_warn_in_state_pipeline(self, caplog: pytest.LogCaptureFixture) -> None:
        """v2 模式不应输出 warning。"""
        import logging

        s = State()
        s._set_output("x", 1)

        def fn(state: State):
            return state.x  # type: ignore[attr-defined]

        with caplog.at_level(logging.WARNING, logger="harness._internal.compat"):
            call_with_compat(fn, s, in_state_pipeline=True)
        assert "v1 模式" not in caplog.text

    def test_v1_no_warn_without_flag(self, caplog: pytest.LogCaptureFixture) -> None:
        """in_state_pipeline=False（默认）不输出 warning。"""
        import logging

        s = State()
        with caplog.at_level(logging.WARNING, logger="harness._internal.compat"):
            call_with_compat(lambda r: r, s)
        assert "v1 模式" not in caplog.text

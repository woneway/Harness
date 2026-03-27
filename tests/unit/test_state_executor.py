"""tests/unit/test_state_executor.py — State 模式 executor 和 output_key 测试。"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from harness._internal.executor import execute_function_task, execute_shell_task
from harness.state import State
from harness.tasks import FunctionTask, ShellTask, Result, TaskConfig


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


class TestFunctionTaskWithState:
    @pytest.mark.asyncio
    async def test_v1_mode_with_state(self) -> None:
        """v1 callable (lambda results: ...) 在有 state 时仍收到 list[Result]。"""
        s = State()
        s._append_result(_make_result("prev"))

        task = FunctionTask(fn=lambda results: len(results))
        r = await execute_function_task(
            task, "1", s._results, "run1",
            harness_config=TaskConfig(max_retries=0),
            state=s,
        )
        assert r.success
        assert r.output == 1

    @pytest.mark.asyncio
    async def test_v2_mode_with_state(self) -> None:
        """v2 callable (def fn(state: State)) 收到 State 对象。"""
        s = State()
        s._set_output("x", 42)

        def fn(state: State):
            return state.x  # type: ignore[attr-defined]

        task = FunctionTask(fn=fn)
        r = await execute_function_task(
            task, "1", s._results, "run1",
            harness_config=TaskConfig(max_retries=0),
            state=s,
        )
        assert r.success
        assert r.output == 42

    @pytest.mark.asyncio
    async def test_v2_by_name(self) -> None:
        """参数名 'state' 也触发 v2 模式。"""
        s = State()
        s._set_output("val", "hello")

        def fn(state):
            return state.val  # type: ignore[attr-defined]

        task = FunctionTask(fn=fn)
        r = await execute_function_task(
            task, "1", s._results, "run1",
            harness_config=TaskConfig(max_retries=0),
            state=s,
        )
        assert r.success
        assert r.output == "hello"

    @pytest.mark.asyncio
    async def test_no_state_fallback(self) -> None:
        """state=None 时正常走 v1 路径。"""
        task = FunctionTask(fn=lambda results: "no state")
        r = await execute_function_task(
            task, "0", [], "run1",
            harness_config=TaskConfig(max_retries=0),
        )
        assert r.success
        assert r.output == "no state"


class TestShellTaskWithState:
    @pytest.mark.asyncio
    async def test_v1_cmd_callable_with_state(self) -> None:
        """v1 cmd callable 在有 state 时仍收到 list[Result]。"""
        s = State()
        s._append_result(_make_result("prev"))

        task = ShellTask(cmd=lambda results: "echo ok")
        r = await execute_shell_task(
            task, "0", s._results, "run1",
            harness_config=TaskConfig(max_retries=0),
            state=s,
        )
        assert r.success
        assert "ok" in r.output

    @pytest.mark.asyncio
    async def test_v2_cmd_callable_with_state(self) -> None:
        """v2 cmd callable 收到 State。"""
        s = State()
        s._set_output("msg", "hello")

        def cmd(state: State) -> str:
            return f"echo {state.msg}"  # type: ignore[attr-defined]

        task = ShellTask(cmd=cmd)
        r = await execute_shell_task(
            task, "0", s._results, "run1",
            harness_config=TaskConfig(max_retries=0),
            state=s,
        )
        assert r.success
        assert "hello" in r.output


class TestAsyncFunctionTask:
    """async FunctionTask 支持测试（Issue #7）。"""

    @pytest.mark.asyncio
    async def test_async_fn_v2(self) -> None:
        """async def fn(state) 返回值被正确 await。"""
        s = State()
        s._set_output("val", 10)

        async def fn(state: State):
            return state.val * 2  # type: ignore[attr-defined]

        task = FunctionTask(fn=fn)
        r = await execute_function_task(
            task, "0", s._results, "run1",
            harness_config=TaskConfig(max_retries=0),
            state=s,
        )
        assert r.success
        assert r.output == 20

    @pytest.mark.asyncio
    async def test_async_fn_v1(self) -> None:
        """async def fn(results) 在无 state 时也能正确 await。"""
        async def fn(results):
            return f"count={len(results)}"

        task = FunctionTask(fn=fn)
        r = await execute_function_task(
            task, "0", [], "run1",
            harness_config=TaskConfig(max_retries=0),
        )
        assert r.success
        assert r.output == "count=0"

    @pytest.mark.asyncio
    async def test_sync_fn_still_works(self) -> None:
        """同步函数不受影响。"""
        task = FunctionTask(fn=lambda results: 42)
        r = await execute_function_task(
            task, "0", [], "run1",
            harness_config=TaskConfig(max_retries=0),
        )
        assert r.success
        assert r.output == 42


class TestOutputKey:
    def test_output_key_on_llm_task(self) -> None:
        from harness.tasks import LLMTask
        task = LLMTask(prompt="test", output_key="analysis")
        assert task.output_key == "analysis"

    def test_output_key_on_function_task(self) -> None:
        task = FunctionTask(fn=lambda r: 42, output_key="count")
        assert task.output_key == "count"

    def test_output_key_on_shell_task(self) -> None:
        task = ShellTask(cmd="echo hi", output_key="output")
        assert task.output_key == "output"

    def test_output_key_default_none(self) -> None:
        from harness.tasks import LLMTask
        task = LLMTask(prompt="test")
        assert task.output_key is None

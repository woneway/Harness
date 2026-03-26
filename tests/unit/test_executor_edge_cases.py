"""tests/unit/test_executor_edge_cases.py — Executor 缺失路径覆盖。"""

from __future__ import annotations

import asyncio
import os

import pytest

from harness._internal.exceptions import TaskFailedError
from harness._internal.executor import (
    _emit_separator,
    execute_function_task,
    execute_llm_task,
    execute_shell_task,
)
from harness._internal.session import SessionManager
from harness.runners.base import AbstractRunner, RunnerResult
from harness.task import FunctionTask, LLMTask, Result, ShellTask, TaskConfig


# ---------------------------------------------------------------------------
# LLMTask 超时
# ---------------------------------------------------------------------------


class SlowRunner(AbstractRunner):
    """模拟超时的 runner。"""

    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        await asyncio.sleep(100)
        return RunnerResult(text="ok", tokens_used=0, session_id=None)


class TestLLMTaskTimeout:
    @pytest.mark.asyncio
    async def test_timeout_marks_session_broken(self) -> None:
        """LLMTask 超时后 session 应被标记为 broken。"""
        sm = SessionManager()
        task = LLMTask(
            prompt="slow",
            config=TaskConfig(max_retries=0, timeout=0.01),
        )
        with pytest.raises(TaskFailedError) as exc_info:
            await execute_llm_task(
                task, "0", [], "run-1",
                harness_system_prompt="",
                harness_runner=SlowRunner(),
                harness_config=None,
                session_manager=sm,
                memory_injection="",
            )
        assert "timed out" in exc_info.value.error


# ---------------------------------------------------------------------------
# FunctionTask env_overrides
# ---------------------------------------------------------------------------


class TestFunctionTaskEnvOverrides:
    @pytest.mark.asyncio
    async def test_env_overrides_applied_and_restored(self) -> None:
        """env_overrides 在执行期间生效，执行后恢复。"""
        original = os.environ.get("HARNESS_TEST_VAR")

        captured = {}

        def fn(results):
            captured["var"] = os.environ.get("HARNESS_TEST_VAR")
            return "ok"

        task = FunctionTask(fn=fn)
        await execute_function_task(
            task, "0", [], "run-1",
            harness_config=None,
            env_overrides={"HARNESS_TEST_VAR": "test_value"},
        )

        assert captured["var"] == "test_value"
        assert os.environ.get("HARNESS_TEST_VAR") == original

    @pytest.mark.asyncio
    async def test_env_overrides_remove_var(self) -> None:
        """env_overrides 值为空字符串时删除环境变量。"""
        os.environ["HARNESS_TEST_REMOVE"] = "exists"
        captured = {}

        def fn(results):
            captured["var"] = os.environ.get("HARNESS_TEST_REMOVE")
            return "ok"

        task = FunctionTask(fn=fn)
        try:
            await execute_function_task(
                task, "0", [], "run-1",
                harness_config=None,
                env_overrides={"HARNESS_TEST_REMOVE": ""},
            )
            assert captured["var"] is None
        finally:
            # 确保恢复
            assert os.environ.get("HARNESS_TEST_REMOVE") == "exists"
            os.environ.pop("HARNESS_TEST_REMOVE", None)

    @pytest.mark.asyncio
    async def test_env_overrides_restored_on_exception(self) -> None:
        """fn 抛异常时 env_overrides 仍然被恢复。"""
        original = os.environ.get("HARNESS_TEST_ERR")

        def fn(results):
            raise RuntimeError("boom")

        task = FunctionTask(
            fn=fn,
            config=TaskConfig(max_retries=0),
        )
        with pytest.raises(TaskFailedError):
            await execute_function_task(
                task, "0", [], "run-1",
                harness_config=None,
                env_overrides={"HARNESS_TEST_ERR": "temp"},
            )

        assert os.environ.get("HARNESS_TEST_ERR") == original


# ---------------------------------------------------------------------------
# ShellTask 超时
# ---------------------------------------------------------------------------


class TestShellTaskTimeout:
    @pytest.mark.asyncio
    async def test_shell_timeout_kills_process(self) -> None:
        """ShellTask 超时后进程被 kill。"""
        task = ShellTask(
            cmd="sleep 100",
            config=TaskConfig(max_retries=0, timeout=0.05),
        )
        with pytest.raises(TaskFailedError) as exc_info:
            await execute_shell_task(
                task, "0", [], "run-1",
                harness_config=None,
            )
        assert "timed out" in exc_info.value.error


class TestShellTaskEnvOverrides:
    @pytest.mark.asyncio
    async def test_env_overrides_passed_to_subprocess(self) -> None:
        """env_overrides 传递到 shell 子进程。"""
        task = ShellTask(cmd="echo $HARNESS_SHELL_TEST")
        result = await execute_shell_task(
            task, "0", [], "run-1",
            harness_config=None,
            env_overrides={"HARNESS_SHELL_TEST": "hello_env"},
        )
        assert "hello_env" in result.output

    @pytest.mark.asyncio
    async def test_task_env_passed_to_subprocess(self) -> None:
        """task.env 传递到 shell 子进程。"""
        task = ShellTask(
            cmd="echo $HARNESS_TASK_ENV",
            env={"HARNESS_TASK_ENV": "from_task"},
        )
        result = await execute_shell_task(
            task, "0", [], "run-1",
            harness_config=None,
        )
        assert "from_task" in result.output

    @pytest.mark.asyncio
    async def test_env_overrides_override_task_env(self) -> None:
        """env_overrides 优先级高于 task.env。"""
        task = ShellTask(
            cmd="echo $HARNESS_PRIORITY_TEST",
            env={"HARNESS_PRIORITY_TEST": "from_task"},
        )
        result = await execute_shell_task(
            task, "0", [], "run-1",
            harness_config=None,
            env_overrides={"HARNESS_PRIORITY_TEST": "from_harness"},
        )
        assert "from_harness" in result.output


# ---------------------------------------------------------------------------
# _emit_separator 回调异常
# ---------------------------------------------------------------------------


class TestEmitSeparator:
    def test_callback_exception_suppressed(self) -> None:
        """stream_callback 抛异常不影响执行。"""

        def bad_cb(text: str) -> None:
            raise RuntimeError("callback error")

        # 不应抛异常
        _emit_separator("0", "llm", bad_cb, None)

    def test_callback_receives_separator(self) -> None:
        captured = []
        _emit_separator("1", "function", captured.append, None)
        assert len(captured) == 1
        assert "Task 1" in captured[0]
        assert "function" in captured[0]

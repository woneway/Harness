"""tests/unit/test_parallel_edge_cases.py — Parallel 缺失路径覆盖。"""

from __future__ import annotations

import asyncio

import pytest

from harness._internal.exceptions import TaskFailedError
from harness._internal.parallel import (
    _run_all_or_nothing,
    _run_best_effort,
    _task_type_str,
    execute_parallel,
)
from harness._internal.session import SessionManager
from harness.runners.base import AbstractRunner, RunnerResult
from harness.task import FunctionTask, LLMTask, Parallel, PollingTask, Result, ShellTask, TaskConfig


class MockRunner(AbstractRunner):
    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        return RunnerResult(text="ok", tokens_used=0, session_id=None)


def make_result(index: str = "0") -> Result:
    return Result(
        task_index=index, task_type="llm", output="x",
        raw_text="x", tokens_used=0, duration_seconds=0.0,
        success=True, error=None,
    )


def make_parallel_kwargs(**overrides):
    defaults = dict(
        run_id="run-1",
        harness_system_prompt="",
        harness_runner=MockRunner(),
        harness_config=None,
        session_manager=SessionManager(),
        memory_injection="",
        storage=None,
        is_new_session=False,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# _run_all_or_nothing: task cancellation & non-TaskFailedError wrapping
# ---------------------------------------------------------------------------


class TestAllOrNothingCancellation:
    @pytest.mark.asyncio
    async def test_failure_cancels_remaining_tasks(self) -> None:
        """一个任务失败时，其余任务应被取消。"""
        slow_cancelled = False

        async def slow_coro():
            nonlocal slow_cancelled
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                slow_cancelled = True
                raise
            return make_result("0.0")

        async def fail_coro():
            raise TaskFailedError("run-1", "0.1", "function", "boom")

        with pytest.raises(TaskFailedError):
            await _run_all_or_nothing(
                [slow_coro(), fail_coro()],
                "run-1",
                0,
            )
        # 给取消一点时间完成
        await asyncio.sleep(0.05)
        assert slow_cancelled

    @pytest.mark.asyncio
    async def test_non_task_failed_error_wrapped(self) -> None:
        """非 TaskFailedError 应被包装为 TaskFailedError。"""

        async def fail_coro():
            raise ValueError("unexpected")

        async def ok_coro():
            return make_result("0.0")

        with pytest.raises(TaskFailedError) as exc_info:
            await _run_all_or_nothing(
                [ok_coro(), fail_coro()],
                "run-1",
                0,
            )
        assert "unexpected" in exc_info.value.error


# ---------------------------------------------------------------------------
# _run_best_effort: exception wrapping
# ---------------------------------------------------------------------------


class TestBestEffortErrorWrapping:
    @pytest.mark.asyncio
    async def test_failed_tasks_marked_as_failed(self) -> None:
        """失败的 task 在 best_effort 模式下标记为 success=False。"""

        async def ok_coro():
            return make_result("0.0")

        async def fail_coro():
            raise RuntimeError("boom")

        results = await _run_best_effort(
            [ok_coro(), fail_coro()],
            ["0.0", "0.1"],
            ["function", "llm"],
        )

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error == "boom"
        assert results[1].task_type == "llm"
        assert results[1].task_index == "0.1"


# ---------------------------------------------------------------------------
# Stream callback merging in _execute_one
# ---------------------------------------------------------------------------


class TestStreamCallbackMerging:
    @pytest.mark.asyncio
    async def test_harness_callback_used_when_task_has_none(self) -> None:
        """Task 无 stream_callback 时使用 Harness 级回调。"""
        harness_cb_called = []

        def harness_cb(text: str) -> None:
            harness_cb_called.append(text)

        task = LLMTask(prompt="hello")
        p = Parallel(tasks=[task])
        kwargs = make_parallel_kwargs(
            harness_stream_callback=harness_cb,
        )

        results = await execute_parallel(p, 0, [], **kwargs)

        assert len(results) == 1
        assert results[0].success is True
        # harness_cb should have been called at least for separator
        assert len(harness_cb_called) > 0

    @pytest.mark.asyncio
    async def test_task_callback_takes_priority(self) -> None:
        """Task 有 stream_callback 时覆盖 Harness 级。"""
        task_cb_called = []
        harness_cb_called = []

        def task_cb(text: str) -> None:
            task_cb_called.append(text)

        def harness_cb(text: str) -> None:
            harness_cb_called.append(text)

        task = LLMTask(prompt="hello", stream_callback=task_cb)
        p = Parallel(tasks=[task])
        kwargs = make_parallel_kwargs(
            harness_stream_callback=harness_cb,
        )

        results = await execute_parallel(p, 0, [], **kwargs)
        assert len(results) == 1
        # Task-level callback used
        assert len(task_cb_called) > 0
        # Harness-level not used
        assert len(harness_cb_called) == 0


# ---------------------------------------------------------------------------
# ShellTask dispatch in parallel
# ---------------------------------------------------------------------------


class TestParallelShellTask:
    @pytest.mark.asyncio
    async def test_shell_task_in_parallel(self) -> None:
        """ShellTask 在 Parallel 中正常执行。"""
        p = Parallel(tasks=[ShellTask(cmd="echo parallel_shell")])
        kwargs = make_parallel_kwargs()

        results = await execute_parallel(p, 0, [], **kwargs)
        assert len(results) == 1
        assert results[0].success is True
        assert "parallel_shell" in results[0].output


# ---------------------------------------------------------------------------
# _task_type_str coverage
# ---------------------------------------------------------------------------


class TestTaskTypeStr:
    def test_llm(self) -> None:
        assert _task_type_str(LLMTask(prompt="x")) == "llm"

    def test_function(self) -> None:
        assert _task_type_str(FunctionTask(fn=lambda r: None)) == "function"

    def test_shell(self) -> None:
        assert _task_type_str(ShellTask(cmd="echo x")) == "shell"

    def test_polling(self) -> None:
        t = PollingTask(
            submit_fn=lambda r: None,
            poll_fn=lambda h: None,
            success_condition=lambda r: True,
        )
        assert _task_type_str(t) == "polling"


# ---------------------------------------------------------------------------
# Best effort with Parallel block
# ---------------------------------------------------------------------------


class TestParallelBestEffort:
    @pytest.mark.asyncio
    async def test_best_effort_partial_failure(self) -> None:
        """best_effort 模式下部分失败不抛异常。"""

        def fail_fn(results):
            raise RuntimeError("fail!")

        p = Parallel(
            tasks=[
                FunctionTask(fn=lambda r: "ok"),
                FunctionTask(fn=fail_fn, config=TaskConfig(max_retries=0)),
            ],
            error_policy="best_effort",
        )
        kwargs = make_parallel_kwargs()

        results = await execute_parallel(p, 0, [], **kwargs)
        assert len(results) == 2
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        assert len(successes) == 1
        assert len(failures) == 1

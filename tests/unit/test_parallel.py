"""tests/unit/test_parallel.py — Parallel 执行逻辑验证。"""

from __future__ import annotations

import asyncio

import pytest

from harness._internal.exceptions import InvalidPipelineError, TaskFailedError
from harness._internal.parallel import execute_parallel
from harness._internal.session import SessionManager
from harness.runners.base import AbstractRunner, RunnerResult
from harness.task import FunctionTask, LLMTask, Parallel, Result, ShellTask, TaskConfig


class MockRunner(AbstractRunner):
    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        return RunnerResult(text="ok", tokens_used=0, session_id=None)


def make_session() -> SessionManager:
    return SessionManager()


def make_parallel_kwargs():
    return dict(
        run_id="run-1",
        harness_system_prompt="",
        harness_runner=MockRunner(),
        harness_config=None,
        session_manager=make_session(),
        memory_injection="",
        storage=None,
        is_new_session=False,
    )


class TestParallelTaskIndex:
    @pytest.mark.asyncio
    async def test_task_index_string_format(self) -> None:
        """task_index 应为字符串 '2.0', '2.1'。"""
        task_indices: list[str] = []

        def record_fn(results):
            return "ok"

        tasks = [
            FunctionTask(fn=lambda r: "a"),
            FunctionTask(fn=lambda r: "b"),
        ]
        parallel = Parallel(tasks=tasks)
        results = await execute_parallel(parallel, 2, [], **make_parallel_kwargs())

        assert len(results) == 2
        assert results[0].task_index == "2.0"
        assert results[1].task_index == "2.1"
        assert all(isinstance(r.task_index, str) for r in results)

    @pytest.mark.asyncio
    async def test_task_index_from_outer_index_0(self) -> None:
        tasks = [FunctionTask(fn=lambda r: "x")]
        parallel = Parallel(tasks=tasks)
        results = await execute_parallel(parallel, 0, [], **make_parallel_kwargs())
        assert results[0].task_index == "0.0"


class TestAllOrNothing:
    @pytest.mark.asyncio
    async def test_all_succeed(self) -> None:
        tasks = [
            FunctionTask(fn=lambda r: "a"),
            FunctionTask(fn=lambda r: "b"),
            FunctionTask(fn=lambda r: "c"),
        ]
        parallel = Parallel(tasks=tasks, error_policy="all_or_nothing")
        results = await execute_parallel(parallel, 1, [], **make_parallel_kwargs())
        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_one_fails_raises(self) -> None:
        """all_or_nothing：一个失败，整体抛 TaskFailedError。"""
        started: list[str] = []
        cancelled: list[str] = []

        async def slow_ok(delay: float, idx: str):
            try:
                started.append(idx)
                await asyncio.sleep(delay)
                return idx
            except asyncio.CancelledError:
                cancelled.append(idx)
                raise

        def make_fn(should_fail: bool):
            def fn(results):
                if should_fail:
                    raise RuntimeError("task failed")
                return "ok"
            return fn

        tasks = [
            FunctionTask(fn=make_fn(False)),
            FunctionTask(fn=make_fn(True)),   # 这个失败
            FunctionTask(fn=make_fn(False)),
        ]
        parallel = Parallel(tasks=tasks, error_policy="all_or_nothing")
        with pytest.raises((TaskFailedError, Exception)):
            await execute_parallel(parallel, 0, [], **make_parallel_kwargs())


class TestBestEffort:
    @pytest.mark.asyncio
    async def test_partial_failure_returns_all_results(self) -> None:
        """best_effort：部分失败时返回所有结果，失败的 success=False。"""
        tasks = [
            FunctionTask(fn=lambda r: "ok"),
            FunctionTask(fn=lambda r: (_ for _ in ()).throw(RuntimeError("fail"))),
            FunctionTask(fn=lambda r: "ok2"),
        ]
        parallel = Parallel(tasks=tasks, error_policy="best_effort")
        results = await execute_parallel(parallel, 3, [], **make_parallel_kwargs())
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

    @pytest.mark.asyncio
    async def test_all_fail_no_exception(self) -> None:
        """best_effort：全部失败时也不抛异常。"""
        tasks = [
            FunctionTask(fn=lambda r: (_ for _ in ()).throw(RuntimeError("f1"))),
            FunctionTask(fn=lambda r: (_ for _ in ()).throw(RuntimeError("f2"))),
        ]
        parallel = Parallel(tasks=tasks, error_policy="best_effort")
        results = await execute_parallel(parallel, 0, [], **make_parallel_kwargs())
        assert len(results) == 2
        assert all(not r.success for r in results)


class TestNestedParallelValidation:
    @pytest.mark.asyncio
    async def test_nested_parallel_raises_invalid_pipeline_error(self) -> None:
        """嵌套 Parallel 应抛 InvalidPipelineError。"""
        inner = Parallel(tasks=[FunctionTask(fn=lambda r: "x")])
        outer = Parallel(tasks=[inner])  # type: ignore[arg-type]
        with pytest.raises(InvalidPipelineError):
            await execute_parallel(outer, 0, [], **make_parallel_kwargs())


class TestParallelMaxRetries:
    """测试 Parallel 块级最大重试限制（all_or_nothing 策略）。

    注意：子 Task 使用 config=TaskConfig(max_retries=0) 以排除子任务内部重试的干扰，
    确保 call_count 计数仅反映"整块重试次数 × 子任务数"。
    """

    @pytest.mark.asyncio
    async def test_parallel_max_retries_exceeded(self) -> None:
        """all_or_nothing 策略下，子 Task 持续失败，达到 max_retries 后抛 TaskFailedError。"""
        call_count = 0
        no_retry_config = TaskConfig(max_retries=0)

        def always_fail(results):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("persistent failure")

        parallel = Parallel(
            tasks=[FunctionTask(fn=always_fail, config=no_retry_config)],
            error_policy="all_or_nothing",
            max_retries=2,
        )
        with pytest.raises(TaskFailedError):
            await execute_parallel(parallel, 0, [], **make_parallel_kwargs())

        # max_retries=2 意味着最多执行 3 次（初始 1 次 + 2 次重试）
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_parallel_max_retries_default(self) -> None:
        """验证 Parallel.max_retries 默认值为 2。"""
        parallel = Parallel(tasks=[])
        assert parallel.max_retries == 2

    @pytest.mark.asyncio
    async def test_parallel_succeeds_within_retries(self) -> None:
        """第 2 次重试时子 Task 成功，整块应成功返回结果。"""
        call_count = 0
        no_retry_config = TaskConfig(max_retries=0)

        def succeed_on_third(results):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError(f"attempt {call_count} failed")
            return "success"

        parallel = Parallel(
            tasks=[FunctionTask(fn=succeed_on_third, config=no_retry_config)],
            error_policy="all_or_nothing",
            max_retries=2,
        )
        results = await execute_parallel(parallel, 0, [], **make_parallel_kwargs())

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].output == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_parallel_max_retries_zero_no_retry(self) -> None:
        """max_retries=0 时，失败不重试，直接抛 TaskFailedError。"""
        call_count = 0
        no_retry_config = TaskConfig(max_retries=0)

        def always_fail(results):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        parallel = Parallel(
            tasks=[FunctionTask(fn=always_fail, config=no_retry_config)],
            error_policy="all_or_nothing",
            max_retries=0,
        )
        with pytest.raises(TaskFailedError):
            await execute_parallel(parallel, 0, [], **make_parallel_kwargs())

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_parallel_max_retries_best_effort_not_affected(self) -> None:
        """best_effort 策略下，max_retries 字段存在但整块不重试，失败的子 Task 标记 success=False。"""
        call_count = 0
        no_retry_config = TaskConfig(max_retries=0)

        def always_fail(results):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        parallel = Parallel(
            tasks=[FunctionTask(fn=always_fail, config=no_retry_config)],
            error_policy="best_effort",
            max_retries=2,
        )
        results = await execute_parallel(parallel, 0, [], **make_parallel_kwargs())

        # best_effort 不受整块 max_retries 影响，返回失败结果而非重试
        assert len(results) == 1
        assert results[0].success is False
        assert call_count == 1

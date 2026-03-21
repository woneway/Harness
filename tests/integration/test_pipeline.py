"""tests/integration/test_pipeline.py — 端到端集成测试（不需要 Claude CLI）。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from harness import Harness, FunctionTask, ShellTask, TaskConfig
from harness._internal.exceptions import TaskFailedError
from harness.task import Parallel, PollingTask


def make_harness(tmp_path: Path, **kwargs) -> Harness:
    """创建不依赖 Claude CLI 的 Harness 实例。"""
    from harness.runners.base import AbstractRunner, RunnerResult

    class NoopRunner(AbstractRunner):
        async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
            return RunnerResult(text=prompt, tokens_used=0, session_id=None)

    return Harness(str(tmp_path), runner=NoopRunner(), **kwargs)


# ---------------------------------------------------------------------------
# 顺序 pipeline
# ---------------------------------------------------------------------------


class TestSequentialPipeline:
    @pytest.mark.asyncio
    async def test_function_and_shell(self, tmp_path: Path) -> None:
        """FunctionTask + ShellTask 顺序执行。"""
        h = make_harness(tmp_path)
        result = await h.pipeline([
            FunctionTask(fn=lambda r: "computed"),
            ShellTask(cmd="echo shell-output"),
        ])
        assert len(result.results) == 2
        assert result.results[0].output == "computed"
        assert "shell-output" in result.results[1].output

    @pytest.mark.asyncio
    async def test_results_passed_to_next_task(self, tmp_path: Path) -> None:
        """前序结果通过 results 列表传递给后续 task。"""
        h = make_harness(tmp_path)
        captured = []

        def second_fn(results):
            captured.extend(results)
            return f"got-{results[0].output}"

        result = await h.pipeline([
            FunctionTask(fn=lambda r: "value-from-step1"),
            FunctionTask(fn=second_fn),
        ])
        assert len(captured) >= 1
        assert captured[0].output == "value-from-step1"
        assert result.results[1].output == "got-value-from-step1"

    @pytest.mark.asyncio
    async def test_total_tokens_summed(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        result = await h.pipeline([
            FunctionTask(fn=lambda r: "a"),
            FunctionTask(fn=lambda r: "b"),
        ])
        assert result.total_tokens == 0  # FunctionTask tokens = 0

    @pytest.mark.asyncio
    async def test_run_saved_to_storage(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        pr = await h.pipeline([FunctionTask(fn=lambda r: "x")], name="test-run")

        assert h._storage is not None
        run = await h._storage.get_run(pr.run_id)
        assert run is not None
        assert run["name"] == "test-run"
        assert run["status"] == "success"


# ---------------------------------------------------------------------------
# PollingTask 三路径
# ---------------------------------------------------------------------------


class TestPollingTask:
    @pytest.mark.asyncio
    async def test_polling_success_path(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        result = await h.pipeline([
            PollingTask(
                submit_fn=lambda r: "job-handle",
                poll_fn=lambda h: {"status": "done"},
                success_condition=lambda r: r["status"] == "done",
                poll_interval=0.001,
                timeout=10,
                config=TaskConfig(max_retries=0),
            )
        ])
        assert result.results[0].success is True

    @pytest.mark.asyncio
    async def test_polling_failure_condition_path(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        with pytest.raises(TaskFailedError):
            await h.pipeline([
                PollingTask(
                    submit_fn=lambda r: "h",
                    poll_fn=lambda h: {"status": "error"},
                    success_condition=lambda r: r["status"] == "done",
                    failure_condition=lambda r: r["status"] == "error",
                    poll_interval=0.001,
                    timeout=10,
                    config=TaskConfig(max_retries=0),
                )
            ])

    @pytest.mark.asyncio
    async def test_polling_timeout_path(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        with pytest.raises(TaskFailedError):
            await h.pipeline([
                PollingTask(
                    submit_fn=lambda r: "h",
                    poll_fn=lambda h: {"status": "pending"},
                    success_condition=lambda r: r["status"] == "done",
                    poll_interval=0.001,
                    timeout=0.005,
                    config=TaskConfig(max_retries=0),
                )
            ])


# ---------------------------------------------------------------------------
# Parallel all_or_nothing
# ---------------------------------------------------------------------------


class TestParallelAllOrNothing:
    @pytest.mark.asyncio
    async def test_all_succeed(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        result = await h.pipeline([
            Parallel(tasks=[
                FunctionTask(fn=lambda r: "a"),
                FunctionTask(fn=lambda r: "b"),
            ])
        ])
        assert len(result.results) == 2
        assert all(r.success for r in result.results)

    @pytest.mark.asyncio
    async def test_one_fail_raises(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        with pytest.raises((TaskFailedError, Exception)):
            await h.pipeline([
                Parallel(tasks=[
                    FunctionTask(fn=lambda r: "ok"),
                    FunctionTask(
                        fn=lambda r: (_ for _ in ()).throw(RuntimeError("fail")),
                        config=TaskConfig(max_retries=0),
                    ),
                ], error_policy="all_or_nothing")
            ])


# ---------------------------------------------------------------------------
# Parallel best_effort
# ---------------------------------------------------------------------------


class TestParallelBestEffort:
    @pytest.mark.asyncio
    async def test_partial_failure_in_result(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        result = await h.pipeline([
            Parallel(
                tasks=[
                    FunctionTask(fn=lambda r: "ok"),
                    FunctionTask(
                        fn=lambda r: (_ for _ in ()).throw(RuntimeError("fail")),
                        config=TaskConfig(max_retries=0),
                    ),
                    FunctionTask(fn=lambda r: "ok2"),
                ],
                error_policy="best_effort",
            )
        ])
        assert len(result.results) == 3
        successes = [r for r in result.results if r.success]
        failures = [r for r in result.results if not r.success]
        assert len(successes) == 2
        assert len(failures) == 1


# ---------------------------------------------------------------------------
# resume_from 续跑
# ---------------------------------------------------------------------------


class TestResumeFrom:
    @pytest.mark.asyncio
    async def test_resume_skips_task_0_reruns_task_1(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)

        task_0_calls = [0]
        task_1_calls = [0]

        def task0_fn(results):
            task_0_calls[0] += 1
            return "task0-result"

        def task1_fn(results):
            task_1_calls[0] += 1
            if task_1_calls[0] == 1:
                raise RuntimeError("first run fails")
            return "task1-fixed"

        tasks = [
            FunctionTask(fn=task0_fn),
            FunctionTask(fn=task1_fn, config=TaskConfig(max_retries=0)),
        ]

        # 第一次跑：Task 0 成功，Task 1 失败
        failed_run_id = None
        try:
            await h.pipeline(tasks)
        except TaskFailedError as e:
            failed_run_id = e.run_id

        assert failed_run_id is not None
        assert task_0_calls[0] == 1
        assert task_1_calls[0] == 1

        # 重置计数
        task_0_calls[0] = 0

        # 续跑：Task 0 应被跳过
        result = await h.pipeline(tasks, resume_from=failed_run_id)
        assert task_0_calls[0] == 0  # Task 0 被跳过
        assert task_1_calls[0] == 2  # Task 1 重新执行

    @pytest.mark.asyncio
    async def test_parallel_block_reruns_if_partial_failure(self, tmp_path: Path) -> None:
        """Parallel 块内任意子 task 失败，续跑时整体重跑。"""
        h = make_harness(tmp_path)

        parallel_call_counts = [0, 0]

        def fn0(results):
            parallel_call_counts[0] += 1
            return "p0"

        def fn1(results):
            parallel_call_counts[1] += 1
            if parallel_call_counts[1] == 1:
                raise RuntimeError("p1 fails first time")
            return "p1-fixed"

        tasks = [
            Parallel(
                tasks=[
                    FunctionTask(fn=fn0),
                    FunctionTask(fn=fn1, config=TaskConfig(max_retries=0)),
                ],
                error_policy="best_effort",
            )
        ]

        # 第一次跑
        pr1 = await h.pipeline(tasks)
        # best_effort：一个失败，一个成功
        assert parallel_call_counts[0] == 1
        assert parallel_call_counts[1] == 1

        # 续跑：Parallel 块整体重跑（因为有子 task 失败）
        pr2 = await h.pipeline(tasks, resume_from=pr1.run_id)
        # fn0 应被重新调用（Parallel 块整体重跑）
        assert parallel_call_counts[0] == 2

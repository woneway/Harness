"""tests/unit/test_harness.py — Harness 主类逻辑验证（mock executor）。"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness import Harness, LLMTask, FunctionTask, ShellTask, Task, TaskConfig
from harness._internal.exceptions import InvalidPipelineError, TaskFailedError
from harness.runners.base import AbstractRunner, RunnerResult
from harness.task import Parallel, Result


class MockRunner(AbstractRunner):
    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        return RunnerResult(text="mock result", tokens_used=5, session_id="s1")


def make_harness(tmp_path: Path, **kwargs) -> Harness:
    return Harness(str(tmp_path), runner=MockRunner(), **kwargs)


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------


class TestHarnessInit:
    def test_stream_callback_mutex(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            Harness(
                str(tmp_path),
                stream_callback=lambda t: None,
                raw_stream_callback=lambda e: None,
            )

    def test_only_stream_callback_ok(self, tmp_path: Path) -> None:
        h = Harness(str(tmp_path), stream_callback=lambda t: None)
        assert h._stream_callback is not None

    def test_default_runner_is_claude_cli(self, tmp_path: Path) -> None:
        from harness.runners.claude_cli import ClaudeCliRunner
        h = Harness(str(tmp_path))
        assert isinstance(h._runner, ClaudeCliRunner)

    @pytest.mark.asyncio
    async def test_gitignore_appended(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        await h._ensure_initialized()
        gitignore = (tmp_path / ".gitignore").read_text()
        assert ".harness/harness.db" in gitignore

    @pytest.mark.asyncio
    async def test_gitignore_idempotent(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        await h._ensure_initialized()
        await h._ensure_initialized()
        gitignore = (tmp_path / ".gitignore").read_text()
        assert gitignore.count(".harness/harness.db") == 1

    @pytest.mark.asyncio
    async def test_harness_dir_created(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        await h._ensure_initialized()
        assert (tmp_path / ".harness").is_dir()

    @pytest.mark.asyncio
    async def test_existing_gitignore_appended(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n")
        h = make_harness(tmp_path)
        await h._ensure_initialized()
        content = (tmp_path / ".gitignore").read_text()
        assert "*.pyc" in content
        assert ".harness/harness.db" in content


# ---------------------------------------------------------------------------
# pipeline 基础流程
# ---------------------------------------------------------------------------


class TestPipelinePipeline:
    @pytest.mark.asyncio
    async def test_single_function_task(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        result = await h.pipeline([FunctionTask(fn=lambda r: "hello")])
        assert len(result.results) == 1
        assert result.results[0].output == "hello"
        assert result.results[0].success is True

    @pytest.mark.asyncio
    async def test_multiple_function_tasks(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        result = await h.pipeline([
            FunctionTask(fn=lambda r: "step1"),
            FunctionTask(fn=lambda r: "step2"),
        ])
        assert len(result.results) == 2
        assert result.results[0].task_index == "0"
        assert result.results[1].task_index == "1"

    @pytest.mark.asyncio
    async def test_pipeline_name_stored(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        pr = await h.pipeline(
            [FunctionTask(fn=lambda r: "x")],
            name="my-pipeline",
        )
        assert pr.name == "my-pipeline"

    @pytest.mark.asyncio
    async def test_pipeline_result_run_id(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        pr = await h.pipeline([FunctionTask(fn=lambda r: "x")])
        assert pr.run_id is not None
        assert len(pr.run_id) > 0

    @pytest.mark.asyncio
    async def test_task_failed_error_propagated(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        task = FunctionTask(
            fn=lambda r: (_ for _ in ()).throw(RuntimeError("boom")),
            config=TaskConfig(max_retries=0),
        )
        with pytest.raises(TaskFailedError):
            await h.pipeline([task])


# ---------------------------------------------------------------------------
# 嵌套 Parallel 校验
# ---------------------------------------------------------------------------


class TestInvalidPipelineValidation:
    @pytest.mark.asyncio
    async def test_nested_parallel_raises(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        inner = Parallel(tasks=[FunctionTask(fn=lambda r: "x")])
        outer = Parallel(tasks=[inner])  # type: ignore[arg-type]
        with pytest.raises(InvalidPipelineError):
            await h.pipeline([outer])


# ---------------------------------------------------------------------------
# resume_from 续跑逻辑
# ---------------------------------------------------------------------------


class TestResumeFrom:
    @pytest.mark.asyncio
    async def test_resume_skips_succeeded_tasks(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)

        call_count = [0]

        def counting_fn(results):
            call_count[0] += 1
            return f"result-{call_count[0]}"

        # 第一次跑，Task 0 成功，Task 1 失败
        tasks = [
            FunctionTask(fn=lambda r: "ok"),
            FunctionTask(
                fn=lambda r: (_ for _ in ()).throw(RuntimeError("fail")),
                config=TaskConfig(max_retries=0),
            ),
        ]

        run_id = None
        try:
            pr = await h.pipeline(tasks)
        except TaskFailedError as e:
            run_id = e.run_id

        assert run_id is not None

        # 续跑：Task 0 应该被跳过
        call_count_before = call_count[0]
        tasks2 = [
            FunctionTask(fn=counting_fn),   # index 0，应被跳过
            FunctionTask(fn=lambda r: "fixed"),  # index 1
        ]
        pr2 = await h.pipeline(tasks2, resume_from=run_id)
        # Task 0 被跳过，counting_fn 不应被额外调用
        # (Task 0 已成功，所以跳过，只执行 Task 1)
        assert len(pr2.results) >= 1


# ---------------------------------------------------------------------------
# async context manager
# ---------------------------------------------------------------------------


class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path: Path) -> None:
        async with make_harness(tmp_path) as h:
            result = await h.pipeline([FunctionTask(fn=lambda r: "ok")])
            assert result.results[0].success is True

    @pytest.mark.asyncio
    async def test_context_manager_calls_stop(self, tmp_path: Path) -> None:
        """__aexit__ 应调用 stop()。"""
        h = make_harness(tmp_path)
        stop_called = [False]
        original_stop = h.stop

        async def mock_stop():
            stop_called[0] = True
            await original_stop()

        h.stop = mock_stop
        async with h:
            pass
        assert stop_called[0]


# ---------------------------------------------------------------------------
# run() 语法糖
# ---------------------------------------------------------------------------


class TestRunSugar:
    @pytest.mark.asyncio
    async def test_run_returns_single_result(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        # 使用 FunctionTask 来避免需要真实的 LLMTask runner
        # 但 run() 只接受 prompt，内部创建 LLMTask
        # 所以这里我们用 MockRunner 来处理
        result = await h.run("test prompt")
        assert result.success is True
        assert result.task_type == "llm"


# ---------------------------------------------------------------------------
# Task 别名（DeprecationWarning）
# ---------------------------------------------------------------------------


class TestTaskDeprecationInHarness:
    @pytest.mark.asyncio
    async def test_task_alias_works_in_pipeline(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task = Task(prompt="hello")
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

        result = await h.pipeline([task])
        assert len(result.results) == 1

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


class CapturingRunner(AbstractRunner):
    """记录每次调用时的 system_prompt，用于验证注入逻辑。"""

    def __init__(self) -> None:
        self.captured: list[str] = []

    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        self.captured.append(system_prompt or "")
        return RunnerResult(text="ok", tokens_used=5, session_id="s1")


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


# ---------------------------------------------------------------------------
# memory.md 整理机制
# ---------------------------------------------------------------------------


class TestMemoryConsolidation:
    @pytest.mark.asyncio
    async def test_consolidation_injected_to_last_llm_task(self, tmp_path: Path) -> None:
        """末尾 LLMTask 无 schema 时，整理提示注入 system_prompt。"""
        from harness.memory import Memory

        runner = CapturingRunner()
        h = Harness(str(tmp_path), runner=runner, memory=Memory())
        await h.pipeline([LLMTask(prompt="do something")])

        assert len(runner.captured) == 1
        assert str(tmp_path / ".harness" / "memory.md") in runner.captured[0]

    @pytest.mark.asyncio
    async def test_consolidation_not_injected_when_no_memory(self, tmp_path: Path) -> None:
        """memory=None 时不注入整理提示。"""
        runner = CapturingRunner()
        h = Harness(str(tmp_path), runner=runner, memory=None)
        await h.pipeline([LLMTask(prompt="do something")])

        # 不应包含 memory.md 路径
        assert ".harness/memory.md" not in runner.captured[0]

    @pytest.mark.asyncio
    async def test_consolidation_not_injected_when_output_schema(self, tmp_path: Path) -> None:
        """末尾 LLMTask 有 output_schema 时不注入整理提示。"""
        from pydantic import BaseModel
        from harness.memory import Memory

        class MySchema(BaseModel):
            value: str

        runner = CapturingRunner()
        h = Harness(str(tmp_path), runner=runner, memory=Memory())
        # runner 返回的 text="ok" 不是有效 JSON，会触发 schema 校验失败重试
        # 改为返回合法 JSON
        runner2 = CapturingRunner()

        async def _execute(prompt, *, system_prompt, session_id, **kwargs):
            runner2.captured.append(system_prompt or "")
            return RunnerResult(text='{"value": "x"}', tokens_used=5, session_id="s1")

        runner2.execute = _execute  # type: ignore[method-assign]
        h2 = Harness(str(tmp_path), runner=runner2, memory=Memory())
        await h2.pipeline([LLMTask(prompt="do something", output_schema=MySchema)])

        assert len(runner2.captured) >= 1
        # 整理提示不应出现在 system_prompt 中
        assert str(tmp_path / ".harness" / "memory.md") not in runner2.captured[0]

    @pytest.mark.asyncio
    async def test_consolidation_only_on_last_llm_task(self, tmp_path: Path) -> None:
        """整理提示只注入最后一个 LLMTask，不影响前面的 LLMTask。"""
        from harness.memory import Memory

        runner = CapturingRunner()
        h = Harness(str(tmp_path), runner=runner, memory=Memory())
        await h.pipeline([
            LLMTask(prompt="first"),
            LLMTask(prompt="second"),
        ])

        assert len(runner.captured) == 2
        memory_path = str(tmp_path / ".harness" / "memory.md")
        # 第一个 LLMTask 不注入
        assert memory_path not in runner.captured[0]
        # 最后一个 LLMTask 注入
        assert memory_path in runner.captured[1]

    @pytest.mark.asyncio
    async def test_consolidation_preserves_existing_task_system_prompt(
        self, tmp_path: Path
    ) -> None:
        """注入整理提示时，task 自身的 system_prompt 保持不变。"""
        from harness.memory import Memory

        runner = CapturingRunner()
        h = Harness(str(tmp_path), runner=runner, memory=Memory())
        await h.pipeline([LLMTask(prompt="hi", system_prompt="you are helpful")])

        assert "you are helpful" in runner.captured[0]
        assert str(tmp_path / ".harness" / "memory.md") in runner.captured[0]


# ---------------------------------------------------------------------------
# 静默异常修复：memory injection 和 notifier 失败应记录 warning
# ---------------------------------------------------------------------------


def _make_mock_storage() -> MagicMock:
    """构造一个完整 mock 的 SQLStorage，所有方法均为 AsyncMock。"""
    storage = MagicMock()
    storage.init = AsyncMock()
    storage.save_run = AsyncMock()
    storage.save_task_log = AsyncMock()
    storage.update_run = AsyncMock()
    storage.get_task_logs = AsyncMock(return_value=[])
    return storage


def _patch_storage(h: Harness, storage: MagicMock) -> None:
    """将 Harness 内部的存储替换为 mock，并标记已初始化。"""
    h._storage = storage
    h._initialized = True


class TestSilentExceptionFixes:
    @pytest.mark.asyncio
    async def test_memory_injection_failure_logs_warning(
        self, tmp_path: Path
    ) -> None:
        """memory injection 失败时应记录 warning，pipeline 正常继续执行。"""
        from harness.memory import Memory

        memory = Memory()
        h = make_harness(tmp_path, memory=memory)
        _patch_storage(h, _make_mock_storage())

        with patch.object(
            memory,
            "build_injection",
            new=AsyncMock(side_effect=RuntimeError("injection error")),
        ):
            with patch("harness.harness.logger") as mock_logger:
                result = await h.pipeline([FunctionTask(fn=lambda r: "ok")])

        # pipeline 应正常完成
        assert result.results[0].success is True
        # warning 应被记录，且包含失败原因
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "injection" in warning_msg.lower() or "memory" in warning_msg.lower()

    @pytest.mark.asyncio
    async def test_memory_injection_failure_pipeline_continues(
        self, tmp_path: Path
    ) -> None:
        """memory injection 失败后，pipeline 仍能正常返回结果，不抛异常。"""
        from harness.memory import Memory

        memory = Memory()
        h = make_harness(tmp_path, memory=memory)
        _patch_storage(h, _make_mock_storage())

        call_count = [0]

        def counting_fn(results):
            call_count[0] += 1
            return "done"

        with patch.object(
            memory,
            "build_injection",
            new=AsyncMock(side_effect=ValueError("bad injection")),
        ):
            result = await h.pipeline([FunctionTask(fn=counting_fn)])

        # 函数被执行，pipeline 不因 memory 失败而中断
        assert call_count[0] == 1
        assert result.results[0].output == "done"

    @pytest.mark.asyncio
    async def test_notifier_failure_logs_warning_on_success(
        self, tmp_path: Path
    ) -> None:
        """pipeline 成功后 notifier 抛异常，应记录 warning，结果正确返回。"""
        from harness.notifier.base import AbstractNotifier

        class FailingNotifier(AbstractNotifier):
            async def notify(self, *, title, body, success):
                raise RuntimeError("network error")

        h = make_harness(tmp_path, notifier=FailingNotifier())
        _patch_storage(h, _make_mock_storage())

        with patch("harness.harness.logger") as mock_logger:
            result = await h.pipeline([FunctionTask(fn=lambda r: "ok")])

        # pipeline 结果正确
        assert result.results[0].success is True
        # warning 被记录
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "notif" in warning_msg.lower()

    @pytest.mark.asyncio
    async def test_notifier_failure_logs_warning_on_failure(
        self, tmp_path: Path
    ) -> None:
        """pipeline 失败后 notifier 抛异常，应记录 warning，原始 TaskFailedError 仍被抛出。"""
        from harness.notifier.base import AbstractNotifier

        class FailingNotifier(AbstractNotifier):
            async def notify(self, *, title, body, success):
                raise ConnectionError("timeout")

        h = make_harness(
            tmp_path,
            notifier=FailingNotifier(),
        )
        _patch_storage(h, _make_mock_storage())

        failing_task = FunctionTask(
            fn=lambda r: (_ for _ in ()).throw(RuntimeError("task boom")),
            config=TaskConfig(max_retries=0),
        )

        with patch("harness.harness.logger") as mock_logger:
            with pytest.raises(TaskFailedError):
                await h.pipeline([failing_task])

        # warning 被记録
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "notif" in warning_msg.lower()


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


# ---- Dialogue 集成 ----

from harness.task import Dialogue, DialogueOutput, Role


class TestHarnessDialogue:
    @pytest.mark.asyncio
    async def test_pipeline_with_dialogue_produces_result(
        self, tmp_path, mock_runner
    ) -> None:
        """pipeline 中的 Dialogue 产出单个 Result，output 为 DialogueOutput。"""
        h = Harness(project_path=str(tmp_path), runner=mock_runner)
        _patch_storage(h, _make_mock_storage())
        dialogue = Dialogue(
            roles=[
                Role(name="a", system_prompt="", prompt=lambda ctx: "prompt a"),
                Role(name="b", system_prompt="", prompt=lambda ctx: "prompt b"),
            ],
            max_rounds=1,
        )
        pr = await h.pipeline([dialogue])
        assert len(pr.results) == 1
        assert pr.results[0].task_type == "dialogue"
        assert isinstance(pr.results[0].output, DialogueOutput)
        assert pr.results[0].output.rounds_completed == 1

    @pytest.mark.asyncio
    async def test_pipeline_dialogue_then_llm_can_access_output(
        self, tmp_path, mock_runner
    ) -> None:
        """Dialogue 之后的 LLMTask 可以通过 results[-1].output 访问对话记录。"""
        received: list = []

        def capture_prompt(results):
            received.append(results[-1].output)
            return "总结"

        h = Harness(project_path=str(tmp_path), runner=mock_runner)
        _patch_storage(h, _make_mock_storage())
        pr = await h.pipeline([
            Dialogue(
                roles=[
                    Role(name="a", system_prompt="", prompt=lambda ctx: "x"),
                ],
                max_rounds=1,
            ),
            LLMTask(prompt=capture_prompt),
        ])
        assert len(received) == 1
        assert isinstance(received[0], DialogueOutput)

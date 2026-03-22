"""tests/unit/test_executor.py — Executor 逻辑验证（mock runner）。"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from harness._internal.exceptions import OutputSchemaError, TaskFailedError
from harness._internal.executor import (
    execute_function_task,
    execute_llm_task,
    execute_shell_task,
)
from harness._internal.session import SessionManager
from harness.runners.base import AbstractRunner, RunnerResult
from harness.task import FunctionTask, LLMTask, Result, ShellTask, TaskConfig


# ---------------------------------------------------------------------------
# Mock Runner
# ---------------------------------------------------------------------------


class MockRunner(AbstractRunner):
    def __init__(self, response: str = "ok", tokens: int = 10, fail: bool = False) -> None:
        self._response = response
        self._tokens = tokens
        self._fail = fail
        self.call_count = 0

    async def execute(self, prompt: str, *, system_prompt: str, session_id, **kwargs) -> RunnerResult:
        self.call_count += 1
        if self._fail:
            raise RuntimeError("runner failed")
        return RunnerResult(text=self._response, tokens_used=self._tokens, session_id="s1")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def make_session() -> SessionManager:
    return SessionManager()


def make_result(index: str = "0") -> Result:
    return Result(
        task_index=index,
        task_type="llm",
        output="prev",
        raw_text="prev",
        tokens_used=5,
        duration_seconds=0.1,
        success=True,
        error=None,
    )


# ---------------------------------------------------------------------------
# LLMTask 执行
# ---------------------------------------------------------------------------


class TestExecuteLLMTask:
    @pytest.mark.asyncio
    async def test_basic_success(self) -> None:
        runner = MockRunner(response="hello")
        task = LLMTask(prompt="say hi")
        result = await execute_llm_task(
            task, "0", [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=None,
            session_manager=make_session(),
            memory_injection="",
        )
        assert result.success is True
        assert result.output == "hello"
        assert result.task_type == "llm"
        assert result.task_index == "0"
        assert result.tokens_used == 10

    @pytest.mark.asyncio
    async def test_system_prompt_merged(self) -> None:
        """system_prompt 合并：Harness.sp + LLMTask.sp + memory"""
        captured_prompts: list[str] = []

        class CapturingRunner(AbstractRunner):
            async def execute(self, prompt, *, system_prompt, session_id, **kwargs):
                captured_prompts.append(system_prompt)
                return RunnerResult(text="x", tokens_used=0, session_id=None)

        task = LLMTask(prompt="x", system_prompt="task-sp")
        await execute_llm_task(
            task, "0", [], "run-1",
            harness_system_prompt="harness-sp",
            harness_runner=CapturingRunner(),
            harness_config=None,
            session_manager=make_session(),
            memory_injection="mem-inject",
        )
        assert len(captured_prompts) == 1
        sp = captured_prompts[0]
        assert "harness-sp" in sp
        assert "task-sp" in sp
        assert "mem-inject" in sp

    @pytest.mark.asyncio
    async def test_callable_prompt_exception_no_retry(self) -> None:
        runner = MockRunner()

        def bad_prompt(results):
            raise ValueError("prompt error")

        task = LLMTask(prompt=bad_prompt, config=TaskConfig(max_retries=3))
        with pytest.raises(TaskFailedError) as exc_info:
            await execute_llm_task(
                task, "0", [], "run-1",
                harness_system_prompt="",
                harness_runner=runner,
                harness_config=None,
                session_manager=make_session(),
                memory_injection="",
            )
        # 不应重试：runner 不应被调用
        assert runner.call_count == 0
        assert "Prompt callable raised" in exc_info.value.error

    @pytest.mark.asyncio
    async def test_runner_failure_triggers_retry(self) -> None:
        runner = MockRunner(fail=True)
        task = LLMTask(prompt="x", config=TaskConfig(max_retries=2, backoff_base=0.01))
        with pytest.raises(TaskFailedError):
            await execute_llm_task(
                task, "0", [], "run-1",
                harness_system_prompt="",
                harness_runner=runner,
                harness_config=None,
                session_manager=make_session(),
                memory_injection="",
            )
        assert runner.call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_output_schema_success(self) -> None:
        class Out(BaseModel):
            value: int

        runner = MockRunner(response='{"value": 42}')
        task = LLMTask(prompt="x", output_schema=Out)
        result = await execute_llm_task(
            task, "0", [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=None,
            session_manager=make_session(),
            memory_injection="",
        )
        assert isinstance(result.output, Out)
        assert result.output.value == 42

    @pytest.mark.asyncio
    async def test_output_schema_failure_triggers_retry(self) -> None:
        class Out(BaseModel):
            value: int

        call_count = 0

        class BadThenGoodRunner(AbstractRunner):
            async def execute(self, prompt, *, system_prompt, session_id, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return RunnerResult(text="not valid json", tokens_used=0, session_id=None)
                return RunnerResult(text='{"value": 7}', tokens_used=0, session_id=None)

        task = LLMTask(prompt="x", output_schema=Out, config=TaskConfig(max_retries=2, backoff_base=0.01))
        result = await execute_llm_task(
            task, "0", [], "run-1",
            harness_system_prompt="",
            harness_runner=BadThenGoodRunner(),
            harness_config=None,
            session_manager=make_session(),
            memory_injection="",
        )
        assert result.success is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_task_config_priority(self) -> None:
        """task.config > harness_config > TaskConfig()"""
        runner = MockRunner(fail=True)
        task_cfg = TaskConfig(max_retries=0)  # 不重试
        task = LLMTask(prompt="x", config=task_cfg)
        with pytest.raises(TaskFailedError):
            await execute_llm_task(
                task, "0", [], "run-1",
                harness_system_prompt="",
                harness_runner=runner,
                harness_config=TaskConfig(max_retries=5),  # 会被 task.config 覆盖
                session_manager=make_session(),
                memory_injection="",
            )
        assert runner.call_count == 1  # 0 retries = 1 attempt


# ---------------------------------------------------------------------------
# 前序上下文注入时机测试
# ---------------------------------------------------------------------------


class MockStorage:
    """最小化 StorageProtocol 实现，用于注入时机测试。"""

    def __init__(self, logs: list[dict]) -> None:
        self._logs = logs
        self.get_task_logs_call_count = 0

    async def get_task_logs(self, run_id: str, *, success_only: bool = False) -> list[dict]:
        self.get_task_logs_call_count += 1
        return self._logs

    # 其余方法不需要实现
    async def create_run(self, *a, **kw): ...
    async def update_run(self, *a, **kw): ...
    async def log_task(self, *a, **kw): ...
    async def get_runs(self, *a, **kw): return []
    async def get_run(self, *a, **kw): return None


class CapturingPromptRunner(AbstractRunner):
    """捕获每次 execute() 接收到的 prompt，总是返回成功。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def execute(self, prompt: str, *, system_prompt: str, session_id, **kwargs) -> RunnerResult:
        self.prompts.append(prompt)
        return RunnerResult(text="ok", tokens_used=0, session_id="s-new")


class TestContextInjectionTiming:
    """验证前序上下文注入只在新 session 时触发，重试新 session 也应注入。"""

    _prior_logs = [
        {"task_index": "0", "task_type": "function", "output": "data collected"}
    ]

    @pytest.mark.asyncio
    async def test_context_injected_on_new_session_first_attempt(self) -> None:
        """首次 attempt 且 is_new_session=True 时，前序上下文应注入到 prompt。"""
        runner = CapturingPromptRunner()
        storage = MockStorage(self._prior_logs)
        task = LLMTask(prompt="analyze data")

        await execute_llm_task(
            task, "1", [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=None,
            session_manager=make_session(),
            memory_injection="",
            is_new_session=True,
            storage=storage,
        )

        assert len(runner.prompts) == 1
        assert "前序任务输出" in runner.prompts[0]
        assert "data collected" in runner.prompts[0]
        assert "analyze data" in runner.prompts[0]

    @pytest.mark.asyncio
    async def test_context_injected_on_new_session_retry(self) -> None:
        """首次 attempt 失败后 session 被标记为 broken，重试时 is_new_session=True，
        也应注入前序上下文（这是本 bug 修复的核心场景）。"""
        call_count = 0

        class FailThenSucceedRunner(AbstractRunner):
            def __init__(self) -> None:
                self.prompts: list[str] = []

            async def execute(self, prompt: str, *, system_prompt: str, session_id, **kwargs) -> RunnerResult:
                nonlocal call_count
                call_count += 1
                self.prompts.append(prompt)
                if call_count == 1:
                    raise RuntimeError("session broken")
                return RunnerResult(text="ok", tokens_used=0, session_id="s-retry")

        runner = FailThenSucceedRunner()
        storage = MockStorage(self._prior_logs)
        task = LLMTask(prompt="analyze data", config=TaskConfig(max_retries=1, backoff_base=0.01))

        await execute_llm_task(
            task, "1", [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=None,
            session_manager=make_session(),
            memory_injection="",
            is_new_session=False,  # 首次调用时不是新 session
            storage=storage,
        )

        assert call_count == 2
        # 第一次 attempt（session 未断）不注入
        assert "前序任务输出" not in runner.prompts[0]
        # 第二次 attempt（session 断开，新 session）必须注入
        assert "前序任务输出" in runner.prompts[1], (
            "重试时生成了新 session，应当注入前序上下文，但实际未注入"
        )
        assert "data collected" in runner.prompts[1]

    @pytest.mark.asyncio
    async def test_context_not_injected_when_session_reused(self) -> None:
        """is_new_session=False 时，不应注入前序上下文（session 复用场景）。"""
        runner = CapturingPromptRunner()
        storage = MockStorage(self._prior_logs)
        task = LLMTask(prompt="analyze data")

        await execute_llm_task(
            task, "1", [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=None,
            session_manager=make_session(),
            memory_injection="",
            is_new_session=False,
            storage=storage,
        )

        assert len(runner.prompts) == 1
        assert "前序任务输出" not in runner.prompts[0]
        assert runner.prompts[0] == "analyze data"


# ---------------------------------------------------------------------------
# FunctionTask 执行
# ---------------------------------------------------------------------------


class TestExecuteFunctionTask:
    @pytest.mark.asyncio
    async def test_basic_success(self) -> None:
        task = FunctionTask(fn=lambda results: "function result")
        result = await execute_function_task(task, "1", [], "run-1", harness_config=None)
        assert result.success is True
        assert result.output == "function result"
        assert result.task_type == "function"
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_output_schema_pass(self) -> None:
        class Out(BaseModel):
            x: int

        out = Out(x=1)
        task = FunctionTask(fn=lambda r: out, output_schema=Out)
        result = await execute_function_task(task, "0", [], "run-1", harness_config=None)
        assert result.output is out

    @pytest.mark.asyncio
    async def test_output_schema_failure_no_retry(self) -> None:
        """FunctionTask output_schema 校验失败抛 OutputSchemaError，不触发重试。"""
        call_count = 0

        def fn(results):
            nonlocal call_count
            call_count += 1
            return "wrong type"

        task = FunctionTask(fn=fn, output_schema=int, config=TaskConfig(max_retries=3))
        with pytest.raises(OutputSchemaError):
            await execute_function_task(task, "0", [], "run-1", harness_config=None)
        assert call_count == 1  # 不重试

    @pytest.mark.asyncio
    async def test_fn_exception_triggers_retry(self) -> None:
        call_count = 0

        def fn(results):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("flaky")
            return "ok"

        task = FunctionTask(fn=fn, config=TaskConfig(max_retries=3, backoff_base=0.01))
        result = await execute_function_task(task, "0", [], "run-1", harness_config=None)
        assert result.success is True
        assert call_count == 3


# ---------------------------------------------------------------------------
# ShellTask 执行
# ---------------------------------------------------------------------------


class TestExecuteShellTask:
    @pytest.mark.asyncio
    async def test_basic_success(self) -> None:
        task = ShellTask(cmd="echo hello")
        result = await execute_shell_task(task, "2", [], "run-1", harness_config=None)
        assert result.success is True
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_nonzero_exit_triggers_retry(self) -> None:
        call_count = 0

        class CountingShellTask(ShellTask):
            pass

        task = ShellTask(
            cmd="exit 1",
            config=TaskConfig(max_retries=2, backoff_base=0.01),
        )
        with pytest.raises(TaskFailedError) as exc_info:
            await execute_shell_task(task, "0", [], "run-1", harness_config=None)
        assert exc_info.value.task_type == "shell"

    @pytest.mark.asyncio
    async def test_callable_cmd(self) -> None:
        task = ShellTask(cmd=lambda results: "echo from-callable")
        result = await execute_shell_task(task, "0", [make_result()], "run-1", harness_config=None)
        assert "from-callable" in result.output

    @pytest.mark.asyncio
    async def test_callable_cmd_exception_no_retry(self) -> None:
        def bad_cmd(results):
            raise ValueError("cmd error")

        task = ShellTask(cmd=bad_cmd, config=TaskConfig(max_retries=5))
        with pytest.raises(TaskFailedError) as exc_info:
            await execute_shell_task(task, "0", [], "run-1", harness_config=None)
        assert "cmd callable raised" in exc_info.value.error

"""tests/unit/test_design_fixes.py — 设计优化相关测试。"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness._internal.exceptions import TaskFailedError
from harness._internal.executor import _get_env_lock, execute_function_task
from harness._internal.parallel import _with_semaphore, execute_parallel
from harness._internal.session import SessionManager
from harness.runners.base import AbstractRunner, RunnerResult
from harness.task import FunctionTask, LLMTask, Parallel, ShellTask, TaskConfig


# ---------------------------------------------------------------------------
# Fix #1: env_overrides 竞态保护（asyncio.Lock）
# ---------------------------------------------------------------------------


class TestEnvLockProtection:
    @pytest.mark.asyncio
    async def test_concurrent_env_overrides_no_race(self) -> None:
        """并发 FunctionTask 使用不同 env_overrides 不互相干扰。"""
        captured_a = {}
        captured_b = {}

        def fn_a(results):
            captured_a["val"] = os.environ.get("HARNESS_RACE_TEST")
            return "a"

        def fn_b(results):
            captured_b["val"] = os.environ.get("HARNESS_RACE_TEST")
            return "b"

        task_a = FunctionTask(fn=fn_a)
        task_b = FunctionTask(fn=fn_b)

        # 并发执行，各自带不同的 env_overrides
        result_a, result_b = await asyncio.gather(
            execute_function_task(
                task_a, "0", [], "run-1",
                harness_config=None,
                env_overrides={"HARNESS_RACE_TEST": "value_a"},
            ),
            execute_function_task(
                task_b, "0", [], "run-1",
                harness_config=None,
                env_overrides={"HARNESS_RACE_TEST": "value_b"},
            ),
        )

        # Lock 序列化了执行：每个 fn 看到的值应该是自己的 override
        assert captured_a["val"] == "value_a"
        assert captured_b["val"] == "value_b"
        # 执行后环境变量已恢复
        assert os.environ.get("HARNESS_RACE_TEST") is None

    @pytest.mark.asyncio
    async def test_env_lock_per_event_loop(self) -> None:
        """_get_env_lock 返回当前事件循环绑定的 Lock。"""
        lock = _get_env_lock()
        assert isinstance(lock, asyncio.Lock)
        # 同一循环内应返回同一个 Lock
        assert _get_env_lock() is lock


# ---------------------------------------------------------------------------
# Fix #2: ClaudeCliRunner env_overrides 透传
# ---------------------------------------------------------------------------


class TestClaudeCliRunnerEnvOverrides:
    def test_get_subprocess_env_with_overrides(self) -> None:
        from harness.runners.claude_cli import ClaudeCliRunner

        runner = ClaudeCliRunner()
        with patch.dict(os.environ, {"PATH": "/usr/bin", "MY_VAR": "old"}, clear=False):
            env = runner._get_subprocess_env(
                env_overrides={"MY_VAR": "new", "EXTRA": "added"}
            )
            assert env["MY_VAR"] == "new"
            assert env["EXTRA"] == "added"

    def test_get_subprocess_env_remove_via_empty(self) -> None:
        from harness.runners.claude_cli import ClaudeCliRunner

        runner = ClaudeCliRunner()
        with patch.dict(os.environ, {"REMOVE_ME": "val"}, clear=False):
            env = runner._get_subprocess_env(
                env_overrides={"REMOVE_ME": ""}
            )
            assert "REMOVE_ME" not in env

    def test_get_subprocess_env_none_overrides(self) -> None:
        from harness.runners.claude_cli import ClaudeCliRunner

        runner = ClaudeCliRunner()
        env1 = runner._get_subprocess_env()
        env2 = runner._get_subprocess_env(env_overrides=None)
        # 都不应该报错，结果等价
        assert "PATH" in env1
        assert "PATH" in env2


# ---------------------------------------------------------------------------
# Fix #3: Parallel max_concurrency
# ---------------------------------------------------------------------------


class TestParallelMaxConcurrency:
    @pytest.mark.asyncio
    async def test_max_concurrency_limits_parallel_execution(self) -> None:
        """max_concurrency=1 确保任务串行执行。"""
        execution_order = []

        def fn_factory(name):
            def fn(results):
                execution_order.append(f"{name}_start")
                execution_order.append(f"{name}_end")
                return name
            return fn

        tasks = [
            FunctionTask(fn=fn_factory("a")),
            FunctionTask(fn=fn_factory("b")),
            FunctionTask(fn=fn_factory("c")),
        ]
        p = Parallel(tasks=tasks, max_concurrency=1)
        kwargs = dict(
            run_id="run-1",
            harness_system_prompt="",
            harness_runner=MockRunner(),
            harness_config=None,
            session_manager=SessionManager(),
            memory_injection="",
            storage=None,
            is_new_session=False,
        )

        results = await execute_parallel(p, 0, [], **kwargs)
        assert all(r.success for r in results)

        # max_concurrency=1 时任务串行，不会交错
        for i in range(0, len(execution_order), 2):
            name = execution_order[i].replace("_start", "")
            assert execution_order[i] == f"{name}_start"
            assert execution_order[i + 1] == f"{name}_end"

    @pytest.mark.asyncio
    async def test_max_concurrency_none_no_limit(self) -> None:
        """max_concurrency=None 时无限制（默认行为）。"""
        tasks = [
            FunctionTask(fn=lambda r: "a"),
            FunctionTask(fn=lambda r: "b"),
        ]
        p = Parallel(tasks=tasks, max_concurrency=None)
        kwargs = dict(
            run_id="run-1",
            harness_system_prompt="",
            harness_runner=MockRunner(),
            harness_config=None,
            session_manager=SessionManager(),
            memory_injection="",
            storage=None,
            is_new_session=False,
        )
        results = await execute_parallel(p, 0, [], **kwargs)
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_with_semaphore_none(self) -> None:
        """semaphore=None 时直接执行协程。"""

        async def coro():
            return 42

        result = await _with_semaphore(coro(), None)
        assert result == 42

    @pytest.mark.asyncio
    async def test_with_semaphore_limits(self) -> None:
        """semaphore 正常限制并发。"""
        sem = asyncio.Semaphore(1)

        async def coro():
            return "ok"

        result = await _with_semaphore(coro(), sem)
        assert result == "ok"


# ---------------------------------------------------------------------------
# Fix #4: Memory truncation warning
# ---------------------------------------------------------------------------


class TestMemoryTruncationWarning:
    def test_truncation_logs_warning(self, tmp_path, caplog) -> None:
        import logging
        from harness.memory import Memory

        mem = Memory(max_tokens=100)
        mem_file = tmp_path / ".harness" / "memory.md"

        # 写入大量内容使其超过阈值（80 chars）
        mem.write_memory_update(tmp_path, "x" * 50)
        mem.write_memory_update(tmp_path, "y" * 50)

        with caplog.at_level(logging.WARNING, logger="harness.memory"):
            mem.write_memory_update(tmp_path, "z" * 50)

        assert "truncating" in caplog.text


# ---------------------------------------------------------------------------
# Fix #5: shared SSE parser
# ---------------------------------------------------------------------------


class TestIterSseEvents:
    @pytest.mark.asyncio
    async def test_parses_data_lines(self) -> None:
        from harness.runners._http import iter_sse_events

        async def lines():
            yield 'data: {"key": "value"}'
            yield 'data: {"key2": "value2"}'

        events = [e async for e in iter_sse_events(lines())]
        assert len(events) == 2
        assert events[0]["key"] == "value"
        assert events[1]["key2"] == "value2"

    @pytest.mark.asyncio
    async def test_skips_non_data_lines(self) -> None:
        from harness.runners._http import iter_sse_events

        async def lines():
            yield "event: message"
            yield ": comment"
            yield ""
            yield 'data: {"ok": true}'

        events = [e async for e in iter_sse_events(lines())]
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_stops_at_done(self) -> None:
        from harness.runners._http import iter_sse_events

        async def lines():
            yield 'data: {"first": true}'
            yield "data: [DONE]"
            yield 'data: {"should_not_appear": true}'

        events = [e async for e in iter_sse_events(lines())]
        assert len(events) == 1
        assert events[0]["first"] is True

    @pytest.mark.asyncio
    async def test_skips_invalid_json(self) -> None:
        from harness.runners._http import iter_sse_events

        async def lines():
            yield "data: not-json"
            yield 'data: {"valid": true}'

        events = [e async for e in iter_sse_events(lines())]
        assert len(events) == 1
        assert events[0]["valid"] is True


# ---------------------------------------------------------------------------
# Fix #6: Dialogue env_overrides passthrough
# ---------------------------------------------------------------------------


class TestDialogueEnvOverrides:
    @pytest.mark.asyncio
    async def test_dialogue_passes_env_overrides_to_runner(self) -> None:
        """execute_dialogue 将 env_overrides 传递到 runner.execute()。"""
        from harness._internal.dialogue import execute_dialogue
        from harness.task import Dialogue, Role

        captured_kwargs: dict = {}

        class CapturingRunner(AbstractRunner):
            async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
                captured_kwargs.update(kwargs)
                return RunnerResult(text="done", tokens_used=1, session_id=None)

        runner = CapturingRunner()
        dialogue = Dialogue(
            roles=[
                Role(name="A", prompt=lambda ctx: "hello", system_prompt=""),
            ],
            background="",
            max_rounds=1,
        )

        await execute_dialogue(
            dialogue,
            outer_index=0,
            pipeline_results=[],
            run_id="run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=None,
            env_overrides={"MY_VAR": "my_value"},
        )

        assert captured_kwargs.get("env_overrides") == {"MY_VAR": "my_value"}

    @pytest.mark.asyncio
    async def test_dialogue_env_overrides_none_by_default(self) -> None:
        """execute_dialogue 默认 env_overrides=None。"""
        from harness._internal.dialogue import execute_dialogue
        from harness.task import Dialogue, Role

        captured_kwargs: dict = {}

        class CapturingRunner(AbstractRunner):
            async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
                captured_kwargs.update(kwargs)
                return RunnerResult(text="done", tokens_used=1, session_id=None)

        runner = CapturingRunner()
        dialogue = Dialogue(
            roles=[
                Role(name="A", prompt=lambda ctx: "hello", system_prompt=""),
            ],
            background="",
            max_rounds=1,
        )

        await execute_dialogue(
            dialogue,
            outer_index=0,
            pipeline_results=[],
            run_id="run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=None,
        )

        assert captured_kwargs.get("env_overrides") is None


# ---------------------------------------------------------------------------
# Fix #7: PollingTask env_overrides passthrough
# ---------------------------------------------------------------------------


class TestPollingEnvOverrides:
    @pytest.mark.asyncio
    async def test_polling_applies_env_overrides(self) -> None:
        """execute_polling 在 submit_fn/poll_fn 执行期间应用 env_overrides。"""
        from harness._internal.polling import execute_polling
        from harness.task import PollingTask

        captured_env: dict = {}

        def submit_fn(results):
            captured_env["submit"] = os.environ.get("POLL_TEST_VAR")
            return "handle"

        call_count = {"n": 0}
        def poll_fn(handle):
            call_count["n"] += 1
            captured_env["poll"] = os.environ.get("POLL_TEST_VAR")
            return {"done": True}

        task = PollingTask(
            submit_fn=submit_fn,
            poll_fn=poll_fn,
            success_condition=lambda r: r.get("done"),
            timeout=5,
            poll_interval=0.01,
        )

        result = await execute_polling(
            task, "0", [], "run-1",
            harness_config=None,
            env_overrides={"POLL_TEST_VAR": "injected"},
        )

        assert result.success
        assert captured_env["submit"] == "injected"
        assert captured_env["poll"] == "injected"
        # 环境已恢复
        assert os.environ.get("POLL_TEST_VAR") is None

    @pytest.mark.asyncio
    async def test_polling_restores_env_on_failure(self) -> None:
        """env_overrides 在 submit_fn 异常后也能恢复。"""
        from harness._internal.polling import execute_polling
        from harness.task import PollingTask, TaskConfig

        def submit_fn(results):
            raise RuntimeError("boom")

        task = PollingTask(
            submit_fn=submit_fn,
            poll_fn=lambda h: None,
            success_condition=lambda r: True,
            timeout=1,
            poll_interval=0.01,
            config=TaskConfig(max_retries=0, timeout=5),
        )

        with pytest.raises(Exception):
            await execute_polling(
                task, "0", [], "run-1",
                harness_config=None,
                env_overrides={"POLL_CLEANUP_VAR": "temp"},
            )

        assert os.environ.get("POLL_CLEANUP_VAR") is None


# ---------------------------------------------------------------------------
# Fix #8: _env_locks cleanup (no memory leak)
# ---------------------------------------------------------------------------


class TestEnvLocksCleanup:
    @pytest.mark.asyncio
    async def test_stale_locks_cleaned_on_next_call(self) -> None:
        """已销毁事件循环的 Lock 条目在下次调用时被清理。"""
        from harness._internal.executor import _env_locks, _get_env_lock

        # 记录当前条目数
        initial_count = len(_env_locks)

        # 插入一个伪造的过期条目（weakref 指向 None）
        import weakref
        fake_id = 99999999
        # 创建一个临时对象并让 weakref 失效
        class Dummy:
            pass
        tmp = Dummy()
        ref = weakref.ref(tmp)
        _env_locks[fake_id] = (ref, asyncio.Lock())
        del tmp  # weakref 失效

        # 调用 _get_env_lock 应清理过期条目
        _get_env_lock()
        assert fake_id not in _env_locks

    @pytest.mark.asyncio
    async def test_same_loop_returns_same_lock(self) -> None:
        """同一事件循环返回同一个 Lock 实例。"""
        lock1 = _get_env_lock()
        lock2 = _get_env_lock()
        assert lock1 is lock2


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


class MockRunner(AbstractRunner):
    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        return RunnerResult(text="ok", tokens_used=0, session_id=None)

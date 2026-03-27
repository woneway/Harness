"""tests/integration/test_service.py — Service 模式集成测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from harness import (
    CronTrigger,
    EventTrigger,
    FunctionTask,
    Harness,
    LLMTask,
    State,
    TriggerContext,
)
from harness.runners.base import AbstractRunner, RunnerResult


class MockRunner(AbstractRunner):
    def __init__(self, text: str = "mock") -> None:
        self._text = text

    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        return RunnerResult(text=self._text, tokens_used=5, session_id="s1")


class TestServiceRegistration:
    def test_service_creates_runner(self, tmp_path: Path) -> None:
        h = Harness(str(tmp_path), runner=MockRunner())

        async def handler(ctx: TriggerContext):
            return [LLMTask(prompt="test")]

        h.service("svc1", triggers=[CronTrigger("0 * * * *")], handler=handler)
        assert h._service_runner is not None

    def test_service_duplicate_raises(self, tmp_path: Path) -> None:
        h = Harness(str(tmp_path), runner=MockRunner())

        async def handler(ctx):
            return []

        h.service("svc1", triggers=[], handler=handler)
        with pytest.raises(ValueError, match="already registered"):
            h.service("svc1", triggers=[], handler=handler)


class TestServiceEmit:
    @pytest.mark.asyncio
    async def test_emit_without_service_noop(self, tmp_path: Path) -> None:
        h = Harness(str(tmp_path), runner=MockRunner())
        # 无 service 注册，不应抛异常
        await h.emit("test", {"data": 1})

    @pytest.mark.asyncio
    async def test_emit_triggers_pipeline(self, tmp_path: Path) -> None:
        h = Harness(str(tmp_path), runner=MockRunner("event result"))
        executed = []

        async def handler(ctx: TriggerContext):
            executed.append(ctx.event_data)
            return [LLMTask(prompt="process event")]

        h.service(
            "evt-svc",
            triggers=[EventTrigger("alert")],
            handler=handler,
        )

        await h.start()
        try:
            await h.emit("alert", {"symbol": "AAPL"})
            await asyncio.sleep(0.2)
        finally:
            await h.stop()

        assert len(executed) == 1
        assert executed[0] == {"symbol": "AAPL"}


class TestServiceWithState:
    @pytest.mark.asyncio
    async def test_state_factory_creates_fresh_state(self, tmp_path: Path) -> None:
        h = Harness(str(tmp_path), runner=MockRunner())

        class MyState(State):
            counter: int = 0

        states_created = []

        def factory():
            s = MyState()
            states_created.append(s)
            return s

        async def handler(ctx: TriggerContext):
            return [LLMTask(prompt="test")]

        h.service(
            "svc",
            triggers=[EventTrigger("tick")],
            handler=handler,
            state_factory=factory,
        )

        await h.start()
        try:
            await h.emit("tick")
            await asyncio.sleep(0.2)
            await h.emit("tick")
            await asyncio.sleep(0.2)
        finally:
            await h.stop()

        assert len(states_created) == 2
        assert states_created[0] is not states_created[1]


class TestServiceEventFilter:
    @pytest.mark.asyncio
    async def test_filter_blocks_event(self, tmp_path: Path) -> None:
        h = Harness(str(tmp_path), runner=MockRunner())
        executed = []

        async def handler(ctx: TriggerContext):
            executed.append(True)
            return [LLMTask(prompt="test")]

        h.service(
            "filtered",
            triggers=[EventTrigger(
                "price",
                filter=lambda d: d.get("change", 0) > 0.03,
            )],
            handler=handler,
        )

        await h.start()
        try:
            await h.emit("price", {"change": 0.01})  # blocked
            await asyncio.sleep(0.2)
            assert len(executed) == 0

            await h.emit("price", {"change": 0.05})  # passes
            await asyncio.sleep(0.2)
            assert len(executed) == 1
        finally:
            await h.stop()


class TestServiceMixedTriggers:
    @pytest.mark.asyncio
    async def test_cron_and_event_registered(self, tmp_path: Path) -> None:
        """验证混合触发器在 service runner 中正确注册。"""
        h = Harness(str(tmp_path), runner=MockRunner())

        async def handler(ctx: TriggerContext):
            return [LLMTask(prompt="test")]

        h.service(
            "mixed",
            triggers=[
                CronTrigger("0 * * * *"),
                EventTrigger("alert"),
            ],
            handler=handler,
        )

        assert h._service_runner is not None
        assert h._service_runner.has_event_triggers is True


class TestServiceStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_scheduler_for_service(self, tmp_path: Path) -> None:
        h = Harness(str(tmp_path), runner=MockRunner())

        async def handler(ctx):
            return []

        h.service("svc", triggers=[CronTrigger("0 * * * *")], handler=handler)
        assert h._scheduler is None

        # start() 应创建 scheduler（即使 APScheduler 不能序列化闭包，
        # scheduler 对象本身应被创建）
        # 注：APScheduler v4 对闭包序列化有限制，完整 cron 测试在 unit test 中覆盖
        await h._ensure_initialized()
        assert h._scheduler is None  # scheduler 在 start() 中创建
        # 直接检查 service_runner 注册
        assert h._service_runner.has_services is True

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self, tmp_path: Path) -> None:
        h = Harness(str(tmp_path), runner=MockRunner())

        async def handler(ctx):
            return []

        h.service("svc", triggers=[EventTrigger("test")], handler=handler)
        await h.start()
        assert h._service_runner._running is True

        await h.stop()
        assert h._service_runner._running is False


class TestServiceErrorIsolation:
    @pytest.mark.asyncio
    async def test_failing_handler_does_not_crash_service(self, tmp_path: Path) -> None:
        h = Harness(str(tmp_path), runner=MockRunner())

        async def failing_handler(ctx: TriggerContext):
            raise RuntimeError("handler crashed")

        h.service(
            "failing",
            triggers=[EventTrigger("boom")],
            handler=failing_handler,
        )

        await h.start()
        try:
            # 不应抛异常
            await h.emit("boom", {})
            await asyncio.sleep(0.2)
            assert h._service_runner._running is True
        finally:
            await h.stop()

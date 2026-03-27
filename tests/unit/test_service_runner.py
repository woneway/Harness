"""tests/unit/test_service_runner.py — ServiceRunner 测试。"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness._internal.service import ServiceDef, ServiceRunner
from harness.runners.base import AbstractRunner, RunnerResult
from harness.scheduler.base import AbstractScheduler
from harness.state import State
from harness.tasks.llm import LLMTask
from harness.triggers import CronTrigger, EventTrigger, TriggerContext


class MockScheduler(AbstractScheduler):
    """记录注册的 jobs。"""

    def __init__(self) -> None:
        self.jobs: list[tuple[Any, str]] = []

    def add_job(self, fn, cron) -> None:
        self.jobs.append((fn, cron))

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class MockHarness:
    """最小 Harness mock，只提供 pipeline()。"""

    def __init__(self) -> None:
        self.pipeline_calls: list[dict] = []

    async def pipeline(self, tasks, *, name=None, state=None, resume_from=None):
        self.pipeline_calls.append({
            "tasks": tasks,
            "name": name,
            "state": state,
        })
        return MagicMock()


def _make_handler(steps=None):
    """创建返回固定 steps 的 async handler。"""
    _steps = steps or [LLMTask(prompt="test")]

    async def handler(ctx: TriggerContext):
        return _steps

    return handler


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


class TestServiceRunnerRegister:
    def test_register_service(self) -> None:
        sr = ServiceRunner(MockHarness())  # type: ignore
        svc = ServiceDef(
            name="svc1",
            triggers=[CronTrigger(cron="0 * * * *")],
            handler=_make_handler(),
            state_factory=None,
            pipeline_name=None,
        )
        sr.register(svc)
        assert "svc1" in sr._services

    def test_register_duplicate_raises(self) -> None:
        sr = ServiceRunner(MockHarness())  # type: ignore
        svc = ServiceDef(
            name="svc1",
            triggers=[],
            handler=_make_handler(),
            state_factory=None,
            pipeline_name=None,
        )
        sr.register(svc)
        with pytest.raises(ValueError, match="already registered"):
            sr.register(svc)

    def test_has_services(self) -> None:
        sr = ServiceRunner(MockHarness())  # type: ignore
        assert sr.has_services is False
        sr.register(ServiceDef(
            name="s", triggers=[], handler=_make_handler(),
            state_factory=None, pipeline_name=None,
        ))
        assert sr.has_services is True

    def test_has_event_triggers(self) -> None:
        sr = ServiceRunner(MockHarness())  # type: ignore
        sr.register(ServiceDef(
            name="s1",
            triggers=[CronTrigger(cron="0 * * * *")],
            handler=_make_handler(),
            state_factory=None,
            pipeline_name=None,
        ))
        assert sr.has_event_triggers is False

        sr.register(ServiceDef(
            name="s2",
            triggers=[EventTrigger(event="test")],
            handler=_make_handler(),
            state_factory=None,
            pipeline_name=None,
        ))
        assert sr.has_event_triggers is True


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


class TestServiceRunnerStart:
    @pytest.mark.asyncio
    async def test_cron_trigger_registered(self) -> None:
        sr = ServiceRunner(MockHarness())  # type: ignore
        scheduler = MockScheduler()

        sr.register(ServiceDef(
            name="cron-svc",
            triggers=[CronTrigger(cron="15 15 * * 1-5")],
            handler=_make_handler(),
            state_factory=None,
            pipeline_name=None,
        ))
        await sr.start(scheduler)

        assert len(scheduler.jobs) == 1
        assert scheduler.jobs[0][1] == "15 15 * * 1-5"
        assert sr._running is True

    @pytest.mark.asyncio
    async def test_event_trigger_creates_bus(self) -> None:
        sr = ServiceRunner(MockHarness())  # type: ignore
        scheduler = MockScheduler()

        sr.register(ServiceDef(
            name="evt-svc",
            triggers=[EventTrigger(event="price_alert")],
            handler=_make_handler(),
            state_factory=None,
            pipeline_name=None,
        ))
        await sr.start(scheduler)

        assert sr._event_bus is not None

    @pytest.mark.asyncio
    async def test_no_event_trigger_no_bus(self) -> None:
        sr = ServiceRunner(MockHarness())  # type: ignore
        scheduler = MockScheduler()

        sr.register(ServiceDef(
            name="cron-only",
            triggers=[CronTrigger(cron="0 * * * *")],
            handler=_make_handler(),
            state_factory=None,
            pipeline_name=None,
        ))
        await sr.start(scheduler)

        assert sr._event_bus is None

    @pytest.mark.asyncio
    async def test_stop_cleans_bus(self) -> None:
        sr = ServiceRunner(MockHarness())  # type: ignore
        scheduler = MockScheduler()

        sr.register(ServiceDef(
            name="evt-svc",
            triggers=[EventTrigger(event="test")],
            handler=_make_handler(),
            state_factory=None,
            pipeline_name=None,
        ))
        await sr.start(scheduler)
        assert sr._event_bus is not None

        await sr.stop()
        assert sr._event_bus is None
        assert sr._running is False


# ---------------------------------------------------------------------------
# emit / event execution
# ---------------------------------------------------------------------------


class TestServiceRunnerEmit:
    @pytest.mark.asyncio
    async def test_emit_no_bus_silent(self) -> None:
        sr = ServiceRunner(MockHarness())  # type: ignore
        # 无 EventTrigger，emit 不应抛异常
        await sr.emit("test", {"data": 1})

    @pytest.mark.asyncio
    async def test_emit_triggers_handler(self) -> None:
        harness = MockHarness()
        sr = ServiceRunner(harness)  # type: ignore
        scheduler = MockScheduler()

        sr.register(ServiceDef(
            name="evt-svc",
            triggers=[EventTrigger(event="alert")],
            handler=_make_handler([LLMTask(prompt="triggered")]),
            state_factory=None,
            pipeline_name=None,
        ))
        await sr.start(scheduler)
        await sr.emit("alert", {"key": "val"})

        # pyee 异步执行
        await asyncio.sleep(0.1)

        assert len(harness.pipeline_calls) == 1
        call = harness.pipeline_calls[0]
        assert len(call["tasks"]) == 1
        assert call["tasks"][0].prompt == "triggered"

    @pytest.mark.asyncio
    async def test_event_filter_blocks(self) -> None:
        harness = MockHarness()
        sr = ServiceRunner(harness)  # type: ignore
        scheduler = MockScheduler()

        sr.register(ServiceDef(
            name="filtered",
            triggers=[EventTrigger(
                event="price",
                filter=lambda d: d.get("change", 0) > 0.03,
            )],
            handler=_make_handler(),
            state_factory=None,
            pipeline_name=None,
        ))
        await sr.start(scheduler)

        # 不满足 filter
        await sr.emit("price", {"change": 0.01})
        await asyncio.sleep(0.1)
        assert len(harness.pipeline_calls) == 0

    @pytest.mark.asyncio
    async def test_event_filter_passes(self) -> None:
        harness = MockHarness()
        sr = ServiceRunner(harness)  # type: ignore
        scheduler = MockScheduler()

        sr.register(ServiceDef(
            name="filtered",
            triggers=[EventTrigger(
                event="price",
                filter=lambda d: d.get("change", 0) > 0.03,
            )],
            handler=_make_handler(),
            state_factory=None,
            pipeline_name=None,
        ))
        await sr.start(scheduler)

        # 满足 filter
        await sr.emit("price", {"change": 0.05})
        await asyncio.sleep(0.1)
        assert len(harness.pipeline_calls) == 1


# ---------------------------------------------------------------------------
# cron execution
# ---------------------------------------------------------------------------


class TestServiceRunnerCron:
    @pytest.mark.asyncio
    async def test_cron_handler_calls_pipeline(self) -> None:
        harness = MockHarness()
        sr = ServiceRunner(harness)  # type: ignore
        scheduler = MockScheduler()

        sr.register(ServiceDef(
            name="cron-svc",
            triggers=[CronTrigger(cron="0 * * * *")],
            handler=_make_handler([LLMTask(prompt="cron job")]),
            state_factory=None,
            pipeline_name=None,
        ))
        await sr.start(scheduler)

        # 手动调用 cron handler
        cron_fn = scheduler.jobs[0][0]
        await cron_fn()

        assert len(harness.pipeline_calls) == 1
        assert harness.pipeline_calls[0]["tasks"][0].prompt == "cron job"


# ---------------------------------------------------------------------------
# state_factory / pipeline_name
# ---------------------------------------------------------------------------


class TestServiceRunnerStateFactory:
    @pytest.mark.asyncio
    async def test_state_factory_called_per_trigger(self) -> None:
        harness = MockHarness()
        sr = ServiceRunner(harness)  # type: ignore
        scheduler = MockScheduler()

        call_count = 0

        class MyState(State):
            pass

        def factory():
            nonlocal call_count
            call_count += 1
            return MyState()

        sr.register(ServiceDef(
            name="svc",
            triggers=[CronTrigger(cron="0 * * * *")],
            handler=_make_handler(),
            state_factory=factory,
            pipeline_name=None,
        ))
        await sr.start(scheduler)

        cron_fn = scheduler.jobs[0][0]
        await cron_fn()
        await cron_fn()

        assert call_count == 2
        # 每次是不同的 State 实例
        s1 = harness.pipeline_calls[0]["state"]
        s2 = harness.pipeline_calls[1]["state"]
        assert s1 is not s2

    @pytest.mark.asyncio
    async def test_default_state_when_no_factory(self) -> None:
        harness = MockHarness()
        sr = ServiceRunner(harness)  # type: ignore
        scheduler = MockScheduler()

        sr.register(ServiceDef(
            name="svc",
            triggers=[CronTrigger(cron="0 * * * *")],
            handler=_make_handler(),
            state_factory=None,
            pipeline_name=None,
        ))
        await sr.start(scheduler)

        await scheduler.jobs[0][0]()
        assert isinstance(harness.pipeline_calls[0]["state"], State)

    @pytest.mark.asyncio
    async def test_custom_pipeline_name(self) -> None:
        harness = MockHarness()
        sr = ServiceRunner(harness)  # type: ignore
        scheduler = MockScheduler()

        sr.register(ServiceDef(
            name="svc",
            triggers=[CronTrigger(cron="0 * * * *")],
            handler=_make_handler(),
            state_factory=None,
            pipeline_name="custom-name",
        ))
        await sr.start(scheduler)

        await scheduler.jobs[0][0]()
        assert harness.pipeline_calls[0]["name"] == "custom-name"

    @pytest.mark.asyncio
    async def test_auto_pipeline_name(self) -> None:
        harness = MockHarness()
        sr = ServiceRunner(harness)  # type: ignore
        scheduler = MockScheduler()

        sr.register(ServiceDef(
            name="my-svc",
            triggers=[CronTrigger(cron="0 * * * *")],
            handler=_make_handler(),
            state_factory=None,
            pipeline_name=None,
        ))
        await sr.start(scheduler)

        await scheduler.jobs[0][0]()
        name = harness.pipeline_calls[0]["name"]
        assert name.startswith("my-svc-")


# ---------------------------------------------------------------------------
# error isolation
# ---------------------------------------------------------------------------


class TestServiceRunnerErrorIsolation:
    @pytest.mark.asyncio
    async def test_handler_exception_does_not_crash(self) -> None:
        harness = MockHarness()
        sr = ServiceRunner(harness)  # type: ignore
        scheduler = MockScheduler()

        async def failing_handler(ctx):
            raise RuntimeError("boom")

        sr.register(ServiceDef(
            name="failing",
            triggers=[CronTrigger(cron="0 * * * *")],
            handler=failing_handler,
            state_factory=None,
            pipeline_name=None,
        ))
        await sr.start(scheduler)

        # 不应抛异常
        await scheduler.jobs[0][0]()
        assert sr._running is True

    @pytest.mark.asyncio
    async def test_pipeline_exception_does_not_crash(self) -> None:
        class FailingHarness:
            async def pipeline(self, tasks, **kwargs):
                raise RuntimeError("pipeline boom")

        sr = ServiceRunner(FailingHarness())  # type: ignore
        scheduler = MockScheduler()

        sr.register(ServiceDef(
            name="svc",
            triggers=[CronTrigger(cron="0 * * * *")],
            handler=_make_handler(),
            state_factory=None,
            pipeline_name=None,
        ))
        await sr.start(scheduler)

        # 不应抛异常
        await scheduler.jobs[0][0]()
        assert sr._running is True


# ---------------------------------------------------------------------------
# single step wrapping
# ---------------------------------------------------------------------------


class TestServiceRunnerSingleStep:
    @pytest.mark.asyncio
    async def test_handler_returns_single_step(self) -> None:
        harness = MockHarness()
        sr = ServiceRunner(harness)  # type: ignore
        scheduler = MockScheduler()

        async def single_handler(ctx):
            return LLMTask(prompt="single")  # 返回单个 step 而非 list

        sr.register(ServiceDef(
            name="single",
            triggers=[CronTrigger(cron="0 * * * *")],
            handler=single_handler,
            state_factory=None,
            pipeline_name=None,
        ))
        await sr.start(scheduler)

        await scheduler.jobs[0][0]()
        tasks = harness.pipeline_calls[0]["tasks"]
        assert isinstance(tasks, list)
        assert len(tasks) == 1

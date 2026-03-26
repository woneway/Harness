"""service.py — ServiceRunner，管理长驻服务的触发与执行。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from harness._internal.event_bus import EventBus
from harness.state import State
from harness.tasks.base import BaseTask
from harness.triggers import CronTrigger, EventTrigger, Trigger, TriggerContext

if TYPE_CHECKING:
    from harness.harness import Harness
    from harness.scheduler.base import AbstractScheduler
    from harness.tasks.types import PipelineStep

logger = logging.getLogger(__name__)

ServiceHandler = Callable[
    [TriggerContext],
    Awaitable["list[PipelineStep] | PipelineStep"],
]


@dataclass
class ServiceDef:
    """服务定义（内部数据结构）。"""

    name: str
    triggers: list[Trigger]
    handler: ServiceHandler
    state_factory: Callable[[], State] | None
    pipeline_name: str | None


class ServiceRunner:
    """管理所有已注册 service 的触发与执行。

    职责：
    1. 为 CronTrigger 注册 APScheduler 任务
    2. 为 EventTrigger 注册 EventBus 监听
    3. 触发时调用 handler → h.pipeline()
    4. 错误隔离：单次触发失败不影响服务继续运行
    """

    def __init__(self, harness: Harness) -> None:
        self._harness = harness
        self._services: dict[str, ServiceDef] = {}
        self._event_bus: EventBus | None = None
        self._running = False

    @property
    def has_services(self) -> bool:
        return len(self._services) > 0

    @property
    def has_event_triggers(self) -> bool:
        return any(
            isinstance(t, EventTrigger)
            for svc in self._services.values()
            for t in svc.triggers
        )

    def register(self, svc: ServiceDef) -> None:
        """注册服务定义。

        Raises:
            ValueError: 重复注册同名服务。
        """
        if svc.name in self._services:
            raise ValueError(f"Service '{svc.name}' already registered")
        self._services[svc.name] = svc

    async def start(self, scheduler: "AbstractScheduler") -> None:
        """将所有 trigger 绑定到 scheduler / event_bus。"""
        has_events = False

        for svc in self._services.values():
            for trigger in svc.triggers:
                if isinstance(trigger, CronTrigger):
                    scheduler.add_job(
                        self._make_cron_handler(svc, trigger),
                        trigger.cron,
                    )
                elif isinstance(trigger, EventTrigger):
                    has_events = True

        # 懒初始化 EventBus（仅在有 EventTrigger 时）
        if has_events:
            self._event_bus = EventBus()
            for svc in self._services.values():
                for trigger in svc.triggers:
                    if isinstance(trigger, EventTrigger):
                        self._event_bus.on(
                            trigger.event,
                            self._make_event_handler(svc, trigger),
                        )

        self._running = True

    async def stop(self) -> None:
        """清理 EventBus 监听。"""
        if self._event_bus is not None:
            self._event_bus.remove_all_listeners()
            self._event_bus = None
        self._running = False

    async def emit(self, event: str, data: Any = None) -> None:
        """转发事件到 EventBus。无 EventBus 时静默忽略。"""
        if self._event_bus is not None:
            self._event_bus.emit(event, data)

    def _make_cron_handler(
        self, svc: ServiceDef, trigger: CronTrigger
    ) -> Callable[[], Awaitable[None]]:
        """创建 cron 触发的 async handler。"""

        async def _handler() -> None:
            ctx = TriggerContext(
                service_name=svc.name,
                trigger_type="cron",
                event_name=None,
                event_data=None,
                triggered_at=datetime.now(),
            )
            await self._execute(svc, ctx)

        return _handler

    def _make_event_handler(
        self, svc: ServiceDef, trigger: EventTrigger
    ) -> Callable[..., Awaitable[None]]:
        """创建 event 触发的 handler。"""

        async def _handler(data: Any = None) -> None:
            # 可选过滤
            if trigger.filter is not None and not trigger.filter(data):
                return
            ctx = TriggerContext(
                service_name=svc.name,
                trigger_type="event",
                event_name=trigger.event,
                event_data=data,
                triggered_at=datetime.now(),
            )
            await self._execute(svc, ctx)

        return _handler

    async def _execute(self, svc: ServiceDef, ctx: TriggerContext) -> None:
        """触发后执行 handler → pipeline。错误隔离，不中断服务。"""
        try:
            steps = await svc.handler(ctx)
            if isinstance(steps, BaseTask):
                steps = [steps]

            state = svc.state_factory() if svc.state_factory else State()

            name = (
                svc.pipeline_name
                or f"{svc.name}-{ctx.triggered_at:%Y%m%d-%H%M%S}"
            )

            await self._harness.pipeline(steps, name=name, state=state)

        except Exception:
            logger.exception(
                "Service '%s' trigger execution failed", svc.name
            )

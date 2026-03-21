"""APScheduler v4 调度后端。

注意：APScheduler v4 alpha 版本的 add_schedule() 是协程，
必须在 start() 中通过 async context manager 注册。
因此 add_job() 先缓存，start() 时再批量注册。
"""

from __future__ import annotations

from typing import Callable

from apscheduler import AsyncScheduler
from apscheduler.triggers.cron import CronTrigger

from harness.scheduler.base import AbstractScheduler


class APSchedulerBackend(AbstractScheduler):
    """使用 APScheduler v4 实现的调度后端。"""

    def __init__(self) -> None:
        self._pending_jobs: list[tuple[Callable, str]] = []
        self._scheduler: AsyncScheduler | None = None

    def add_job(self, fn: Callable, cron: str) -> None:
        """缓存定时任务，待 start() 时批量注册。

        Args:
            fn: 要调度的异步可调用对象。
            cron: Cron 表达式（5 字段，与 Unix cron 兼容）。
        """
        self._pending_jobs.append((fn, cron))

    async def start(self) -> None:
        """启动调度器并注册所有缓存的任务。"""
        self._scheduler = AsyncScheduler()
        await self._scheduler.__aenter__()
        for fn, cron in self._pending_jobs:
            await self._scheduler.add_schedule(fn, CronTrigger.from_crontab(cron))

    async def stop(self) -> None:
        """停止调度器。"""
        if self._scheduler is not None:
            await self._scheduler.__aexit__(None, None, None)
            self._scheduler = None

"""AbstractScheduler — 调度器抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class AbstractScheduler(ABC):
    """调度器抽象接口。"""

    @abstractmethod
    def add_job(self, fn: Callable, cron: str) -> None:
        """添加定时任务。

        Args:
            fn: 要执行的异步函数。
            cron: Cron 表达式，如 "0 2 * * *"。
        """

    @abstractmethod
    async def start(self) -> None:
        """启动调度器。"""

    @abstractmethod
    async def stop(self) -> None:
        """停止调度器。"""

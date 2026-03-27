"""event_bus.py — 事件总线，封装 pyee AsyncIOEventEmitter。"""

from __future__ import annotations

from typing import Any, Callable


class EventBus:
    """事件总线。

    pyee 是可选依赖，仅在创建 EventBus 时导入。
    未安装时给出清晰的安装提示。
    """

    def __init__(self) -> None:
        try:
            from pyee.asyncio import AsyncIOEventEmitter
        except ImportError:
            raise ImportError(
                "pyee is required for EventTrigger. "
                "Install it with: pip install 'harness-ai[service]'"
            ) from None
        self._emitter = AsyncIOEventEmitter()

    def on(self, event: str, handler: Callable) -> None:
        """注册事件监听器。"""
        self._emitter.on(event, handler)

    def emit(self, event: str, data: Any = None) -> None:
        """发射事件。"""
        self._emitter.emit(event, data)

    def remove_all_listeners(self) -> None:
        """移除所有监听器。"""
        self._emitter.remove_all_listeners()

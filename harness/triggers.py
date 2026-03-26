"""triggers.py — Service 模式的触发器定义。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


@dataclass
class CronTrigger:
    """定时触发器，复用 APScheduler。

    Args:
        cron: Cron 表达式（5 字段），如 ``"15 15 * * 1-5"``。
        name: 可选名称，用于日志标识。
    """

    cron: str
    name: str | None = None


@dataclass
class EventTrigger:
    """事件触发器，基于 pyee。

    Args:
        event: 事件名（如 ``"price_alert"``）。
        filter: 可选过滤函数，接收事件数据，返回 True 才触发。
        name: 可选名称，用于日志标识。
    """

    event: str
    filter: Callable[[Any], bool] | None = None
    name: str | None = None


Trigger = CronTrigger | EventTrigger


@dataclass
class TriggerContext:
    """handler 接收的触发上下文。

    Attributes:
        service_name: 所属 service 名称。
        trigger_type: ``"cron"`` 或 ``"event"``。
        event_name: EventTrigger 的事件名，cron 时为 None。
        event_data: 事件数据，cron 时为 None。
        triggered_at: 触发时间。
    """

    service_name: str
    trigger_type: str  # "cron" | "event"
    event_name: str | None
    event_data: Any
    triggered_at: datetime

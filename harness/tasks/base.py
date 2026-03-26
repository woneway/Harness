"""BaseTask / DialogueProgressEvent — 任务基类与进度事件。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from harness.tasks.config import TaskConfig


@dataclass
class DialogueProgressEvent:
    """progress_callback 接收的结构化事件。

    Attributes:
        event:         "start" | "complete" | "error"
        round_or_turn: 当前轮次（从 0 开始）
        role_name:     当前发言角色名
        content:       发言内容（event="complete"）或错误信息（event="error"），
                       event="start" 时为 None
    """

    event: Literal["start", "complete", "error", "streaming"]
    round_or_turn: int
    role_name: str
    content: str | None = None


@dataclass
class BaseTask:
    """所有 Task 类型的公共基类，用户不直接实例化。

    stream_callback 和 raw_stream_callback 互斥，同时设置时抛 ValueError。
    """

    config: TaskConfig | None = None
    stream_callback: Callable[[str], None] | None = None
    raw_stream_callback: Callable[[dict], None] | None = None

    def __post_init__(self) -> None:
        if self.stream_callback is not None and self.raw_stream_callback is not None:
            raise ValueError(
                "stream_callback and raw_stream_callback are mutually exclusive. "
                "Set only one of them."
            )

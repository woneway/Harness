"""PollingTask — 提交后轮询任务。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from harness.tasks.base import BaseTask
from harness.tasks.result import Result


@dataclass
class PollingTask(BaseTask):
    """提交异步任务后轮询结果，适用于视频生成、TTS、图像生成等外部 AI API。"""

    submit_fn: Callable[[list[Result]], Any] = field(default=lambda results: None)
    poll_fn: Callable[[Any], Any] = field(default=lambda handle: None)
    success_condition: Callable[[Any], bool] = field(default=lambda r: False)
    failure_condition: Callable[[Any], bool] | None = None
    poll_interval: float = 10.0
    timeout: int = 900
    output_schema: type[BaseModel] | None = None

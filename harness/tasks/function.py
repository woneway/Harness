"""FunctionTask — 纯 Python 函数任务。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from harness.tasks.base import BaseTask
from harness.tasks.result import Result


@dataclass
class FunctionTask(BaseTask):
    """执行纯 Python 函数，框架直接调用，不经过任何 runner。

    output_schema 校验失败时抛 OutputSchemaError，不触发重试。
    """

    fn: Callable[[list[Result]], Any] = field(default=lambda results: None)
    output_schema: type[BaseModel] | None = None
    output_key: str | None = None  # v2: 执行后将 result.output 写入 state 属性

"""LLMTask — 语言模型调用任务。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from harness.tasks.base import BaseTask
from harness.tasks.result import Result


@dataclass
class LLMTask(BaseTask):
    """调用语言模型，默认用 ClaudeCliRunner。"""

    prompt: str | Callable[[list[Result]], str] = ""
    system_prompt: str = ""
    output_schema: type[BaseModel] | None = None
    runner: Any | None = None  # AbstractRunner | None，避免循环导入

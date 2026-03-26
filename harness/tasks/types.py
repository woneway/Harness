"""PipelineStep union 类型别名 + Task 废弃别名。"""

from __future__ import annotations

import warnings
from typing import Any

from harness.tasks.condition import Condition
from harness.tasks.dialogue import Dialogue
from harness.tasks.discussion import Discussion
from harness.tasks.function import FunctionTask
from harness.tasks.llm import LLMTask
from harness.tasks.loop import Loop
from harness.tasks.parallel import Parallel
from harness.tasks.polling import PollingTask
from harness.tasks.shell import ShellTask

PipelineStep = (
    LLMTask | FunctionTask | ShellTask | PollingTask | Parallel | Dialogue
    | Discussion | Condition | Loop
)


def Task(*args: Any, **kwargs: Any) -> LLMTask:
    """LLMTask 的已废弃别名，v2 将移除。"""
    warnings.warn(
        "Task is deprecated, use LLMTask instead. Task will be removed in v2.",
        DeprecationWarning,
        stacklevel=2,
    )
    return LLMTask(*args, **kwargs)

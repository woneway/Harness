"""Condition — 条件分支任务。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from harness.tasks.base import BaseTask

if TYPE_CHECKING:
    from harness.state import State
    from harness.tasks.types import PipelineStep


@dataclass
class Condition(BaseTask):
    """条件分支：根据 check(state) 结果执行 if_true 或 if_false 分支。

    Example::

        Condition(
            check=lambda state: state.score > 0.8,
            if_true=[LLMTask("生成报告", output_key="report")],
            if_false=[LLMTask("重新分析", output_key="analysis")],
        )
    """

    check: Callable[["State"], bool] = field(default=lambda state: False)
    if_true: list[Any] = field(default_factory=list)   # list[PipelineStep]
    if_false: list[Any] = field(default_factory=list)  # list[PipelineStep]

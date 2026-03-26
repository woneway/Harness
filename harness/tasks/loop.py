"""Loop — 循环任务。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from harness.tasks.base import BaseTask

if TYPE_CHECKING:
    from harness.state import State
    from harness.tasks.types import PipelineStep


@dataclass
class Loop(BaseTask):
    """循环执行 body 步骤直到 until(state) 返回 True 或达到 max_iterations。

    Example::

        Loop(
            body=[
                LLMTask("写文章", output_key="draft"),
                LLMTask("评审", output_schema=Review, output_key="review"),
            ],
            until=lambda state: state.review.passed,
            max_iterations=5,
        )
    """

    body: list[Any] = field(default_factory=list)  # list[PipelineStep]
    until: Callable[["State"], bool] = field(default=lambda state: True)
    max_iterations: int = 5

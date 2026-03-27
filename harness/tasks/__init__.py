"""harness.tasks — Task 类型体系（从 harness.task 拆分）。

统一导出所有公开符号，保持与 harness.task 完全一致的 API。
"""

from harness.tasks.base import BaseTask, DialogueProgressEvent
from harness.tasks.condition import Condition
from harness.tasks.config import TaskConfig
from harness.tasks.dialogue import Dialogue, DialogueOutput, DialogueTurn, Role
from harness.tasks.discussion import (
    Discussion,
    DiscussionOutput,
    DiscussionProgressEvent,
    DiscussionTurn,
)
from harness.tasks.function import FunctionTask
from harness.tasks.llm import LLMTask
from harness.tasks.loop import Loop
from harness.tasks.parallel import Parallel
from harness.tasks.polling import PollingTask
from harness.tasks.result import PipelineResult, Result, result_by_type
from harness.tasks.shell import ShellTask
from harness.tasks.types import PipelineStep, Task

__all__ = [
    # config
    "TaskConfig",
    # result
    "Result",
    "PipelineResult",
    "result_by_type",
    # base
    "BaseTask",
    "DialogueProgressEvent",
    # task types
    "LLMTask",
    "FunctionTask",
    "ShellTask",
    "PollingTask",
    "Parallel",
    # flow control
    "Condition",
    "Loop",
    # dialogue
    "Dialogue",
    "DialogueOutput",
    "DialogueTurn",
    "Role",
    # discussion
    "Discussion",
    "DiscussionOutput",
    "DiscussionTurn",
    "DiscussionProgressEvent",
    # types
    "PipelineStep",
    "Task",
]

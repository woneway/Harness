"""Harness — AI-native 通用自动化流水线框架。

公开 API：

    from harness import (
        Harness,
        LLMTask, FunctionTask, ShellTask, PollingTask, Parallel,
        Task,           # LLMTask 的已废弃别名
        Result, PipelineResult,
        TaskConfig,
        Memory,
        # Runner 基类（自定义 runner 时继承）
        AbstractRunner, RunnerResult,
        # 内置 API runners
        OpenAIRunner,
        AnthropicRunner,
    )

    # PermissionMode 需从 harness.runners.claude_cli 单独导入
    from harness.runners.claude_cli import PermissionMode
"""

from harness.agent import Agent
from harness.harness import Harness
from harness.memory import Memory
from harness.runners.anthropic import AnthropicRunner
from harness.state import State
from harness.runners.base import AbstractRunner, RunnerResult
from harness.runners.openai import OpenAIRunner
from harness.triggers import CronTrigger, EventTrigger, TriggerContext
from harness.tasks import (
    Condition,
    Dialogue,
    DialogueProgressEvent,
    FunctionTask,
    LLMTask,
    Loop,
    Parallel,
    PipelineResult,
    PollingTask,
    Result,
    Role,
    ShellTask,
    Task,
    TaskConfig,
    result_by_type,
)

__all__ = [
    "Harness",
    "Agent",
    "State",
    "CronTrigger",
    "EventTrigger",
    "TriggerContext",
    "Condition",
    "Loop",
    "LLMTask",
    "FunctionTask",
    "ShellTask",
    "PollingTask",
    "Parallel",
    "Dialogue",
    "DialogueProgressEvent",
    "Role",
    "Task",
    "Result",
    "PipelineResult",
    "TaskConfig",
    "Memory",
    # Runner
    "AbstractRunner",
    "RunnerResult",
    "OpenAIRunner",
    "AnthropicRunner",
    # 辅助函数
    "result_by_type",
]

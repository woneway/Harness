"""Harness — AI-native 通用自动化流水线框架。

公开 API：

    from harness import (
        Harness,
        LLMTask, FunctionTask, ShellTask, PollingTask, Parallel,
        Task,           # LLMTask 的已废弃别名
        Result, PipelineResult,
        TaskConfig,
        Memory,
    )

    # PermissionMode 需从 harness.runners.claude_cli 单独导入
    from harness.runners.claude_cli import PermissionMode
"""

from harness.harness import Harness
from harness.memory import Memory
from harness.task import (
    Dialogue,
    FunctionTask,
    LLMTask,
    Parallel,
    PipelineResult,
    PollingTask,
    Result,
    Role,
    ShellTask,
    Task,
    TaskConfig,
)

__all__ = [
    "Harness",
    "LLMTask",
    "FunctionTask",
    "ShellTask",
    "PollingTask",
    "Parallel",
    "Dialogue",
    "Role",
    "Task",
    "Result",
    "PipelineResult",
    "TaskConfig",
    "Memory",
]

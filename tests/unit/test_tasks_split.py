"""tests/unit/test_tasks_split.py — 验证 tasks/ 拆分后两条 import 路径都可用。"""

from __future__ import annotations


class TestNewImportPath:
    """从 harness.tasks 导入。"""

    def test_import_all_types(self) -> None:
        from harness.tasks import (
            BaseTask,
            Dialogue,
            DialogueOutput,
            DialogueProgressEvent,
            DialogueTurn,
            FunctionTask,
            LLMTask,
            Parallel,
            PipelineResult,
            PipelineStep,
            PollingTask,
            Result,
            Role,
            ShellTask,
            Task,
            TaskConfig,
            result_by_type,
        )
        assert LLMTask is not None
        assert PipelineStep is not None

    def test_import_submodules(self) -> None:
        from harness.tasks.config import TaskConfig
        from harness.tasks.result import Result, PipelineResult, result_by_type
        from harness.tasks.base import BaseTask, DialogueProgressEvent
        from harness.tasks.llm import LLMTask
        from harness.tasks.function import FunctionTask
        from harness.tasks.shell import ShellTask
        from harness.tasks.polling import PollingTask
        from harness.tasks.parallel import Parallel
        from harness.tasks.dialogue import Dialogue, Role, DialogueTurn, DialogueOutput
        from harness.tasks.types import PipelineStep, Task
        assert TaskConfig is not None


class TestLegacyImportPath:
    """从 harness.task（旧路径）导入——向后兼容。"""

    def test_import_all_types(self) -> None:
        from harness.task import (
            BaseTask,
            Dialogue,
            DialogueOutput,
            DialogueProgressEvent,
            DialogueTurn,
            FunctionTask,
            LLMTask,
            Parallel,
            PipelineResult,
            PipelineStep,
            PollingTask,
            Result,
            Role,
            ShellTask,
            Task,
            TaskConfig,
            result_by_type,
        )
        assert LLMTask is not None


class TestIdentity:
    """两条路径导入的是同一个类。"""

    def test_same_classes(self) -> None:
        from harness.task import LLMTask as Old
        from harness.tasks import LLMTask as New
        assert Old is New

    def test_same_result(self) -> None:
        from harness.task import Result as Old
        from harness.tasks import Result as New
        assert Old is New

    def test_same_task_config(self) -> None:
        from harness.task import TaskConfig as Old
        from harness.tasks import TaskConfig as New
        assert Old is New

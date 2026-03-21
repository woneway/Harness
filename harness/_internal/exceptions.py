"""Harness 自定义异常。"""

from __future__ import annotations


class TaskFailedError(Exception):
    """Task 超过最大重试次数或不可恢复失败时抛出。"""

    def __init__(
        self,
        run_id: str,
        task_index: str,
        task_type: str,
        error: str,
        partial_results: list | None = None,
    ) -> None:
        self.run_id = run_id
        self.task_index = task_index
        self.task_type = task_type
        self.error = error
        self.partial_results = partial_results or []
        super().__init__(
            f"Task {task_index} [{task_type}] failed in run {run_id}: {error}"
        )


class ClaudeNotFoundError(Exception):
    """Claude CLI 未安装或未在 PATH 中时抛出。"""

    def __init__(self) -> None:
        super().__init__(
            "Claude CLI not found. Please install it: https://claude.ai/code"
        )


class InvalidPipelineError(Exception):
    """Pipeline 配置无效时抛出，例如 Parallel 嵌套。"""

    pass


class OutputSchemaError(Exception):
    """FunctionTask output_schema 校验失败时抛出，不触发重试。"""

    def __init__(self, task_index: str, expected_type: type, actual_type: type) -> None:
        self.task_index = task_index
        self.expected_type = expected_type
        self.actual_type = actual_type
        super().__init__(
            f"Task {task_index} output schema validation failed: "
            f"expected {expected_type.__name__}, got {actual_type.__name__}"
        )

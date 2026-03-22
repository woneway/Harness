"""StorageProtocol — 存储层抽象接口。"""

from __future__ import annotations

from typing import Any, Protocol


class StorageProtocol(Protocol):
    """存储层协议，所有存储实现需满足此接口。"""

    async def save_run(
        self,
        run_id: str,
        project_path: str,
        name: str | None,
    ) -> None:
        """创建一条新的 run 记录（状态 running）。"""
        ...

    async def update_run(
        self,
        run_id: str,
        *,
        status: str,
        total_tokens: int | None = None,
        summary: str | None = None,
        error: str | None = None,
    ) -> None:
        """更新 run 记录（状态、token 数、摘要、错误信息）。"""
        ...

    async def save_task_log(
        self,
        run_id: str,
        task_index: str,
        task_type: str,
        *,
        output: Any = None,
        output_schema_class: str | None = None,
        raw_text: str | None = None,
        tokens_used: int = 0,
        duration_seconds: float = 0.0,
        attempt: int = 1,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """保存单个 Task 的执行日志。"""
        ...

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        """获取指定 run 的信息。"""
        ...

    async def list_runs(
        self,
        project_path: str,
        *,
        limit: int = 20,
        failed_only: bool = False,
    ) -> list[dict[str, Any]]:
        """列出指定项目的 run 历史。"""
        ...

    async def get_task_logs(
        self,
        run_id: str,
        *,
        success_only: bool = False,
    ) -> list[dict[str, Any]]:
        """获取指定 run 的所有 task 日志。"""
        ...

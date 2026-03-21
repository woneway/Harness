"""SQLAlchemy async 通用实现（SQLite / MySQL / PG）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone


def _now() -> datetime:
    return datetime.now(timezone.utc)
from typing import Any

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from harness.storage.models import Base, Run, TaskLog


class SQLStorage:
    """StorageProtocol 的 SQLAlchemy async 实现。

    支持 SQLite（默认）、MySQL、PostgreSQL。
    SQLite 首次连接后执行 PRAGMA journal_mode=WAL。
    """

    def __init__(self, url: str) -> None:
        self._url = url
        connect_args: dict[str, Any] = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        self._engine = create_async_engine(url, connect_args=connect_args)
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init(self) -> None:
        """创建数据库表，SQLite 启用 WAL 模式。"""
        async with self._engine.begin() as conn:
            if self._url.startswith("sqlite"):
                await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.run_sync(Base.metadata.create_all)

    async def save_run(
        self,
        run_id: str,
        project_path: str,
        name: str | None,
    ) -> None:
        async with self._session_factory() as session:
            run = Run(
                id=run_id,
                project_path=project_path,
                name=name,
                started_at=_now(),
                status="running",
            )
            session.add(run)
            await session.commit()

    async def update_run(
        self,
        run_id: str,
        *,
        status: str,
        total_tokens: int | None = None,
        summary: str | None = None,
        error: str | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status}
        if status in ("success", "failed"):
            values["completed_at"] = _now()
        if total_tokens is not None:
            values["total_tokens"] = total_tokens
        if summary is not None:
            values["summary"] = summary
        if error is not None:
            values["error"] = error

        async with self._session_factory() as session:
            stmt = update(Run).where(Run.id == run_id).values(**values)
            await session.execute(stmt)
            await session.commit()

    async def save_task_log(
        self,
        run_id: str,
        task_index: str,
        task_type: str,
        *,
        prompt_preview: str | None = None,
        output: Any = None,
        raw_text: str | None = None,
        tokens_used: int = 0,
        duration_seconds: float = 0.0,
        attempt: int = 1,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        # 序列化 output 为 JSON 字符串
        output_str: str | None
        if output is None:
            output_str = None
        elif hasattr(output, "model_dump_json"):
            output_str = output.model_dump_json()
        else:
            try:
                output_str = json.dumps(output, ensure_ascii=False, default=str)
            except Exception:
                output_str = str(output)

        async with self._session_factory() as session:
            log = TaskLog(
                run_id=run_id,
                task_index=task_index,
                task_type=task_type,
                prompt_preview=prompt_preview,
                output=output_str,
                raw_text=raw_text,
                tokens_used=tokens_used,
                duration_seconds=duration_seconds,
                attempt=attempt,
                success=success,
                error=error,
                created_at=_now(),
            )
            session.add(log)
            await session.commit()

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        async with self._session_factory() as session:
            result = await session.get(Run, run_id)
            if result is None:
                return None
            return {
                "id": result.id,
                "project_path": result.project_path,
                "name": result.name,
                "started_at": result.started_at,
                "completed_at": result.completed_at,
                "status": result.status,
                "total_tokens": result.total_tokens,
                "summary": result.summary,
                "error": result.error,
            }

    async def list_runs(
        self,
        project_path: str,
        *,
        limit: int = 20,
        failed_only: bool = False,
    ) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            stmt = (
                select(Run)
                .where(Run.project_path == project_path)
                .order_by(Run.started_at.desc())
                .limit(limit)
            )
            if failed_only:
                stmt = stmt.where(Run.status == "failed")

            rows = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "id": r.id,
                    "project_path": r.project_path,
                    "name": r.name,
                    "started_at": r.started_at,
                    "completed_at": r.completed_at,
                    "status": r.status,
                    "total_tokens": r.total_tokens,
                    "summary": r.summary,
                    "error": r.error,
                }
                for r in rows
            ]

    async def get_task_logs(
        self,
        run_id: str,
        *,
        success_only: bool = False,
    ) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            stmt = select(TaskLog).where(TaskLog.run_id == run_id).order_by(TaskLog.id)
            if success_only:
                stmt = stmt.where(TaskLog.success.is_(True))

            rows = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "id": r.id,
                    "run_id": r.run_id,
                    "task_index": r.task_index,
                    "task_type": r.task_type,
                    "prompt_preview": r.prompt_preview,
                    "output": r.output,
                    "raw_text": r.raw_text,
                    "tokens_used": r.tokens_used,
                    "duration_seconds": r.duration_seconds,
                    "attempt": r.attempt,
                    "success": r.success,
                    "error": r.error,
                    "created_at": r.created_at,
                }
                for r in rows
            ]

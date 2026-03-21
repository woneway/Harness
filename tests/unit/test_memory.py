"""tests/unit/test_memory.py — Memory 注入格式和增长控制验证。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from harness.memory import Memory


class MockStorage:
    """最小 StorageProtocol 实现，用于测试。"""

    def __init__(self, runs: list[dict[str, Any]]) -> None:
        self._runs = runs

    async def list_runs(self, project_path: str, *, limit: int = 20, **kwargs) -> list[dict]:
        return self._runs[:limit]

    async def get_task_logs(self, run_id: str, *, success_only: bool = False) -> list[dict]:
        return []

    async def save_run(self, *args, **kwargs) -> None: ...
    async def update_run(self, *args, **kwargs) -> None: ...
    async def save_task_log(self, *args, **kwargs) -> None: ...
    async def get_run(self, run_id: str) -> dict | None: ...


class TestMemoryBuildInjection:
    @pytest.mark.asyncio
    async def test_empty_storage_and_no_file(self, tmp_path: Path) -> None:
        memory = Memory()
        storage = MockStorage(runs=[])
        result = await memory.build_injection(storage, tmp_path)
        assert result == ""

    @pytest.mark.asyncio
    async def test_run_history_format(self, tmp_path: Path) -> None:
        runs = [
            {"started_at": datetime(2026, 3, 20), "summary": "fixed auth bug"},
            {"started_at": datetime(2026, 3, 19), "summary": "added rate limit"},
        ]
        storage = MockStorage(runs=runs)
        memory = Memory(history_runs=3)
        result = await memory.build_injection(storage, tmp_path)

        assert "=== 最近运行历史 ===" in result
        assert "2026-03-19" in result
        assert "2026-03-20" in result
        assert "fixed auth bug" in result
        assert "added rate limit" in result

    @pytest.mark.asyncio
    async def test_memory_file_included(self, tmp_path: Path) -> None:
        harness_dir = tmp_path / ".harness"
        harness_dir.mkdir()
        (harness_dir / "memory.md").write_text("project convention: use snake_case")

        storage = MockStorage(runs=[])
        memory = Memory()
        result = await memory.build_injection(storage, tmp_path)

        assert "=== 项目记忆 ===" in result
        assert "project convention: use snake_case" in result

    @pytest.mark.asyncio
    async def test_max_tokens_truncation(self, tmp_path: Path) -> None:
        """超过 max_tokens 时从头部截断。"""
        runs = [{"started_at": datetime(2026, 3, 20), "summary": "x" * 2000}]
        storage = MockStorage(runs=runs)
        memory = Memory(max_tokens=100)
        result = await memory.build_injection(storage, tmp_path)
        assert len(result) <= 100

    @pytest.mark.asyncio
    async def test_respects_history_runs_limit(self, tmp_path: Path) -> None:
        runs = [
            {"started_at": datetime(2026, 3, 20), "summary": f"run {i}"}
            for i in range(10)
        ]
        storage = MockStorage(runs=runs)
        memory = Memory(history_runs=2)
        result = await memory.build_injection(storage, tmp_path)
        # 只注入最近 2 条
        count = result.count("run ")
        assert count <= 2


class TestWriteMemoryUpdate:
    def test_append_to_new_file(self, tmp_path: Path) -> None:
        memory = Memory(memory_file=".harness/memory.md")
        memory.write_memory_update(tmp_path, "first update")
        content = (tmp_path / ".harness" / "memory.md").read_text()
        assert "first update" in content

    def test_append_to_existing_file(self, tmp_path: Path) -> None:
        harness_dir = tmp_path / ".harness"
        harness_dir.mkdir()
        (harness_dir / "memory.md").write_text("existing content")

        memory = Memory(memory_file=".harness/memory.md")
        memory.write_memory_update(tmp_path, "new update")
        content = (tmp_path / ".harness" / "memory.md").read_text()
        assert "existing content" in content
        assert "new update" in content

    def test_truncation_when_over_80_percent(self, tmp_path: Path) -> None:
        """写入后文件超过 max_tokens 的 80% 时做硬截断。"""
        harness_dir = tmp_path / ".harness"
        harness_dir.mkdir()
        memory_path = harness_dir / "memory.md"
        # 写入接近阈值的内容
        memory_path.write_text("x" * 900)

        memory = Memory(memory_file=".harness/memory.md", max_tokens=1000)
        memory.write_memory_update(tmp_path, "y" * 200)
        content = memory_path.read_text()
        # 80% of 1000 = 800
        assert len(content) <= 800

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        memory = Memory(memory_file=".harness/subdir/memory.md")
        memory.write_memory_update(tmp_path, "hello")
        assert (tmp_path / ".harness" / "subdir" / "memory.md").exists()

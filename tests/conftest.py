"""Shared pytest fixtures."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from harness import Harness
from harness.runners.base import AbstractRunner, RunnerResult


class _MockRunner(AbstractRunner):
    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        return RunnerResult(text="mock result", tokens_used=5, session_id="s1")


@pytest.fixture
def mock_runner() -> _MockRunner:
    return _MockRunner()


def make_mock_storage() -> MagicMock:
    """构造一个完整 mock 的 SQLStorage，所有方法均为 AsyncMock。"""
    storage = MagicMock()
    storage.init = AsyncMock()
    storage.save_run = AsyncMock()
    storage.save_task_log = AsyncMock()
    storage.update_run = AsyncMock()
    storage.get_task_logs = AsyncMock(return_value=[])
    return storage


def patch_storage(h: Harness, storage: MagicMock) -> None:
    """将 Harness 内部的存储替换为 mock，并标记已初始化。"""
    h._storage = storage
    h._initialized = True

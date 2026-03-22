"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from harness.runners.base import AbstractRunner, RunnerResult


class _MockRunner(AbstractRunner):
    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        return RunnerResult(text="mock result", tokens_used=5, session_id="s1")


@pytest.fixture
def mock_runner() -> _MockRunner:
    return _MockRunner()

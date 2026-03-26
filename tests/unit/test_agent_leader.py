"""tests/unit/test_agent_leader.py — AgentLeader 白名单约束验证。"""

from __future__ import annotations

import pytest

from harness.runners.agent_leader import AgentLeader
from harness.runners.base import AbstractRunner, RunnerResult


class SpyRunner(AbstractRunner):
    """记录最近一次调用参数的 runner。"""

    def __init__(self) -> None:
        self.last_system_prompt: str | None = None
        self.last_prompt: str | None = None
        self.last_kwargs: dict = {}

    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.last_kwargs = kwargs
        return RunnerResult(text="ok", tokens_used=5, session_id=None)


class TestAgentLeader:
    @pytest.mark.asyncio
    async def test_appends_whitelist_to_system_prompt(self) -> None:
        spy = SpyRunner()
        leader = AgentLeader(agents=["agent-a", "agent-b"], runner=spy)

        await leader.execute("do something", system_prompt="base sp", session_id=None)

        assert spy.last_system_prompt is not None
        assert "base sp" in spy.last_system_prompt
        assert "agent-a" in spy.last_system_prompt
        assert "agent-b" in spy.last_system_prompt
        assert "白名单" in spy.last_system_prompt

    @pytest.mark.asyncio
    async def test_passes_prompt_and_session(self) -> None:
        spy = SpyRunner()
        leader = AgentLeader(agents=["x"], runner=spy)

        await leader.execute("hello", system_prompt="sp", session_id="s123")

        assert spy.last_prompt == "hello"

    @pytest.mark.asyncio
    async def test_returns_runner_result(self) -> None:
        spy = SpyRunner()
        leader = AgentLeader(agents=["a"], runner=spy)

        result = await leader.execute("q", system_prompt="", session_id=None)

        assert result.text == "ok"
        assert result.tokens_used == 5

    @pytest.mark.asyncio
    async def test_default_runner_is_claude_cli(self) -> None:
        from harness.runners.claude_cli import ClaudeCliRunner

        leader = AgentLeader(agents=["a"])
        assert isinstance(leader._runner, ClaudeCliRunner)

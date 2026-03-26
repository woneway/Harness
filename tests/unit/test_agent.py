"""tests/unit/test_agent.py — Agent 类测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from harness.agent import Agent
from harness.runners.base import AbstractRunner, RunnerResult
from harness.state import State
from harness.tasks.dialogue import Role


class MockRunner(AbstractRunner):
    def __init__(self, text: str = "mock response") -> None:
        self._text = text

    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        return RunnerResult(text=self._text, tokens_used=10, session_id=None)


class CapturingRunner(AbstractRunner):
    """记录调用参数的 runner。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "session_id": session_id,
            "kwargs": kwargs,
        })
        return RunnerResult(text="captured", tokens_used=5, session_id=None)


# ---------------------------------------------------------------------------
# 创建
# ---------------------------------------------------------------------------


class TestAgentCreation:
    def test_required_fields(self) -> None:
        a = Agent(name="test", system_prompt="you are a tester")
        assert a.name == "test"
        assert a.system_prompt == "you are a tester"
        assert a.runner is None
        assert a.tools == []

    def test_with_runner(self) -> None:
        runner = MockRunner()
        a = Agent(name="test", system_prompt="sp", runner=runner)
        assert a.runner is runner

    def test_tools_default_empty(self) -> None:
        a = Agent(name="a", system_prompt="sp")
        assert a.tools == []

    def test_tools_isolated_between_instances(self) -> None:
        a1 = Agent(name="a1", system_prompt="sp")
        a2 = Agent(name="a2", system_prompt="sp")
        a1.tools.append("tool1")
        assert a2.tools == []


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_run_string_prompt(self) -> None:
        a = Agent(name="analyst", system_prompt="you analyze", runner=MockRunner("analysis"))
        result = await a.run("analyze this")
        assert result == "analysis"

    @pytest.mark.asyncio
    async def test_run_passes_system_prompt(self) -> None:
        runner = CapturingRunner()
        a = Agent(name="a", system_prompt="be concise", runner=runner)
        await a.run("hello")
        assert runner.calls[0]["system_prompt"] == "be concise"

    @pytest.mark.asyncio
    async def test_run_passes_session_id_none(self) -> None:
        runner = CapturingRunner()
        a = Agent(name="a", system_prompt="sp", runner=runner)
        await a.run("hello")
        assert runner.calls[0]["session_id"] is None

    @pytest.mark.asyncio
    async def test_run_no_runner_raises(self) -> None:
        a = Agent(name="no-runner", system_prompt="sp")
        with pytest.raises(ValueError, match="has no runner"):
            await a.run("hello")

    @pytest.mark.asyncio
    async def test_run_callable_prompt(self) -> None:
        runner = CapturingRunner()
        a = Agent(name="a", system_prompt="sp", runner=runner)

        class MyState(State):
            data: str = "test_data"

        state = MyState()
        await a.run(lambda s: f"analyze {s.data}", state=state)
        assert runner.calls[0]["prompt"] == "analyze test_data"

    @pytest.mark.asyncio
    async def test_run_callable_prompt_no_state_raises(self) -> None:
        a = Agent(name="a", system_prompt="sp", runner=MockRunner())
        with pytest.raises(ValueError, match="state is None"):
            await a.run(lambda s: "hello")

    @pytest.mark.asyncio
    async def test_run_with_output_schema(self) -> None:
        from pydantic import BaseModel

        class Analysis(BaseModel):
            score: float

        runner = CapturingRunner()
        a = Agent(name="a", system_prompt="sp", runner=runner)
        await a.run("analyze", output_schema=Analysis)
        assert "output_schema_json" in runner.calls[0]["kwargs"]

    @pytest.mark.asyncio
    async def test_run_without_output_schema(self) -> None:
        runner = CapturingRunner()
        a = Agent(name="a", system_prompt="sp", runner=runner)
        await a.run("hello")
        assert "output_schema_json" not in runner.calls[0]["kwargs"]


# ---------------------------------------------------------------------------
# as_role()
# ---------------------------------------------------------------------------


class TestAgentAsRole:
    def test_returns_role(self) -> None:
        a = Agent(name="analyst", system_prompt="be analytical", runner=MockRunner())
        role = a.as_role(lambda ctx: "prompt")
        assert isinstance(role, Role)

    def test_role_inherits_name(self) -> None:
        a = Agent(name="analyst", system_prompt="sp", runner=MockRunner())
        role = a.as_role(lambda ctx: "p")
        assert role.name == "analyst"

    def test_role_inherits_system_prompt(self) -> None:
        a = Agent(name="a", system_prompt="be concise", runner=MockRunner())
        role = a.as_role(lambda ctx: "p")
        assert role.system_prompt == "be concise"

    def test_role_inherits_runner(self) -> None:
        runner = MockRunner()
        a = Agent(name="a", system_prompt="sp", runner=runner)
        role = a.as_role(lambda ctx: "p")
        assert role.runner is runner

    def test_role_inherits_none_runner(self) -> None:
        a = Agent(name="a", system_prompt="sp", runner=None)
        role = a.as_role(lambda ctx: "p")
        assert role.runner is None

    def test_role_uses_provided_prompt(self) -> None:
        a = Agent(name="a", system_prompt="sp")
        prompt_fn = lambda ctx: f"hi from {ctx.role_name}"
        role = a.as_role(prompt_fn)
        assert role.prompt is prompt_fn

    def test_multiple_as_role_independent(self) -> None:
        a = Agent(name="a", system_prompt="sp")
        r1 = a.as_role(lambda ctx: "prompt1")
        r2 = a.as_role(lambda ctx: "prompt2")
        assert r1.prompt is not r2.prompt

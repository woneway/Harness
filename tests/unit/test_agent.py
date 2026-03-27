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


# ---------------------------------------------------------------------------
# build_system_prompt()
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_system_prompt_takes_priority(self) -> None:
        """system_prompt 非空时直接返回，忽略结构化字段。"""
        a = Agent(
            name="test",
            system_prompt="direct prompt",
            description="should be ignored",
            goal="should be ignored",
        )
        assert a.build_system_prompt() == "direct prompt"

    def test_name_only(self) -> None:
        """只有 name，无结构化字段。"""
        a = Agent(name="分析师")
        assert a.build_system_prompt() == "# 角色\n你是分析师。"

    def test_description(self) -> None:
        """name + description。"""
        a = Agent(name="分析师", description="专注技术分析的短线交易员。")
        result = a.build_system_prompt()
        assert "# 角色" in result
        assert "你是分析师。专注技术分析的短线交易员。" in result

    def test_goal(self) -> None:
        """name + goal。"""
        a = Agent(name="trader", goal="最大化短期收益")
        result = a.build_system_prompt()
        assert "# 目标\n最大化短期收益" in result

    def test_backstory(self) -> None:
        """name + backstory。"""
        a = Agent(name="trader", backstory="10年游资经验")
        result = a.build_system_prompt()
        assert "# 背景\n10年游资经验" in result

    def test_constraints(self) -> None:
        """name + constraints。"""
        a = Agent(name="trader", constraints=["不追高", "严格止损"])
        result = a.build_system_prompt()
        assert "# 行为约束" in result
        assert "- 不追高" in result
        assert "- 严格止损" in result

    def test_all_structured_fields(self) -> None:
        """所有结构化字段组合。"""
        a = Agent(
            name="龙头猎手",
            description="辨识龙头的短线选手。",
            goal="抓住每日龙头股",
            backstory="从涨停板战法起家，擅长辨识市场主线。",
            constraints=["只做龙头", "不碰垃圾股"],
        )
        result = a.build_system_prompt()
        assert result.startswith("# 角色\n你是龙头猎手。辨识龙头的短线选手。")
        assert "# 目标\n抓住每日龙头股" in result
        assert "# 背景\n从涨停板战法起家" in result
        assert "# 行为约束\n- 只做龙头\n- 不碰垃圾股" in result
        # 各段落以双换行分隔
        assert "\n\n# 目标" in result
        assert "\n\n# 背景" in result
        assert "\n\n# 行为约束" in result

    def test_empty_constraints_omitted(self) -> None:
        """空 constraints 列表不生成约束段落。"""
        a = Agent(name="a", constraints=[])
        assert "行为约束" not in a.build_system_prompt()


# ---------------------------------------------------------------------------
# run() with structured fields
# ---------------------------------------------------------------------------


class TestAgentRunWithStructuredFields:
    @pytest.mark.asyncio
    async def test_run_uses_build_system_prompt(self) -> None:
        """run() 使用 build_system_prompt() 而非 self.system_prompt。"""
        runner = CapturingRunner()
        a = Agent(
            name="分析师",
            description="技术分析专家。",
            goal="精准判断买卖点",
            runner=runner,
        )
        await a.run("hello")
        sp = runner.calls[0]["system_prompt"]
        assert "你是分析师。技术分析专家。" in sp
        assert "# 目标\n精准判断买卖点" in sp

    @pytest.mark.asyncio
    async def test_run_system_prompt_priority_over_structured(self) -> None:
        """system_prompt 非空时 run() 用它而非结构化字段。"""
        runner = CapturingRunner()
        a = Agent(
            name="a",
            system_prompt="direct",
            description="ignored",
            runner=runner,
        )
        await a.run("hello")
        assert runner.calls[0]["system_prompt"] == "direct"


# ---------------------------------------------------------------------------
# as_role() with structured fields
# ---------------------------------------------------------------------------


class TestAgentAsRoleWithStructuredFields:
    def test_as_role_uses_build_system_prompt(self) -> None:
        """as_role() 生成的 Role 使用 build_system_prompt()。"""
        a = Agent(
            name="trader",
            description="激进短线手。",
            goal="打板",
            runner=MockRunner(),
        )
        role = a.as_role(lambda ctx: "p")
        assert "你是trader。激进短线手。" in role.system_prompt
        assert "# 目标\n打板" in role.system_prompt

    def test_as_role_system_prompt_priority(self) -> None:
        """system_prompt 非空时 as_role() 用它。"""
        a = Agent(name="a", system_prompt="direct sp", description="ignored")
        role = a.as_role(lambda ctx: "p")
        assert role.system_prompt == "direct sp"


# ---------------------------------------------------------------------------
# Agent 创建（增强字段）
# ---------------------------------------------------------------------------


class TestAgentCreationEnhanced:
    def test_structured_fields_default_empty(self) -> None:
        a = Agent(name="a")
        assert a.system_prompt == ""
        assert a.description == ""
        assert a.goal == ""
        assert a.backstory == ""
        assert a.constraints == []

    def test_structured_fields_assignment(self) -> None:
        a = Agent(
            name="x",
            description="d",
            goal="g",
            backstory="b",
            constraints=["c1", "c2"],
        )
        assert a.description == "d"
        assert a.goal == "g"
        assert a.backstory == "b"
        assert a.constraints == ["c1", "c2"]

    def test_constraints_isolated_between_instances(self) -> None:
        a1 = Agent(name="a1")
        a2 = Agent(name="a2")
        a1.constraints.append("no_chase")
        assert a2.constraints == []

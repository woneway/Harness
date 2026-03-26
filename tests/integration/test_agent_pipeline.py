"""tests/integration/test_agent_pipeline.py — Agent 在 pipeline 中的集成测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import Agent, Dialogue, FunctionTask, Harness, LLMTask, State
from harness.runners.base import AbstractRunner, RunnerResult
from harness.tasks.dialogue import DialogueOutput


class MockRunner(AbstractRunner):
    def __init__(self, text: str = "mock") -> None:
        self._text = text

    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        return RunnerResult(text=self._text, tokens_used=5, session_id="s1")


class TestAgentInPipeline:
    @pytest.mark.asyncio
    async def test_agent_run_standalone(self, tmp_path: Path) -> None:
        """Agent.run() 独立调用。"""
        agent = Agent(name="analyst", system_prompt="analyze", runner=MockRunner("analysis result"))
        result = await agent.run("analyze this")
        assert result == "analysis result"

    @pytest.mark.asyncio
    async def test_agent_via_llm_task_with_runner(self, tmp_path: Path) -> None:
        """Agent 的 runner 在 LLMTask 中使用。"""
        agent_runner = MockRunner("agent analysis")
        agent = Agent(name="analyst", system_prompt="analyze", runner=agent_runner)

        h = Harness(str(tmp_path), runner=MockRunner())
        pr = await h.pipeline([
            LLMTask(prompt="analyze this", runner=agent_runner, system_prompt=agent.system_prompt),
        ])

        assert pr.results[0].success
        assert pr.results[0].output == "agent analysis"

    @pytest.mark.asyncio
    async def test_agent_as_role_in_dialogue(self, tmp_path: Path) -> None:
        """Agent.as_role() 在 Dialogue 中使用。"""
        runner = MockRunner("mock dialogue response")
        agent_a = Agent(name="analyst", system_prompt="be analytical", runner=runner)
        agent_b = Agent(name="trader", system_prompt="be practical", runner=runner)

        h = Harness(str(tmp_path), runner=runner)
        pr = await h.pipeline([
            Dialogue(
                roles=[
                    agent_a.as_role(lambda ctx: "analyze the market"),
                    agent_b.as_role(lambda ctx: f"respond to: {ctx.last_from('analyst') or 'nothing'}"),
                ],
                max_rounds=2,
            ),
        ])

        assert pr.results[0].success
        output = pr.results[0].output
        assert isinstance(output, DialogueOutput)
        assert output.rounds_completed == 2
        assert output.total_turns == 4  # 2 rounds × 2 roles

    @pytest.mark.asyncio
    async def test_agent_with_state_in_dialogue(self, tmp_path: Path) -> None:
        """Agent.as_role() 可以在 prompt 中访问 state。"""
        runner = MockRunner("mock response")
        agent = Agent(name="analyst", system_prompt="sp", runner=runner)

        class MyState(State):
            data: str = "initial_data"

        h = Harness(str(tmp_path), runner=runner)
        prompts_seen = []

        def capture_prompt(ctx):
            prompt = f"analyze: {ctx.state.data}"
            prompts_seen.append(prompt)
            return prompt

        pr = await h.pipeline([
            Dialogue(
                roles=[agent.as_role(capture_prompt)],
                max_rounds=1,
            ),
        ], state=MyState())

        assert pr.results[0].success
        assert len(prompts_seen) == 1
        assert "initial_data" in prompts_seen[0]

    @pytest.mark.asyncio
    async def test_agent_function_task_then_dialogue(self, tmp_path: Path) -> None:
        """FunctionTask 写入 state，Agent 参与 Dialogue 读取 state。"""
        runner = MockRunner("mock output")
        analyst = Agent(name="analyst", system_prompt="sp", runner=runner)

        class MyState(State):
            prep: str = ""

        def prepare(state: MyState) -> str:
            return "prepared data"

        h = Harness(str(tmp_path), runner=runner)
        pr = await h.pipeline([
            FunctionTask(fn=prepare, output_key="prep"),
            Dialogue(
                roles=[analyst.as_role(lambda ctx: f"discuss: {ctx.state.prep}")],
                max_rounds=1,
            ),
        ], state=MyState())

        assert len(pr.results) == 2
        assert pr.results[0].output == "prepared data"
        assert pr.results[1].success

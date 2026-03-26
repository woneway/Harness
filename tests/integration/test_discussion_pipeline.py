"""tests/integration/test_discussion_pipeline.py — Discussion pipeline 集成测试。"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from harness import Harness
from harness.agent import Agent
from harness.state import State
from harness.tasks import (
    Condition,
    Discussion,
    DiscussionOutput,
    FunctionTask,
    LLMTask,
    Parallel,
)
from harness.tasks.discussion import all_agree_on
from harness._internal.exceptions import InvalidPipelineError
from harness.runners.base import AbstractRunner, RunnerResult
from tests.conftest import make_mock_storage, patch_storage


class SimplePosition(BaseModel):
    choice: str


def _pos_json(position: dict) -> str:
    return json.dumps(position)


class _DiscussionRunner(AbstractRunner):
    """两阶段 mock runner：根据 output_schema_json 区分 Phase 1/2。"""

    def __init__(
        self,
        responses: list[str],
        extraction_responses: list[str] | None = None,
    ) -> None:
        self._responses = responses
        self._extraction_responses = extraction_responses or []
        self._phase1_idx = 0
        self._phase2_idx = 0

    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        if kwargs.get("output_schema_json") is not None:
            # Phase 2: extraction
            text = self._extraction_responses[self._phase2_idx % len(self._extraction_responses)]
            self._phase2_idx += 1
            return RunnerResult(text=text, tokens_used=5, session_id=None)
        else:
            # Phase 1: free text
            text = self._responses[self._phase1_idx % len(self._responses)]
            self._phase1_idx += 1
            return RunnerResult(text=text, tokens_used=10, session_id=f"s{self._phase1_idx}")


class _SimpleRunner(AbstractRunner):
    """Mock runner for LLMTask (returns plain text)."""

    def __init__(self, text: str = "plan result") -> None:
        self._text = text

    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        return RunnerResult(text=self._text, tokens_used=5, session_id="s-llm")


# ── Discussion in pipeline ─────────────────────────────────────────────────


class TestDiscussionInPipeline:
    @pytest.mark.asyncio
    async def test_function_discussion_llm_pipeline(self) -> None:
        """FunctionTask → Discussion → LLMTask pipeline。"""
        disc_runner = _DiscussionRunner(
            responses=["a picks X", "b picks X"],
            extraction_responses=[
                _pos_json({"choice": "X"}),
                _pos_json({"choice": "X"}),
            ],
        )
        llm_runner = _SimpleRunner("final plan")

        a1 = Agent(name="a", system_prompt="sp", runner=disc_runner)
        a2 = Agent(name="b", system_prompt="sp", runner=disc_runner)

        class MyState(State):
            data: str = ""
            discussion: DiscussionOutput | None = None
            plan: str = ""

        h = Harness(project_path="/tmp/test", runner=llm_runner)
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline([
            FunctionTask(fn=lambda s: "market data", output_key="data"),
            Discussion(
                agents=[a1, a2],
                position_schema=SimplePosition,
                topic="选股",
                max_rounds=1,
                output_key="discussion",
            ),
            LLMTask(
                prompt=lambda state: f"Plan based on: {state.discussion.converged}",
                output_key="plan",
            ),
        ], state=MyState())

        assert len(pr.results) == 3
        assert pr.results[0].output == "market data"
        assert isinstance(pr.results[1].output, DiscussionOutput)
        assert pr.results[2].output == "final plan"

    @pytest.mark.asyncio
    async def test_discussion_state_output_key(self) -> None:
        """Discussion + State + output_key 写入。"""
        runner = _DiscussionRunner(
            responses=["ok"],
            extraction_responses=[_pos_json({"choice": "AAPL"})],
        )
        a1 = Agent(name="a", system_prompt="sp", runner=runner)

        class MyState(State):
            result: DiscussionOutput | None = None

        state = MyState()
        h = Harness(project_path="/tmp/test", runner=runner)
        patch_storage(h, make_mock_storage())

        await h.pipeline([
            Discussion(
                agents=[a1],
                position_schema=SimplePosition,
                topic="test",
                max_rounds=1,
                output_key="result",
            ),
        ], state=state)

        assert state.result is not None
        assert isinstance(state.result, DiscussionOutput)
        assert state.result.final_positions["a"].choice == "AAPL"

    @pytest.mark.asyncio
    async def test_discussion_convergence_in_pipeline(self) -> None:
        """Discussion with convergence stops early in pipeline。"""
        runner = _DiscussionRunner(
            responses=["r0 a", "r0 b", "r1 a→Y", "r1 b"],
            extraction_responses=[
                _pos_json({"choice": "X"}),
                _pos_json({"choice": "Y"}),
                _pos_json({"choice": "Y"}),
                _pos_json({"choice": "Y"}),
            ],
        )
        a1 = Agent(name="a", system_prompt="sp", runner=runner)
        a2 = Agent(name="b", system_prompt="sp", runner=runner)

        h = Harness(project_path="/tmp/test", runner=runner)
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline([
            Discussion(
                agents=[a1, a2],
                position_schema=SimplePosition,
                topic="test",
                max_rounds=10,
                convergence=all_agree_on("choice"),
            ),
        ])

        output = pr.results[0].output
        assert output.converged is True
        assert output.convergence_round == 1


# ── Discussion in Condition ────────────────────────────────────────────────


class TestDiscussionInCondition:
    @pytest.mark.asyncio
    async def test_discussion_in_condition_branch(self) -> None:
        """Discussion 嵌入 Condition if_true 分支。"""
        runner = _DiscussionRunner(
            responses=["ok"],
            extraction_responses=[_pos_json({"choice": "A"})],
        )
        a1 = Agent(name="a", system_prompt="sp", runner=runner)

        class MyState(State):
            should_discuss: bool = True

        h = Harness(project_path="/tmp/test", runner=runner)
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline([
            Condition(
                check=lambda s: s.should_discuss,
                if_true=[
                    Discussion(
                        agents=[a1],
                        position_schema=SimplePosition,
                        topic="test",
                        max_rounds=1,
                    ),
                ],
                if_false=[
                    FunctionTask(fn=lambda s: "skipped"),
                ],
            ),
        ], state=MyState())

        assert len(pr.results) == 1
        assert isinstance(pr.results[0].output, DiscussionOutput)


# ── Agent.task() in pipeline ───────────────────────────────────────────────


class TestAgentTaskInPipeline:
    @pytest.mark.asyncio
    async def test_agent_task_in_pipeline(self) -> None:
        """Agent.task() 创建的 LLMTask 在 pipeline 中正常执行。"""
        runner = _SimpleRunner("agent analysis")
        agent = Agent(name="analyst", system_prompt="you analyze", runner=runner)

        h = Harness(project_path="/tmp/test", runner=runner)
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline([
            agent.task("analyze this", output_key="analysis"),
        ])

        assert pr.results[0].output == "agent analysis"


# ── Discussion not in Parallel ─────────────────────────────────────────────


class TestDiscussionNotInParallel:
    @pytest.mark.asyncio
    async def test_discussion_in_parallel_raises(self) -> None:
        """Discussion 嵌入 Parallel 时抛 InvalidPipelineError。"""
        runner = _DiscussionRunner(
            responses=["ok"],
            extraction_responses=[_pos_json({"choice": "A"})],
        )
        a1 = Agent(name="a", system_prompt="sp", runner=runner)

        h = Harness(project_path="/tmp/test", runner=runner)
        patch_storage(h, make_mock_storage())

        with pytest.raises(InvalidPipelineError, match="Discussion inside Parallel"):
            await h.pipeline([
                Parallel([
                    Discussion(
                        agents=[a1],
                        position_schema=SimplePosition,
                        topic="test",
                        max_rounds=1,
                    ),
                ]),
            ])


# ── position_history 累积 ─────────────────────────────────────────────────


class TestPositionHistoryAccumulation:
    @pytest.mark.asyncio
    async def test_position_history_multi_round(self) -> None:
        """position_history 跨多轮正确累积。"""
        runner = _DiscussionRunner(
            responses=["r0", "r1", "r2"],
            extraction_responses=[
                _pos_json({"choice": "X"}),
                _pos_json({"choice": "Y"}),
                _pos_json({"choice": "Z"}),
            ],
        )
        a1 = Agent(name="a", system_prompt="sp", runner=runner)

        h = Harness(project_path="/tmp/test", runner=runner)
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline([
            Discussion(
                agents=[a1],
                position_schema=SimplePosition,
                topic="test",
                max_rounds=3,
            ),
        ])

        output = pr.results[0].output
        history = output.position_history["a"]
        assert len(history) == 3
        assert [p.choice for p in history] == ["X", "Y", "Z"]


# ── final_positions 正确性 ────────────────────────────────────────────────


class TestFinalPositions:
    @pytest.mark.asyncio
    async def test_final_positions_reflect_last_round(self) -> None:
        """final_positions 反映最后一轮的立场。"""
        runner = _DiscussionRunner(
            responses=["r0a", "r0b", "r1a", "r1b"],
            extraction_responses=[
                _pos_json({"choice": "X"}),
                _pos_json({"choice": "Y"}),
                _pos_json({"choice": "Z"}),
                _pos_json({"choice": "Z"}),
            ],
        )
        a1 = Agent(name="a", system_prompt="sp", runner=runner)
        a2 = Agent(name="b", system_prompt="sp", runner=runner)

        h = Harness(project_path="/tmp/test", runner=runner)
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline([
            Discussion(
                agents=[a1, a2],
                position_schema=SimplePosition,
                topic="test",
                max_rounds=2,
            ),
        ])

        output = pr.results[0].output
        assert output.final_positions["a"].choice == "Z"
        assert output.final_positions["b"].choice == "Z"

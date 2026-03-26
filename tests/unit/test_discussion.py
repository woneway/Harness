"""tests/unit/test_discussion.py — Discussion 单元测试。"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from harness.agent import Agent
from harness.runners.base import AbstractRunner, RunnerResult
from harness.state import State
from harness.tasks.config import TaskConfig
from harness.tasks.discussion import (
    Discussion,
    DiscussionOutput,
    DiscussionProgressEvent,
    DiscussionTurn,
    all_agree_on,
    majority_agree_on,
    positions_stable,
)
from harness._internal.discussion import (
    DiscussionContext,
    _default_prompt_template,
    _make_turn_schema,
    _merge_system_prompt,
    _resolve_prompt,
    execute_discussion,
)
from harness._internal.task_index import TaskIndex


# ── 共用 fixtures ──────────────────────────────────────────────────────────


class TradingPosition(BaseModel):
    top_pick: str
    direction: str
    confidence: float


class SimplePosition(BaseModel):
    choice: str


class MockRunner(AbstractRunner):
    """返回预设 JSON 的 mock runner。"""

    def __init__(self, responses: list[str] | None = None, default: str = "") -> None:
        self._responses = list(responses) if responses else []
        self._default = default
        self._call_count = 0
        self.calls: list[dict] = []

    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "session_id": session_id,
            "kwargs": kwargs,
        })
        if self._responses:
            text = self._responses[self._call_count % len(self._responses)]
        else:
            text = self._default
        self._call_count += 1
        return RunnerResult(text=text, tokens_used=10, session_id=f"sess-{self._call_count}")


def _make_json(response: str, position: dict) -> str:
    return json.dumps({"response": response, "position": position})


def _make_agent(name: str, runner: AbstractRunner | None = None) -> Agent:
    return Agent(name=name, system_prompt=f"You are {name}", runner=runner)


# ── TaskIndex disc_round ───────────────────────────────────────────────────


class TestTaskIndexDiscRound:
    def test_disc_round_str(self) -> None:
        idx = TaskIndex.disc_round(2, 0, 1)
        assert str(idx) == "2.g0.1"

    def test_disc_round_parse(self) -> None:
        parsed = TaskIndex.parse("2.g0.1")
        assert parsed.kind == "disc_round"
        assert parsed.outer == 2
        assert parsed.round_ == 0
        assert parsed.sub == 1

    def test_disc_round_roundtrip(self) -> None:
        idx = TaskIndex.disc_round(5, 3, 2)
        assert TaskIndex.parse(str(idx)) == idx

    def test_disc_round_is_child(self) -> None:
        idx = TaskIndex.disc_round(0, 0, 0)
        assert idx.is_child is True

    def test_disc_round_outer_key(self) -> None:
        idx = TaskIndex.disc_round(3, 1, 0)
        assert idx.outer_key == "3"


# ── DiscussionContext ──────────────────────────────────────────────────────


class TestDiscussionContext:
    def test_last_response_from(self) -> None:
        turns = [
            DiscussionTurn(0, "a", "first", None, False),
            DiscussionTurn(0, "b", "second", None, False),
            DiscussionTurn(1, "a", "third", None, True),
        ]
        ctx = DiscussionContext(round=1, agent_name="b", topic="test", background="",
                                history=turns)
        assert ctx.last_response_from("a") == "third"
        assert ctx.last_response_from("b") == "second"
        assert ctx.last_response_from("c") is None

    def test_all_responses_from(self) -> None:
        turns = [
            DiscussionTurn(0, "a", "r1", None, False),
            DiscussionTurn(0, "b", "r2", None, False),
            DiscussionTurn(1, "a", "r3", None, True),
        ]
        ctx = DiscussionContext(round=1, agent_name="b", topic="test", background="",
                                history=turns)
        assert ctx.all_responses_from("a") == ["r1", "r3"]
        assert ctx.all_responses_from("b") == ["r2"]
        assert ctx.all_responses_from("c") == []

    def test_position_of(self) -> None:
        pos_a = SimplePosition(choice="X")
        ctx = DiscussionContext(round=0, agent_name="a", topic="test", background="",
                                positions={"a": pos_a})
        assert ctx.position_of("a") == pos_a
        assert ctx.position_of("b") is None

    def test_did_change(self) -> None:
        turns = [
            DiscussionTurn(0, "a", "r1", None, False),
            DiscussionTurn(1, "a", "r2", None, True),
        ]
        ctx = DiscussionContext(round=1, agent_name="b", topic="test", background="",
                                history=turns)
        assert ctx.did_change("a") is True

    def test_did_change_no_turns(self) -> None:
        ctx = DiscussionContext(round=0, agent_name="a", topic="test", background="")
        assert ctx.did_change("a") is False

    def test_defaults(self) -> None:
        ctx = DiscussionContext(round=0, agent_name="a", topic="t", background="b")
        assert ctx.history == []
        assert ctx.positions == {}
        assert ctx.position_history == {}
        assert ctx.my_position is None


# ── _make_turn_schema ──────────────────────────────────────────────────────


class TestMakeTurnSchema:
    def test_creates_valid_schema(self) -> None:
        schema = _make_turn_schema(SimplePosition)
        instance = schema(response="test", position=SimplePosition(choice="A"))
        assert instance.response == "test"
        assert instance.position.choice == "A"

    def test_json_roundtrip(self) -> None:
        schema = _make_turn_schema(TradingPosition)
        data = {"response": "my analysis", "position": {"top_pick": "AAPL", "direction": "buy", "confidence": 0.8}}
        instance = schema.model_validate(data)
        assert instance.position.top_pick == "AAPL"

    def test_schema_name_includes_position_name(self) -> None:
        schema = _make_turn_schema(SimplePosition)
        assert "SimplePosition" in schema.__name__


# ── _default_prompt_template ───────────────────────────────────────────────


class TestDefaultPromptTemplate:
    def test_round_0_prompt(self) -> None:
        agent = _make_agent("analyst")
        ctx = DiscussionContext(round=0, agent_name="analyst", topic="选股",
                                background="今日涨停10家")
        text = _default_prompt_template(agent, ctx)
        assert "选股" in text
        assert "今日涨停10家" in text
        assert "初始分析" in text

    def test_round_n_prompt(self) -> None:
        agent = _make_agent("analyst")
        ctx = DiscussionContext(round=1, agent_name="analyst", topic="选股",
                                background="")
        text = _default_prompt_template(agent, ctx)
        assert "更新或坚持" in text

    def test_includes_own_position(self) -> None:
        agent = _make_agent("analyst")
        pos = SimplePosition(choice="AAPL")
        ctx = DiscussionContext(round=1, agent_name="analyst", topic="选股",
                                background="", my_position=pos)
        text = _default_prompt_template(agent, ctx)
        assert "你的当前立场" in text
        assert "AAPL" in text

    def test_includes_others_positions(self) -> None:
        agent = _make_agent("analyst")
        ctx = DiscussionContext(
            round=1, agent_name="analyst", topic="选股", background="",
            positions={"analyst": SimplePosition(choice="AAPL"),
                       "trader": SimplePosition(choice="TSLA")},
        )
        text = _default_prompt_template(agent, ctx)
        assert "trader" in text
        assert "TSLA" in text
        # 不包含自己的重复信息
        assert text.count("analyst") <= 2  # 只在 agent_name 相关处出现

    def test_shows_position_changed_indicator(self) -> None:
        agent = _make_agent("analyst")
        turns = [DiscussionTurn(0, "trader", "changed my mind", None, True)]
        ctx = DiscussionContext(
            round=1, agent_name="analyst", topic="t", background="",
            positions={"trader": SimplePosition(choice="X")},
            history=turns,
        )
        text = _default_prompt_template(agent, ctx)
        assert "上轮已调整立场" in text

    def test_no_background(self) -> None:
        agent = _make_agent("analyst")
        ctx = DiscussionContext(round=0, agent_name="analyst", topic="t", background="")
        text = _default_prompt_template(agent, ctx)
        assert "## 背景信息" not in text


# ── _resolve_prompt ────────────────────────────────────────────────────────


class TestResolvePrompt:
    def test_agent_prompts_priority(self) -> None:
        agent = _make_agent("analyst")
        ctx = DiscussionContext(round=0, agent_name="analyst", topic="t", background="")
        disc = Discussion(
            agents=[agent],
            position_schema=SimplePosition,
            agent_prompts={"analyst": lambda c: "custom prompt"},
        )
        assert _resolve_prompt(agent, ctx, disc) == "custom prompt"

    def test_prompt_template_priority(self) -> None:
        agent = _make_agent("analyst")
        ctx = DiscussionContext(round=0, agent_name="analyst", topic="t", background="")
        disc = Discussion(
            agents=[agent],
            position_schema=SimplePosition,
            prompt_template=lambda a, c: f"template for {a.name}",
        )
        assert _resolve_prompt(agent, ctx, disc) == "template for analyst"

    def test_default_fallback(self) -> None:
        agent = _make_agent("analyst")
        ctx = DiscussionContext(round=0, agent_name="analyst", topic="选股", background="")
        disc = Discussion(agents=[agent], position_schema=SimplePosition)
        text = _resolve_prompt(agent, ctx, disc)
        assert "选股" in text


# ── 收敛工具函数 ────────────────────────────────────────────────────────────


class TestConvergenceUtils:
    def test_all_agree_on_true(self) -> None:
        history = {
            "a": [SimplePosition(choice="X")],
            "b": [SimplePosition(choice="X")],
        }
        assert all_agree_on("choice")(history) is True

    def test_all_agree_on_false(self) -> None:
        history = {
            "a": [SimplePosition(choice="X")],
            "b": [SimplePosition(choice="Y")],
        }
        assert all_agree_on("choice")(history) is False

    def test_all_agree_on_empty(self) -> None:
        history = {"a": []}
        assert all_agree_on("choice")(history) is False

    def test_positions_stable_true(self) -> None:
        pos = SimplePosition(choice="X")
        history = {"a": [pos, pos], "b": [pos, pos]}
        assert positions_stable(2)(history) is True

    def test_positions_stable_false_not_enough_rounds(self) -> None:
        pos = SimplePosition(choice="X")
        history = {"a": [pos]}
        assert positions_stable(2)(history) is False

    def test_positions_stable_false_changed(self) -> None:
        history = {
            "a": [SimplePosition(choice="X"), SimplePosition(choice="Y")],
        }
        assert positions_stable(2)(history) is False

    def test_majority_agree_on_true(self) -> None:
        history = {
            "a": [SimplePosition(choice="X")],
            "b": [SimplePosition(choice="X")],
            "c": [SimplePosition(choice="Y")],
        }
        assert majority_agree_on("choice", threshold=0.6)(history) is True

    def test_majority_agree_on_false(self) -> None:
        history = {
            "a": [SimplePosition(choice="X")],
            "b": [SimplePosition(choice="Y")],
            "c": [SimplePosition(choice="Z")],
        }
        assert majority_agree_on("choice", threshold=0.6)(history) is False

    def test_majority_agree_on_empty(self) -> None:
        assert majority_agree_on("choice")({}) is False

    def test_majority_agree_on_exact_threshold(self) -> None:
        history = {
            "a": [SimplePosition(choice="X")],
            "b": [SimplePosition(choice="X")],
            "c": [SimplePosition(choice="Y")],
            "d": [SimplePosition(choice="Z")],
            "e": [SimplePosition(choice="W")],
        }
        # 2/5 = 0.4 < 0.6
        assert majority_agree_on("choice", threshold=0.6)(history) is False


# ── execute_discussion ─────────────────────────────────────────────────────


class TestExecuteDiscussion:
    @pytest.mark.asyncio
    async def test_single_round_two_agents(self) -> None:
        """单轮两个 Agent，产出正确 DiscussionOutput。"""
        responses = [
            _make_json("I pick AAPL", {"choice": "AAPL"}),
            _make_json("I pick TSLA", {"choice": "TSLA"}),
        ]
        runner = MockRunner(responses=responses)
        a1 = _make_agent("analyst", runner)
        a2 = _make_agent("trader", runner)

        disc = Discussion(
            agents=[a1, a2],
            position_schema=SimplePosition,
            topic="选股",
            max_rounds=1,
        )

        result = await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        output = result.output
        assert isinstance(output, DiscussionOutput)
        assert output.total_turns == 2
        assert output.rounds_completed == 1
        assert "analyst" in output.final_positions
        assert "trader" in output.final_positions
        assert output.final_positions["analyst"].choice == "AAPL"
        assert output.final_positions["trader"].choice == "TSLA"
        assert output.converged is False

    @pytest.mark.asyncio
    async def test_multi_round_position_tracking(self) -> None:
        """多轮立场追踪：position_history 正确累积。"""
        responses = [
            _make_json("round0 a", {"choice": "X"}),
            _make_json("round0 b", {"choice": "Y"}),
            _make_json("round1 a changed", {"choice": "Y"}),
            _make_json("round1 b", {"choice": "Y"}),
        ]
        runner = MockRunner(responses=responses)
        a1 = _make_agent("a", runner)
        a2 = _make_agent("b", runner)

        disc = Discussion(
            agents=[a1, a2],
            position_schema=SimplePosition,
            topic="test",
            max_rounds=2,
        )

        result = await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        output = result.output
        assert output.rounds_completed == 2
        assert len(output.position_history["a"]) == 2
        assert output.position_history["a"][0].choice == "X"
        assert output.position_history["a"][1].choice == "Y"

    @pytest.mark.asyncio
    async def test_convergence_stops_early(self) -> None:
        """收敛检测触发提前终止。"""
        responses = [
            _make_json("r0 a", {"choice": "X"}),
            _make_json("r0 b", {"choice": "Y"}),
            _make_json("r1 a→Y", {"choice": "Y"}),
            _make_json("r1 b", {"choice": "Y"}),
        ]
        runner = MockRunner(responses=responses)
        a1 = _make_agent("a", runner)
        a2 = _make_agent("b", runner)

        disc = Discussion(
            agents=[a1, a2],
            position_schema=SimplePosition,
            topic="test",
            max_rounds=5,
            convergence=all_agree_on("choice"),
        )

        result = await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        output = result.output
        assert output.converged is True
        assert output.convergence_round == 1
        assert output.rounds_completed == 2  # 完成了第 0 轮和第 1 轮
        assert output.total_turns == 4

    @pytest.mark.asyncio
    async def test_until_stops_early(self) -> None:
        """until 回调提前终止。"""
        responses = [
            _make_json("r0 a", {"choice": "X"}),
            _make_json("r0 b", {"choice": "Y"}),
        ]
        runner = MockRunner(responses=responses)
        a1 = _make_agent("a", runner)
        a2 = _make_agent("b", runner)

        call_count = 0

        def until_fn(ctx: DiscussionContext) -> bool:
            nonlocal call_count
            call_count += 1
            return call_count >= 2  # 第二次 until 检查时终止

        disc = Discussion(
            agents=[a1, a2],
            position_schema=SimplePosition,
            topic="test",
            max_rounds=5,
            until=until_fn,
        )

        result = await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        output = result.output
        assert output.total_turns == 2

    @pytest.mark.asyncio
    async def test_custom_prompt_template(self) -> None:
        """自定义 prompt_template。"""
        runner = MockRunner(responses=[
            _make_json("ok", {"choice": "A"}),
        ])
        a1 = _make_agent("a", runner)

        disc = Discussion(
            agents=[a1],
            position_schema=SimplePosition,
            topic="test",
            max_rounds=1,
            prompt_template=lambda agent, ctx: f"Custom for {agent.name}: {ctx.topic}",
        )

        await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        assert runner.calls[0]["prompt"] == "Custom for a: test"

    @pytest.mark.asyncio
    async def test_agent_prompts_override(self) -> None:
        """per-agent prompt override。"""
        runner = MockRunner(responses=[
            _make_json("ok", {"choice": "A"}),
            _make_json("ok", {"choice": "B"}),
        ])
        a1 = _make_agent("a", runner)
        a2 = _make_agent("b", runner)

        disc = Discussion(
            agents=[a1, a2],
            position_schema=SimplePosition,
            topic="test",
            max_rounds=1,
            agent_prompts={"a": lambda ctx: "special prompt for a"},
        )

        await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        assert runner.calls[0]["prompt"] == "special prompt for a"
        # b uses default template
        assert "test" in runner.calls[1]["prompt"]

    @pytest.mark.asyncio
    async def test_json_parse_failure_degradation(self) -> None:
        """JSON 解析失败降级：原文作为 response，不中断讨论。"""
        responses = [
            "not valid json at all",
            _make_json("ok", {"choice": "A"}),
        ]
        runner = MockRunner(responses=responses)
        a1 = _make_agent("a", runner)
        a2 = _make_agent("b", runner)

        disc = Discussion(
            agents=[a1, a2],
            position_schema=SimplePosition,
            topic="test",
            max_rounds=1,
        )

        result = await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        output = result.output
        assert output.total_turns == 2
        # First agent: degraded (raw text as response, no position update)
        assert output.turns[0].response == "not valid json at all"
        assert output.turns[0].position_changed is False
        # Second agent: normal
        assert output.turns[1].position.choice == "A"

    @pytest.mark.asyncio
    async def test_progress_callback(self) -> None:
        """进度回调正常调用。"""
        events: list[DiscussionProgressEvent] = []

        runner = MockRunner(responses=[_make_json("ok", {"choice": "A"})])
        a1 = _make_agent("a", runner)

        disc = Discussion(
            agents=[a1],
            position_schema=SimplePosition,
            topic="test",
            max_rounds=1,
            progress_callback=lambda e: events.append(e),
        )

        await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        assert len(events) == 2
        assert events[0].event == "start"
        assert events[0].agent_name == "a"
        assert events[1].event == "complete"
        assert events[1].content == "ok"

    @pytest.mark.asyncio
    async def test_agent_fallback_harness_runner(self) -> None:
        """Agent 无 runner 时 fallback 到 harness_runner。"""
        runner = MockRunner(responses=[_make_json("ok", {"choice": "A"})])
        a1 = Agent(name="a", system_prompt="sp")  # no runner

        disc = Discussion(
            agents=[a1],
            position_schema=SimplePosition,
            topic="test",
            max_rounds=1,
        )

        result = await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        assert result.output.total_turns == 1

    @pytest.mark.asyncio
    async def test_per_agent_session_isolation(self) -> None:
        """每个 Agent 使用独立 session。"""
        runner = MockRunner(responses=[
            _make_json("r0a", {"choice": "X"}),
            _make_json("r0b", {"choice": "Y"}),
            _make_json("r1a", {"choice": "X"}),
            _make_json("r1b", {"choice": "Y"}),
        ])
        a1 = _make_agent("a", runner)
        a2 = _make_agent("b", runner)

        disc = Discussion(
            agents=[a1, a2],
            position_schema=SimplePosition,
            topic="test",
            max_rounds=2,
        )

        await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        # Round 0: both agents start with None session
        assert runner.calls[0]["session_id"] is None
        assert runner.calls[1]["session_id"] is None
        # Round 1: each agent resumes its own session
        assert runner.calls[2]["session_id"] == "sess-1"  # a's session
        assert runner.calls[3]["session_id"] == "sess-2"  # b's session

    @pytest.mark.asyncio
    async def test_background_callable(self) -> None:
        """background 为 callable 时正确解析。"""
        runner = MockRunner(responses=[_make_json("ok", {"choice": "A"})])
        a1 = _make_agent("a", runner)

        class MyState(State):
            market: str = "涨停20家"

        disc = Discussion(
            agents=[a1],
            position_schema=SimplePosition,
            topic="test",
            background=lambda s: f"行情: {s.market}",
            max_rounds=1,
        )

        await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
            state=MyState(),
        )

        assert "涨停20家" in runner.calls[0]["prompt"]

    @pytest.mark.asyncio
    async def test_position_changed_detection(self) -> None:
        """position_changed 正确检测立场变化。"""
        responses = [
            _make_json("round0", {"choice": "X"}),
            _make_json("round1 same", {"choice": "X"}),
            _make_json("round2 changed", {"choice": "Y"}),
        ]
        runner = MockRunner(responses=responses)
        a1 = _make_agent("a", runner)

        disc = Discussion(
            agents=[a1],
            position_schema=SimplePosition,
            topic="test",
            max_rounds=3,
        )

        result = await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        output = result.output
        assert output.turns[0].position_changed is True   # 首轮：prev=None → changed
        assert output.turns[1].position_changed is False  # 不变
        assert output.turns[2].position_changed is True   # X→Y

    @pytest.mark.asyncio
    async def test_position_schema_required(self) -> None:
        """position_schema 为 None 时报错。"""
        runner = MockRunner()
        a1 = _make_agent("a", runner)
        disc = Discussion(agents=[a1], position_schema=None, max_rounds=1)

        from harness._internal.exceptions import TaskFailedError
        with pytest.raises(TaskFailedError, match="position_schema is required"):
            await execute_discussion(
                disc, 0, [], "run-1",
                harness_system_prompt="",
                harness_runner=runner,
                harness_config=TaskConfig(),
            )

    @pytest.mark.asyncio
    async def test_result_task_type(self) -> None:
        """Result.task_type 为 'discussion'。"""
        runner = MockRunner(responses=[_make_json("ok", {"choice": "A"})])
        a1 = _make_agent("a", runner)
        disc = Discussion(agents=[a1], position_schema=SimplePosition, max_rounds=1)

        result = await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        assert result.task_type == "discussion"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_system_prompt_includes_json_schema(self) -> None:
        """system_prompt 包含 JSON schema 说明。"""
        runner = MockRunner(responses=[_make_json("ok", {"choice": "A"})])
        a1 = _make_agent("a", runner)
        disc = Discussion(agents=[a1], position_schema=SimplePosition, max_rounds=1)

        await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="global sp",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        sp = runner.calls[0]["system_prompt"]
        assert "JSON" in sp
        assert "response" in sp
        assert "position" in sp
        assert "global sp" in sp
        assert "You are a" in sp


# ── Agent.task() ───────────────────────────────────────────────────────────


class TestAgentTask:
    def test_returns_llm_task(self) -> None:
        from harness.tasks.llm import LLMTask
        runner = MockRunner()
        a = Agent(name="a", system_prompt="sp", runner=runner)
        t = a.task("hello")
        assert isinstance(t, LLMTask)
        assert t.prompt == "hello"
        assert t.system_prompt == "sp"
        assert t.runner is runner

    def test_with_output_key(self) -> None:
        a = Agent(name="a", system_prompt="sp", runner=MockRunner())
        t = a.task("hello", output_key="result")
        assert t.output_key == "result"

    def test_with_output_schema(self) -> None:
        a = Agent(name="a", system_prompt="sp", runner=MockRunner())
        t = a.task("hello", output_schema=SimplePosition)
        assert t.output_schema is SimplePosition

    def test_uses_build_system_prompt(self) -> None:
        a = Agent(name="分析师", description="技术分析专家。", runner=MockRunner())
        t = a.task("hello")
        assert "你是分析师。技术分析专家。" in t.system_prompt


# ── DiscussionTurn / DiscussionOutput 数据类 ──────────────────────────────


class TestDataClasses:
    def test_discussion_turn_creation(self) -> None:
        pos = SimplePosition(choice="X")
        turn = DiscussionTurn(round=0, agent_name="a", response="test", position=pos, position_changed=True)
        assert turn.round == 0
        assert turn.agent_name == "a"
        assert turn.response == "test"
        assert turn.position.choice == "X"
        assert turn.position_changed is True

    def test_discussion_output_creation(self) -> None:
        output = DiscussionOutput(
            turns=[],
            rounds_completed=0,
            total_turns=0,
            final_positions={},
            converged=False,
            convergence_round=None,
            position_history={},
        )
        assert output.converged is False
        assert output.convergence_round is None

    def test_discussion_progress_event(self) -> None:
        event = DiscussionProgressEvent(event="start", round=0, agent_name="a")
        assert event.event == "start"
        assert event.content is None

        event2 = DiscussionProgressEvent(event="complete", round=0, agent_name="a", content="done")
        assert event2.content == "done"


# ── _merge_system_prompt ──────────────────────────────────────────────────


class TestMergeSystemPrompt:
    def test_includes_agent_sp(self) -> None:
        agent = _make_agent("analyst")
        schema = _make_turn_schema(SimplePosition)
        sp = _merge_system_prompt(agent, schema, "")
        assert "You are analyst" in sp

    def test_includes_harness_sp(self) -> None:
        agent = _make_agent("analyst")
        schema = _make_turn_schema(SimplePosition)
        sp = _merge_system_prompt(agent, schema, "global instructions")
        assert "global instructions" in sp

    def test_includes_json_schema(self) -> None:
        agent = _make_agent("analyst")
        schema = _make_turn_schema(SimplePosition)
        sp = _merge_system_prompt(agent, schema, "")
        assert "JSON" in sp
        assert "response" in sp

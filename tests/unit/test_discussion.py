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
    _build_extraction_prompt,
    _default_prompt_template,
    _extract_position,
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
    """两阶段 mock runner。

    Phase 1（无 output_schema_json）：返回自然文本。
    Phase 2（有 output_schema_json）：返回 position JSON。
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        extraction_responses: list[str] | None = None,
        default: str = "",
    ) -> None:
        self._responses = list(responses) if responses else []
        self._extraction_responses = list(extraction_responses) if extraction_responses else []
        self._default = default
        self._phase1_count = 0
        self._phase2_count = 0
        self.calls: list[dict] = []

    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "session_id": session_id,
            "kwargs": kwargs,
        })
        if kwargs.get("output_schema_json") is not None:
            # Phase 2: 返回 position JSON
            if self._extraction_responses:
                text = self._extraction_responses[self._phase2_count % len(self._extraction_responses)]
            else:
                text = self._default
            self._phase2_count += 1
            return RunnerResult(text=text, tokens_used=5, session_id=None)
        else:
            # Phase 1: 返回自然文本
            if self._responses:
                text = self._responses[self._phase1_count % len(self._responses)]
            else:
                text = self._default
            self._phase1_count += 1
            return RunnerResult(text=text, tokens_used=10, session_id=f"sess-{self._phase1_count}")


def _pos_json(position: dict) -> str:
    return json.dumps(position)


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


# ── _build_extraction_prompt ─────────────────────────────────────────────


class TestBuildExtractionPrompt:
    def test_includes_response_text(self) -> None:
        prompt = _build_extraction_prompt("我看好AAPL", SimplePosition)
        assert "我看好AAPL" in prompt

    def test_includes_schema(self) -> None:
        prompt = _build_extraction_prompt("some text", SimplePosition)
        assert "choice" in prompt
        assert "JSON" in prompt

    def test_includes_instruction(self) -> None:
        prompt = _build_extraction_prompt("text", SimplePosition)
        assert "提取" in prompt


# ── _extract_position ────────────────────────────────────────────────────


class TestExtractPosition:
    def test_direct_json(self) -> None:
        """直接 JSON 解析成功。"""
        text = '{"choice": "AAPL"}'
        pos = _extract_position(text, SimplePosition)
        assert pos is not None
        assert pos.choice == "AAPL"

    def test_json_code_block(self) -> None:
        """从 ```json 代码块中提取。"""
        text = '这是分析\n```json\n{"choice": "TSLA"}\n```\n结束'
        pos = _extract_position(text, SimplePosition)
        assert pos is not None
        assert pos.choice == "TSLA"

    def test_code_block_no_lang(self) -> None:
        """从无语言标记的代码块中提取。"""
        text = '分析\n```\n{"choice": "GOOG"}\n```'
        pos = _extract_position(text, SimplePosition)
        assert pos is not None
        assert pos.choice == "GOOG"

    def test_embedded_json(self) -> None:
        """从嵌入的 JSON 对象中正则提取。"""
        text = '我的立场是 {"choice": "META"} 就这样'
        pos = _extract_position(text, SimplePosition)
        assert pos is not None
        assert pos.choice == "META"

    def test_returns_none_on_failure(self) -> None:
        """完全无法解析时返回 None。"""
        text = "这是纯文本，没有任何 JSON"
        pos = _extract_position(text, SimplePosition)
        assert pos is None

    def test_invalid_json_returns_none(self) -> None:
        """JSON 结构不匹配 schema 时返回 None。"""
        text = '{"wrong_field": 123}'
        pos = _extract_position(text, SimplePosition)
        assert pos is None

    def test_complex_schema(self) -> None:
        """复杂 schema 的提取。"""
        text = '{"top_pick": "AAPL", "direction": "buy", "confidence": 0.9}'
        pos = _extract_position(text, TradingPosition)
        assert pos is not None
        assert pos.top_pick == "AAPL"
        assert pos.confidence == 0.9


# ── execute_discussion ─────────────────────────────────────────────────────


class TestExecuteDiscussion:
    @pytest.mark.asyncio
    async def test_single_round_two_agents(self) -> None:
        """单轮两个 Agent，产出正确 DiscussionOutput。"""
        runner = MockRunner(
            responses=["I pick AAPL because...", "I pick TSLA because..."],
            extraction_responses=[
                _pos_json({"choice": "AAPL"}),
                _pos_json({"choice": "TSLA"}),
            ],
        )
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
        runner = MockRunner(
            responses=["round0 a", "round0 b", "round1 a changed", "round1 b"],
            extraction_responses=[
                _pos_json({"choice": "X"}),
                _pos_json({"choice": "Y"}),
                _pos_json({"choice": "Y"}),
                _pos_json({"choice": "Y"}),
            ],
        )
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
        runner = MockRunner(
            responses=["r0 a", "r0 b", "r1 a→Y", "r1 b"],
            extraction_responses=[
                _pos_json({"choice": "X"}),
                _pos_json({"choice": "Y"}),
                _pos_json({"choice": "Y"}),
                _pos_json({"choice": "Y"}),
            ],
        )
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
        runner = MockRunner(
            responses=["r0 a", "r0 b"],
            extraction_responses=[
                _pos_json({"choice": "X"}),
                _pos_json({"choice": "Y"}),
            ],
        )
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
        runner = MockRunner(
            responses=["ok"],
            extraction_responses=[_pos_json({"choice": "A"})],
        )
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

        # Phase 1 call uses the custom prompt
        phase1_calls = [c for c in runner.calls if c["kwargs"].get("output_schema_json") is None]
        assert phase1_calls[0]["prompt"] == "Custom for a: test"

    @pytest.mark.asyncio
    async def test_agent_prompts_override(self) -> None:
        """per-agent prompt override。"""
        runner = MockRunner(
            responses=["ok", "ok"],
            extraction_responses=[
                _pos_json({"choice": "A"}),
                _pos_json({"choice": "B"}),
            ],
        )
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

        phase1_calls = [c for c in runner.calls if c["kwargs"].get("output_schema_json") is None]
        assert phase1_calls[0]["prompt"] == "special prompt for a"
        # b uses default template
        assert "test" in phase1_calls[1]["prompt"]

    @pytest.mark.asyncio
    async def test_phase2_extraction_failure_degradation(self) -> None:
        """Phase 2 提取失败降级：保留上一轮立场，不中断讨论。"""
        runner = MockRunner(
            responses=["analysis text", "ok analysis"],
            extraction_responses=[
                "not valid json at all",  # Phase 2 失败
                _pos_json({"choice": "A"}),
            ],
        )
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
        # First agent: degraded (Phase 2 failed, no position update)
        assert output.turns[0].response == "analysis text"
        assert output.turns[0].position_changed is False
        # Second agent: normal
        assert output.turns[1].position.choice == "A"

    @pytest.mark.asyncio
    async def test_progress_callback(self) -> None:
        """进度回调正常调用，包含 phase 事件。"""
        events: list[DiscussionProgressEvent] = []

        runner = MockRunner(
            responses=["ok analysis"],
            extraction_responses=[_pos_json({"choice": "A"})],
        )
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

        event_types = [e.event for e in events]
        assert event_types == ["start", "phase", "phase", "complete"]
        assert events[0].event == "start"
        assert events[0].agent_name == "a"
        # Phase 1 event
        assert events[1].event == "phase"
        assert "Phase 1" in events[1].content
        # Phase 2 event
        assert events[2].event == "phase"
        assert "Phase 2" in events[2].content
        # Complete event
        assert events[3].event == "complete"
        assert events[3].content == "ok analysis"

    @pytest.mark.asyncio
    async def test_agent_fallback_harness_runner(self) -> None:
        """Agent 无 runner 时 fallback 到 harness_runner。"""
        runner = MockRunner(
            responses=["ok"],
            extraction_responses=[_pos_json({"choice": "A"})],
        )
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
        """Phase 1 用 agent session，Phase 2 用 session_id=None。"""
        runner = MockRunner(
            responses=["r0a", "r0b", "r1a", "r1b"],
            extraction_responses=[
                _pos_json({"choice": "X"}),
                _pos_json({"choice": "Y"}),
                _pos_json({"choice": "X"}),
                _pos_json({"choice": "Y"}),
            ],
        )
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

        # Separate Phase 1 and Phase 2 calls
        phase1_calls = [c for c in runner.calls if c["kwargs"].get("output_schema_json") is None]
        phase2_calls = [c for c in runner.calls if c["kwargs"].get("output_schema_json") is not None]

        # Round 0: both agents start with None session (Phase 1)
        assert phase1_calls[0]["session_id"] is None
        assert phase1_calls[1]["session_id"] is None
        # Round 1: each agent resumes its own session (Phase 1)
        assert phase1_calls[2]["session_id"] == "sess-1"  # a's session
        assert phase1_calls[3]["session_id"] == "sess-2"  # b's session
        # All Phase 2 calls use session_id=None
        for call in phase2_calls:
            assert call["session_id"] is None

    @pytest.mark.asyncio
    async def test_background_callable(self) -> None:
        """background 为 callable 时正确解析。"""
        runner = MockRunner(
            responses=["ok"],
            extraction_responses=[_pos_json({"choice": "A"})],
        )
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

        phase1_calls = [c for c in runner.calls if c["kwargs"].get("output_schema_json") is None]
        assert "涨停20家" in phase1_calls[0]["prompt"]

    @pytest.mark.asyncio
    async def test_position_changed_detection(self) -> None:
        """position_changed 正确检测立场变化。"""
        runner = MockRunner(
            responses=["round0", "round1 same", "round2 changed"],
            extraction_responses=[
                _pos_json({"choice": "X"}),
                _pos_json({"choice": "X"}),
                _pos_json({"choice": "Y"}),
            ],
        )
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
        runner = MockRunner(
            responses=["ok"],
            extraction_responses=[_pos_json({"choice": "A"})],
        )
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
    async def test_phase1_no_json_in_system_prompt(self) -> None:
        """Phase 1 的 system_prompt 不包含 JSON 格式说明。"""
        runner = MockRunner(
            responses=["ok"],
            extraction_responses=[_pos_json({"choice": "A"})],
        )
        a1 = _make_agent("a", runner)
        disc = Discussion(agents=[a1], position_schema=SimplePosition, max_rounds=1)

        await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="global sp",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        phase1_calls = [c for c in runner.calls if c["kwargs"].get("output_schema_json") is None]
        sp = phase1_calls[0]["system_prompt"]
        assert "global sp" in sp
        assert "You are a" in sp
        # Phase 1 不包含 JSON schema 指令
        assert "请用 JSON 格式回复" not in sp

    @pytest.mark.asyncio
    async def test_phase2_passes_output_schema(self) -> None:
        """Phase 2 传递 output_schema_json。"""
        runner = MockRunner(
            responses=["ok"],
            extraction_responses=[_pos_json({"choice": "A"})],
        )
        a1 = _make_agent("a", runner)
        disc = Discussion(agents=[a1], position_schema=SimplePosition, max_rounds=1)

        await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        phase2_calls = [c for c in runner.calls if c["kwargs"].get("output_schema_json") is not None]
        assert len(phase2_calls) == 1
        schema = json.loads(phase2_calls[0]["kwargs"]["output_schema_json"])
        assert "choice" in str(schema)

    @pytest.mark.asyncio
    async def test_extraction_runner_used(self) -> None:
        """设置 extraction_runner 时 Phase 2 使用它。"""
        main_runner = MockRunner(
            responses=["analysis text"],
            extraction_responses=[],  # main runner 不应该收到 Phase 2 调用
        )
        extract_runner = MockRunner(
            responses=[],
            extraction_responses=[_pos_json({"choice": "B"})],
        )
        a1 = _make_agent("a", main_runner)

        disc = Discussion(
            agents=[a1],
            position_schema=SimplePosition,
            topic="test",
            max_rounds=1,
            extraction_runner=extract_runner,
        )

        result = await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=main_runner,
            harness_config=TaskConfig(),
        )

        assert result.output.final_positions["a"].choice == "B"
        # Phase 1 went to main_runner
        assert main_runner._phase1_count == 1
        assert main_runner._phase2_count == 0
        # Phase 2 went to extract_runner
        assert extract_runner._phase2_count == 1
        assert extract_runner._phase1_count == 0

    @pytest.mark.asyncio
    async def test_phase2_timeout(self) -> None:
        """Phase 2 有独立超时（不超过 60s），超时后降级。"""

        class SlowExtractRunner(AbstractRunner):
            async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
                if kwargs.get("output_schema_json") is not None:
                    import asyncio
                    await asyncio.sleep(100)  # 超过 60s 超时
                return RunnerResult(text="ok", tokens_used=5, session_id=None)

        main_runner = MockRunner(
            responses=["analysis"],
            extraction_responses=[],
        )
        a1 = _make_agent("a", main_runner)

        disc = Discussion(
            agents=[a1],
            position_schema=SimplePosition,
            topic="test",
            max_rounds=1,
            extraction_runner=SlowExtractRunner(),
        )

        result = await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=main_runner,
            harness_config=TaskConfig(timeout=120),
        )

        # Phase 2 超时降级，但讨论不中断
        output = result.output
        assert output.total_turns == 1
        assert output.turns[0].response == "analysis"
        assert output.turns[0].position_changed is False  # 降级

    @pytest.mark.asyncio
    async def test_tokens_include_both_phases(self) -> None:
        """总 tokens = Phase 1 + Phase 2。"""
        runner = MockRunner(
            responses=["analysis"],
            extraction_responses=[_pos_json({"choice": "A"})],
        )
        a1 = _make_agent("a", runner)
        disc = Discussion(agents=[a1], position_schema=SimplePosition, max_rounds=1)

        result = await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        # Phase 1 = 10 tokens, Phase 2 = 5 tokens
        assert result.tokens_used == 15


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
        sp = _merge_system_prompt(agent, "")
        assert "You are analyst" in sp

    def test_includes_harness_sp(self) -> None:
        agent = _make_agent("analyst")
        sp = _merge_system_prompt(agent, "global instructions")
        assert "global instructions" in sp

    def test_no_json_schema_in_sp(self) -> None:
        agent = _make_agent("analyst")
        sp = _merge_system_prompt(agent, "")
        assert "JSON" not in sp
        assert "response" not in sp


class TestCallableTopic:
    """Discussion.topic 支持 callable 测试（Issue #8）。"""

    @pytest.mark.asyncio
    async def test_callable_topic(self) -> None:
        """topic 为 callable 时正确解析。"""
        runner = MockRunner(
            responses=["分析 MyProject"],
            extraction_responses=[_pos_json({"choice": "A"})],
        )
        a1 = _make_agent("a", runner)

        class MyState(State):
            project_name: str = "MyProject"

        s = MyState()
        disc = Discussion(
            agents=[a1],
            position_schema=SimplePosition,
            topic=lambda state: f"评估 {state.project_name}",
            max_rounds=1,
        )

        result = await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
            state=s,
        )

        output = result.output
        assert isinstance(output, DiscussionOutput)
        assert output.total_turns == 1
        # 验证 prompt 中包含解析后的 topic
        prompt_text = runner.calls[0]["prompt"]
        assert "评估 MyProject" in prompt_text

    @pytest.mark.asyncio
    async def test_str_topic_still_works(self) -> None:
        """topic 为 str 时行为不变。"""
        runner = MockRunner(
            responses=["ok"],
            extraction_responses=[_pos_json({"choice": "B"})],
        )
        a1 = _make_agent("a", runner)

        disc = Discussion(
            agents=[a1],
            position_schema=SimplePosition,
            topic="静态话题",
            max_rounds=1,
        )

        result = await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
        )

        prompt_text = runner.calls[0]["prompt"]
        assert "静态话题" in prompt_text

    @pytest.mark.asyncio
    async def test_callable_topic_no_state(self) -> None:
        """state=None 时 callable topic 解析为空字符串。"""
        runner = MockRunner(
            responses=["ok"],
            extraction_responses=[_pos_json({"choice": "C"})],
        )
        a1 = _make_agent("a", runner)

        disc = Discussion(
            agents=[a1],
            position_schema=SimplePosition,
            topic=lambda state: f"项目: {state.name}",
            max_rounds=1,
        )

        result = await execute_discussion(
            disc, 0, [], "run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=TaskConfig(),
            state=None,
        )

        assert result.output.total_turns == 1
        # 验证 callable topic 在 state=None 时不被调用，prompt 中不含动态内容
        prompt_text = runner.calls[0]["prompt"]
        assert "项目:" not in prompt_text

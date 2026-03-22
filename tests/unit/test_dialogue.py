"""tests/unit/test_dialogue.py — DialogueContext 方法验证。"""

from __future__ import annotations

import pytest

from harness._internal.dialogue import DialogueContext
from harness.task import DialogueTurn


def make_history() -> list[DialogueTurn]:
    return [
        DialogueTurn(round=0, role_name="analyzer", content="分析结论 A"),
        DialogueTurn(round=0, role_name="critic", content="批评意见 B"),
        DialogueTurn(round=1, role_name="analyzer", content="修正结论 C"),
    ]


def make_ctx(round: int = 1, role_name: str = "critic") -> DialogueContext:
    return DialogueContext(
        round=round,
        role_name=role_name,
        background="审查并发模块",
        history=make_history(),
        pipeline_results=[],
    )


class TestDialogueContextLastFrom:
    def test_last_from_returns_most_recent(self) -> None:
        ctx = make_ctx()
        assert ctx.last_from("analyzer") == "修正结论 C"

    def test_last_from_returns_none_when_no_history(self) -> None:
        ctx = make_ctx()
        assert ctx.last_from("nonexistent") is None

    def test_last_from_first_round_returns_none(self) -> None:
        ctx = DialogueContext(
            round=0,
            role_name="analyzer",
            background="",
            history=[],
            pipeline_results=[],
        )
        assert ctx.last_from("critic") is None


class TestDialogueContextAllFrom:
    def test_all_from_returns_all_entries(self) -> None:
        ctx = make_ctx()
        result = ctx.all_from("analyzer")
        assert result == ["分析结论 A", "修正结论 C"]

    def test_all_from_empty_when_no_match(self) -> None:
        ctx = make_ctx()
        assert ctx.all_from("nobody") == []

    def test_all_from_returns_in_order(self) -> None:
        ctx = make_ctx()
        result = ctx.all_from("analyzer")
        assert result[0] == "分析结论 A"
        assert result[1] == "修正结论 C"


# ---- execute_dialogue ----

import asyncio
from unittest.mock import AsyncMock, MagicMock

from harness._internal.dialogue import execute_dialogue
from harness._internal.session import SessionManager
from harness.runners.base import AbstractRunner, RunnerResult
from harness.task import Dialogue, DialogueOutput, Role


class MockRunner(AbstractRunner):
    """每次调用返回固定文本，记录调用次数。"""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.calls: list[dict] = []
        self._responses = responses or []
        self._call_count = 0

    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        self.calls.append({"prompt": prompt, "session_id": session_id})
        text = self._responses[self._call_count] if self._call_count < len(self._responses) else "ok"
        self._call_count += 1
        return RunnerResult(text=text, tokens_used=5, session_id=f"session-{self._call_count}")


class TestExecuteDialogueBasic:
    @pytest.mark.asyncio
    async def test_single_round_two_roles(self) -> None:
        """1 轮 2 角色：共调用 runner 2 次，产出 DialogueOutput。"""
        runner = MockRunner(["分析结论", "批评意见"])
        dialogue = Dialogue(
            roles=[
                Role(name="analyzer", system_prompt="", prompt=lambda ctx: "分析"),
                Role(name="critic", system_prompt="", prompt=lambda ctx: "批评"),
            ],
            max_rounds=1,
        )
        result = await execute_dialogue(
            dialogue=dialogue,
            outer_index=0,
            pipeline_results=[],
            run_id="run-1",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=None,
        )
        assert result.success is True
        assert result.task_type == "dialogue"
        assert isinstance(result.output, DialogueOutput)
        assert result.output.rounds_completed == 1
        assert len(result.output.turns) == 2
        assert result.output.turns[0].role_name == "analyzer"
        assert result.output.turns[1].role_name == "critic"
        assert result.output.final_speaker == "critic"
        assert result.output.final_content == "批评意见"
        assert runner._call_count == 2

    @pytest.mark.asyncio
    async def test_three_rounds(self) -> None:
        """3 轮 2 角色：共调用 6 次。"""
        runner = MockRunner(["a", "b"] * 3)
        dialogue = Dialogue(
            roles=[
                Role(name="a", system_prompt="", prompt=lambda ctx: "prompt"),
                Role(name="b", system_prompt="", prompt=lambda ctx: "prompt"),
            ],
            max_rounds=3,
        )
        result = await execute_dialogue(
            dialogue=dialogue,
            outer_index=1,
            pipeline_results=[],
            run_id="run-2",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=None,
        )
        assert result.output.rounds_completed == 3
        assert len(result.output.turns) == 6
        assert runner._call_count == 6

    @pytest.mark.asyncio
    async def test_until_stops_early(self) -> None:
        """until 在第 1 轮后返回 True，应只跑 1 轮（共 2 次调用）。"""
        runner = MockRunner(["分析", "我同意"] * 5)
        dialogue = Dialogue(
            roles=[
                Role(name="analyzer", system_prompt="", prompt=lambda ctx: "分析"),
                Role(name="critic", system_prompt="", prompt=lambda ctx: "批评"),
            ],
            max_rounds=5,
            until=lambda ctx: "我同意" in (ctx.last_from("critic") or ""),
        )
        result = await execute_dialogue(
            dialogue=dialogue,
            outer_index=0,
            pipeline_results=[],
            run_id="run-3",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=None,
        )
        assert result.output.rounds_completed == 1
        assert runner._call_count == 2

    @pytest.mark.asyncio
    async def test_context_passed_correctly(self) -> None:
        """DialogueContext 中 round、role_name、history 正确传递。"""
        received_contexts: list[DialogueContext] = []

        def capture_prompt(ctx: DialogueContext) -> str:
            received_contexts.append(ctx)
            return "prompt"

        runner = MockRunner(["resp-a", "resp-b"])
        dialogue = Dialogue(
            roles=[
                Role(name="a", system_prompt="", prompt=capture_prompt),
                Role(name="b", system_prompt="", prompt=capture_prompt),
            ],
            max_rounds=1,
        )
        await execute_dialogue(
            dialogue=dialogue,
            outer_index=0,
            pipeline_results=[],
            run_id="run-4",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=None,
        )
        # 第 1 个角色 a：round=0，history 为空
        assert received_contexts[0].round == 0
        assert received_contexts[0].role_name == "a"
        assert received_contexts[0].history == []

        # 第 2 个角色 b：round=0，history 含 a 的发言
        assert received_contexts[1].round == 0
        assert received_contexts[1].role_name == "b"
        assert len(received_contexts[1].history) == 1
        assert received_contexts[1].history[0].role_name == "a"
        assert received_contexts[1].history[0].content == "resp-a"

    @pytest.mark.asyncio
    async def test_task_index_format(self) -> None:
        """task_index 格式为 '{outer}.r{round}.{role_index}'。"""
        # 通过 storage mock 验证 task_index
        saved_indices: list[str] = []

        class StorageMock:
            async def save_task_log(self, run_id, task_index, *args, **kwargs):
                saved_indices.append(task_index)

        runner = MockRunner(["a", "b"])
        dialogue = Dialogue(
            roles=[
                Role(name="analyzer", system_prompt="", prompt=lambda ctx: ""),
                Role(name="critic", system_prompt="", prompt=lambda ctx: ""),
            ],
            max_rounds=1,
        )
        await execute_dialogue(
            dialogue=dialogue,
            outer_index=2,
            pipeline_results=[],
            run_id="run-5",
            harness_system_prompt="",
            harness_runner=runner,
            harness_config=None,
            storage=StorageMock(),
        )
        assert "2.r0.0" in saved_indices
        assert "2.r0.1" in saved_indices

    @pytest.mark.asyncio
    async def test_independent_sessions_per_role(self) -> None:
        """每个角色应使用不同的 session_id。"""
        session_ids_by_role: dict[str, list] = {"analyzer": [], "critic": []}

        class TrackingRunner(AbstractRunner):
            async def execute(self, prompt, *, system_prompt, session_id, **kwargs):
                # 从 prompt 猜 role（测试用）
                role = "analyzer" if "分析" in prompt else "critic"
                session_ids_by_role[role].append(session_id)
                return RunnerResult(text="ok", tokens_used=0, session_id=f"sess-{role}")

        dialogue = Dialogue(
            roles=[
                Role(name="analyzer", system_prompt="", prompt=lambda ctx: "分析"),
                Role(name="critic", system_prompt="", prompt=lambda ctx: "批评"),
            ],
            max_rounds=2,
        )
        await execute_dialogue(
            dialogue=dialogue,
            outer_index=0,
            pipeline_results=[],
            run_id="run-6",
            harness_system_prompt="",
            harness_runner=TrackingRunner(),
            harness_config=None,
        )
        # round 1 时，analyzer 应该 resume 上一轮的 session
        assert session_ids_by_role["analyzer"][1] == "sess-analyzer"
        assert session_ids_by_role["critic"][1] == "sess-critic"

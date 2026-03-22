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

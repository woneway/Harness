"""Research pipeline 集成测试 — 用 MockRunner 验证 pipeline 结构和 lambda 正确性。"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from harness import Harness
from harness.runners.base import AbstractRunner, RunnerResult
from tests.conftest import make_mock_storage, patch_storage

from examples.research.pipeline import (
    _format_full_discussion,
    _sanitize_filename,
    _save_report,
    build_pipeline,
)
from examples.research.schemas.position import ProjectEvaluation
from examples.research.state import ResearchState
from examples.research.tasks.collect_github import _extract_owner_repo
from examples.research.tasks.parse_input import parse_input


# ── Mock Runner ───────────────────────────────────────────────────────────

_MOCK_POSITION = json.dumps({
    "project_name": "edict",
    "score": 8.0,
    "strengths": ["good API", "fast"],
    "weaknesses": ["docs lacking"],
    "risks": ["single maintainer"],
    "best_for": "small projects",
    "recommendation": "推荐",
})

_MOCK_REPORT = "---\ntitle: test\n---\n# Report\nSome content."


class _ResearchMockRunner(AbstractRunner):
    """模拟 runner：区分 Discussion Phase 1/2 和普通 LLMTask。"""

    def __init__(self) -> None:
        self._phase1_count = 0
        self._phase2_count = 0
        self._llm_count = 0
        self.calls: list[dict] = []

    async def execute(
        self, prompt, *, system_prompt, session_id, **kwargs
    ) -> RunnerResult:
        self.calls.append({
            "prompt": prompt[:200],
            "system_prompt": (system_prompt or "")[:100],
            "session_id": session_id,
            "has_schema": kwargs.get("output_schema_json") is not None,
        })

        # Discussion Phase 2: extraction (has output_schema_json)
        if kwargs.get("output_schema_json") is not None:
            self._phase2_count += 1
            return RunnerResult(text=_MOCK_POSITION, tokens_used=5, session_id=None)

        # Discussion Phase 1: free text (agent system_prompt 包含角色关键词)
        if system_prompt and any(
            kw in system_prompt for kw in (
                "技术深潜", "生态观察", "魔鬼代言", "选型顾问",
                "源码", "社区", "挑战", "综合",
            )
        ):
            self._phase1_count += 1
            return RunnerResult(
                text=f"Agent analysis round {self._phase1_count}: 经过调研发现项目质量良好。",
                tokens_used=10,
                session_id=f"disc-{self._phase1_count}",
            )

        # Regular LLMTask (recorder report)
        self._llm_count += 1
        if "报告" in prompt or "记录员" in prompt:
            return RunnerResult(
                text=_MOCK_REPORT, tokens_used=20, session_id=f"llm-{self._llm_count}"
            )
        return RunnerResult(
            text="default mock response", tokens_used=5, session_id=f"llm-{self._llm_count}"
        )


# ── parse_input 单元测试 ──────────────────────────────────────────────────


class TestParseInput:
    def test_project_name(self):
        state = ResearchState(raw_input="crewai")
        result = parse_input(state)
        data = json.loads(result)
        assert data["input_type"] == "project"
        assert state.input_type == "project"
        assert state.target_project == "crewai"

    def test_github_url(self):
        state = ResearchState(raw_input="https://github.com/langchain-ai/langchain")
        result = parse_input(state)
        data = json.loads(result)
        assert data["input_type"] == "project"
        assert state.target_project == "langchain"
        assert state.target_url == "https://github.com/langchain-ai/langchain"

    def test_comparison(self):
        state = ResearchState(raw_input="langchain vs llamaindex vs haystack")
        result = parse_input(state)
        data = json.loads(result)
        assert data["input_type"] == "comparison"
        assert state.target_project == "langchain"
        assert len(state.competitors) == 2

    def test_topic(self):
        state = ResearchState(raw_input="Python AI 编排框架")
        result = parse_input(state)
        data = json.loads(result)
        assert data["input_type"] == "topic"
        assert state.input_type == "topic"


# ── _extract_owner_repo 单元测试 ──────────────────────────────────────────


class TestExtractOwnerRepo:
    def test_full_url(self):
        assert _extract_owner_repo("x", "https://github.com/org/repo") == "org/repo"

    def test_owner_repo_format(self):
        assert _extract_owner_repo("x", "langchain-ai/langchain") == "langchain-ai/langchain"

    def test_empty_url(self):
        assert _extract_owner_repo("x", "") is None

    def test_garbage(self):
        assert _extract_owner_repo("x", "not a url at all") is None


# ── _sanitize_filename 单元测试 ───────────────────────────────────────────


class TestSanitizeFilename:
    def test_simple_name(self):
        assert _sanitize_filename("crewai") == "crewai"

    def test_chinese_name(self):
        result = _sanitize_filename("技术深潜员")
        assert result == "技术深潜员"

    def test_special_chars(self):
        result = _sanitize_filename("foo/bar@baz")
        assert "/" not in result
        assert "@" not in result

    def test_spaces(self):
        result = _sanitize_filename("hello world")
        assert result == "hello-world"

    def test_empty(self):
        assert _sanitize_filename("") == "unknown"


# ── _save_report 文件名测试 ──────────────────────────────────────────────


class TestSaveReport:
    def test_project_filename(self, tmp_path, monkeypatch):
        monkeypatch.setattr("examples.research.pipeline.REPORT_DIR", tmp_path)
        state = ResearchState(target_project="crewai", report="# Report")
        path = _save_report(state)
        assert "crewai-research.md" in path
        assert (tmp_path / "crewai-research.md").exists()

    def test_topic_filename(self, tmp_path, monkeypatch):
        monkeypatch.setattr("examples.research.pipeline.REPORT_DIR", tmp_path)
        state = ResearchState(
            target_project="",
            raw_input="Python AI 编排框架",
            report="# Report",
        )
        path = _save_report(state)
        assert "research.md" in path
        assert "-research.md" in path
        filename = path.split("/")[-1]
        assert not filename.startswith("-")


# ── _format_full_discussion 测试 ─────────────────────────────────────────


class TestFormatFullDiscussion:
    def test_none_discussion(self):
        assert _format_full_discussion(None) == "无讨论数据"

    def test_formats_all_turns(self):
        """验证所有轮次的完整文本都被包含（不截断）。"""
        from harness.tasks.discussion import DiscussionOutput, DiscussionTurn

        long_text = "A" * 2000  # 远超旧的 600 字限制
        turns = [
            DiscussionTurn(
                round=0,
                agent_name="技术深潜员",
                response=long_text,
                position=ProjectEvaluation(
                    project_name="test",
                    score=8.0,
                    strengths=["good"],
                    weaknesses=["bad"],
                    risks=["risky"],
                    best_for="small teams",
                    recommendation="推荐",
                ),
                position_changed=False,
            ),
        ]
        output = DiscussionOutput(
            turns=turns,
            rounds_completed=1,
            total_turns=1,
            final_positions={"技术深潜员": turns[0].position},
            converged=False,
            convergence_round=None,
            position_history={"技术深潜员": [turns[0].position]},
        )
        result = _format_full_discussion(output)
        # 完整文本必须被包含
        assert long_text in result
        assert "技术深潜员" in result
        assert "8.0" in result or "8" in result


# ── Pipeline 集成测试 ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_project_mode(tmp_path):
    """测试 project 模式的完整 pipeline（全 mock，无 LLM/CLI 调用）。"""
    runner = _ResearchMockRunner()
    steps, state = build_pipeline(runner)

    state.raw_input = "edict"

    h = Harness(project_path=str(tmp_path), runner=runner)
    patch_storage(h, make_mock_storage())

    with patch(
        "examples.research.pipeline._save_discussion_diaries",
        lambda state: str(tmp_path / "diaries"),
    ), patch(
        "examples.research.pipeline._save_report",
        lambda state: str(tmp_path / "test-report.md"),
    ):
        pr = await h.pipeline(steps, state=state)

    # 验证 parse_input 正确设置了 state
    assert state.input_type == "project"
    assert state.target_project == "edict"

    # 验证 pipeline 完成
    assert len(pr.results) > 0
    assert all(r.success for r in pr.results)

    # 验证 Discussion 结果
    assert state.discussion is not None
    assert state.discussion.rounds_completed >= 1

    # 验证 report 被生成
    assert state.report != ""


@pytest.mark.asyncio
async def test_pipeline_comparison_mode(tmp_path):
    """测试 comparison 模式（agents 收到对比项目列表）。"""
    runner = _ResearchMockRunner()
    steps, state = build_pipeline(runner)

    state.raw_input = "crewai vs autogen vs langraph"

    h = Harness(project_path=str(tmp_path), runner=runner)
    patch_storage(h, make_mock_storage())

    with patch(
        "examples.research.pipeline._save_discussion_diaries",
        lambda state: str(tmp_path / "diaries"),
    ), patch(
        "examples.research.pipeline._save_report",
        lambda state: str(tmp_path / "test-report.md"),
    ):
        pr = await h.pipeline(steps, state=state)

    # comparison 模式下 competitors 应由 parse_input 填充
    assert state.input_type == "comparison"
    assert state.target_project == "crewai"
    assert len(state.competitors) == 2

    assert all(r.success for r in pr.results)

"""Research pipeline 集成测试 — 用 MockRunner 验证 pipeline 结构和 lambda 正确性。"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from harness import Harness
from harness.runners.base import AbstractRunner, RunnerResult
from tests.conftest import make_mock_storage, patch_storage

from examples.research.pipeline import build_pipeline
from examples.research.schemas.position import ProjectEvaluation
from examples.research.state import ResearchState
from examples.research.tasks.parse_input import parse_input


# ── Mock Runner ───────────────────────────────────────────────────────────

_MOCK_COMPETITORS_JSON = json.dumps([
    {"name": "project-a", "url": "https://github.com/org/project-a", "description": "A tool"},
    {"name": "project-b", "url": "https://github.com/org/project-b", "description": "B tool"},
])

_MOCK_POSITION = json.dumps({
    "project_name": "edict",
    "score": 8.0,
    "strengths": ["good API", "fast"],
    "weaknesses": ["docs lacking"],
    "best_for": "small projects",
    "recommendation": "推荐",
})

_MOCK_QUALITATIVE = "项目A技术架构优秀，社区活跃。项目B文档详尽。"
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

        # Discussion Phase 1: free text (has agent system_prompt)
        if system_prompt and any(
            kw in system_prompt for kw in ("技术分析", "社区评估", "架构师", "技术架构", "社区生态", "选型")
        ):
            self._phase1_count += 1
            return RunnerResult(
                text=f"Agent analysis round {self._phase1_count}",
                tokens_used=10,
                session_id=f"disc-{self._phase1_count}",
            )

        # Regular LLMTask
        self._llm_count += 1
        if "竞品" in prompt or "搜索" in prompt or "JSON" in prompt:
            return RunnerResult(
                text=_MOCK_COMPETITORS_JSON, tokens_used=15, session_id=f"llm-{self._llm_count}"
            )
        if "定性调研" in prompt:
            return RunnerResult(
                text=_MOCK_QUALITATIVE, tokens_used=15, session_id=f"llm-{self._llm_count}"
            )
        if "报告" in prompt:
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


# ── Pipeline 集成测试 ─────────────────────────────────────────────────────


def _mock_collect_github(state):
    """Mock collect_github: 不调用 gh CLI。"""
    state.github_metrics = {
        "edict": {"stars": 100, "forks": 10, "language": "Python", "license": "MIT"},
    }
    return json.dumps(state.github_metrics)


@pytest.mark.asyncio
async def test_pipeline_project_mode(tmp_path):
    """测试 project 模式的完整 pipeline（全 mock，无 LLM/CLI 调用）。"""
    runner = _ResearchMockRunner()
    steps, state = build_pipeline(runner)

    state.raw_input = "edict"

    h = Harness(project_path=str(tmp_path), runner=runner)
    patch_storage(h, make_mock_storage())

    # Mock collect_github 和 _save_report
    with patch(
        "examples.research.pipeline.collect_github", _mock_collect_github
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
    """测试 comparison 模式（跳过 discover 步骤）。"""
    runner = _ResearchMockRunner()
    steps, state = build_pipeline(runner)

    state.raw_input = "crewai vs autogen vs langraph"

    h = Harness(project_path=str(tmp_path), runner=runner)
    patch_storage(h, make_mock_storage())

    with patch(
        "examples.research.pipeline.collect_github", _mock_collect_github
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

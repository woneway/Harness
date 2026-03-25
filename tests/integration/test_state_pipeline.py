"""tests/integration/test_state_pipeline.py — State 模式 pipeline 集成测试。"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from harness import Harness
from harness.state import State
from harness.tasks import FunctionTask, LLMTask, ShellTask, Parallel
from harness.runners.base import RunnerResult
from tests.conftest import make_mock_storage, patch_storage


class _MockRunner:
    """简易 mock runner，返回固定文本。"""
    def __init__(self, text: str = "mock result"):
        self._text = text

    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        return RunnerResult(text=self._text, tokens_used=5, session_id="s1")


class TestStatePipelineBasic:
    @pytest.mark.asyncio
    async def test_state_created_automatically(self) -> None:
        """不传 state 时 pipeline 自动创建 State。"""
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline([
            FunctionTask(fn=lambda results: 42),
        ])
        assert pr.results[0].output == 42

    @pytest.mark.asyncio
    async def test_custom_state_passed(self) -> None:
        """用户传入自定义 State。"""
        class MyState(State):
            count: int = 0

        s = MyState(count=10)
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        def check(state: State):
            return state.count  # type: ignore[attr-defined]

        pr = await h.pipeline(
            [FunctionTask(fn=check)],
            state=s,
        )
        assert pr.results[0].output == 10

    @pytest.mark.asyncio
    async def test_output_key_writes_to_state(self) -> None:
        """output_key 将 result.output 写入 state。"""
        s = State()
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        def produce(results):
            return 42

        def consume(state: State):
            return state.count  # type: ignore[attr-defined]

        pr = await h.pipeline(
            [
                FunctionTask(fn=produce, output_key="count"),
                FunctionTask(fn=consume),
            ],
            state=s,
        )
        assert pr.results[1].output == 42
        assert s.count == 42  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_mixed_v1_v2_callables(self) -> None:
        """混合 v1 和 v2 callable 在同一 pipeline 中正常工作。"""
        s = State()
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        # v1 style
        def step1(results):
            return "hello"

        # v2 style
        def step2(state: State):
            return f"got: {state.greeting}"  # type: ignore[attr-defined]

        pr = await h.pipeline(
            [
                FunctionTask(fn=step1, output_key="greeting"),
                FunctionTask(fn=step2, output_key="result"),
            ],
            state=s,
        )
        assert pr.results[0].output == "hello"
        assert pr.results[1].output == "got: hello"
        assert s.greeting == "hello"  # type: ignore[attr-defined]
        assert s.result == "got: hello"  # type: ignore[attr-defined]


class TestStatePipelineWithLLM:
    @pytest.mark.asyncio
    async def test_llm_output_key(self) -> None:
        """LLMTask 的 output_key 将 runner 输出写入 state。"""
        s = State()
        h = Harness(project_path="/tmp/test", runner=_MockRunner("analysis result"))
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [LLMTask(prompt="分析", output_key="analysis")],
            state=s,
        )
        assert s.analysis == "analysis result"  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_llm_callable_prompt_v2(self) -> None:
        """LLMTask 的 callable prompt 支持 v2 模式。"""
        s = State()
        s._set_output("topic", "AI")
        h = Harness(project_path="/tmp/test", runner=_MockRunner("done"))
        patch_storage(h, make_mock_storage())

        def prompt(state: State) -> str:
            return f"分析 {state.topic}"  # type: ignore[attr-defined]

        pr = await h.pipeline(
            [LLMTask(prompt=prompt)],
            state=s,
        )
        assert pr.results[0].success


class TestStatePipelineWithShell:
    @pytest.mark.asyncio
    async def test_shell_output_key(self) -> None:
        """ShellTask 的 output_key 将输出写入 state。"""
        s = State()
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [ShellTask(cmd="echo hello", output_key="shell_out")],
            state=s,
        )
        assert "hello" in s.shell_out  # type: ignore[attr-defined]


class TestStatePipelineParallel:
    @pytest.mark.asyncio
    async def test_parallel_results_in_state(self) -> None:
        """Parallel 子任务的 results 被写入 state._results。"""
        s = State()
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [
                Parallel(tasks=[
                    FunctionTask(fn=lambda r: "a"),
                    FunctionTask(fn=lambda r: "b"),
                ]),
            ],
            state=s,
        )
        assert len(s._results) == 2

    @pytest.mark.asyncio
    async def test_parallel_output_key(self) -> None:
        """Parallel 子任务的 output_key 写入 state。"""
        s = State()
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [
                Parallel(tasks=[
                    FunctionTask(fn=lambda r: "alpha", output_key="first"),
                    FunctionTask(fn=lambda r: "beta", output_key="second"),
                ]),
            ],
            state=s,
        )
        assert s.first == "alpha"  # type: ignore[attr-defined]
        assert s.second == "beta"  # type: ignore[attr-defined]


class TestStateResultsSync:
    @pytest.mark.asyncio
    async def test_state_results_sync_with_pipeline_results(self) -> None:
        """state._results 与 PipelineResult.results 保持同步。"""
        s = State()
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [
                FunctionTask(fn=lambda r: "a"),
                FunctionTask(fn=lambda r: "b"),
                FunctionTask(fn=lambda r: "c"),
            ],
            state=s,
        )
        assert len(s._results) == len(pr.results) == 3
        for sr, pr_r in zip(s._results, pr.results):
            assert sr.output == pr_r.output

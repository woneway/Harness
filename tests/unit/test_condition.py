"""tests/unit/test_condition.py — Condition 测试。"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from harness import Harness, Condition, State
from harness.tasks import FunctionTask, LLMTask, Parallel, ShellTask
from harness.runners.base import RunnerResult
from tests.conftest import make_mock_storage, patch_storage


class _MockRunner:
    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        return RunnerResult(text="mock", tokens_used=5, session_id="s1")


class TestConditionBasic:
    def test_instantiation(self) -> None:
        c = Condition(
            check=lambda state: True,
            if_true=[FunctionTask(fn=lambda r: "yes")],
            if_false=[FunctionTask(fn=lambda r: "no")],
        )
        assert len(c.if_true) == 1
        assert len(c.if_false) == 1

    def test_default_check_returns_false(self) -> None:
        c = Condition()
        s = State()
        assert c.check(s) is False

    def test_empty_branches(self) -> None:
        c = Condition(check=lambda state: True)
        assert c.if_true == []
        assert c.if_false == []


class TestConditionPipeline:
    @pytest.mark.asyncio
    async def test_true_branch_executes(self) -> None:
        s = State()
        s._set_output("score", 0.9)
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [
                Condition(
                    check=lambda state: state.score > 0.5,
                    if_true=[FunctionTask(fn=lambda r: "high", output_key="result")],
                    if_false=[FunctionTask(fn=lambda r: "low", output_key="result")],
                ),
            ],
            state=s,
        )
        assert s.result == "high"  # type: ignore[attr-defined]
        assert len(pr.results) == 1

    @pytest.mark.asyncio
    async def test_false_branch_executes(self) -> None:
        s = State()
        s._set_output("score", 0.3)
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [
                Condition(
                    check=lambda state: state.score > 0.5,
                    if_true=[FunctionTask(fn=lambda r: "high", output_key="result")],
                    if_false=[FunctionTask(fn=lambda r: "low", output_key="result")],
                ),
            ],
            state=s,
        )
        assert s.result == "low"  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_empty_true_branch(self) -> None:
        """if_true 为空时，不执行任何步骤。"""
        s = State()
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [
                Condition(
                    check=lambda state: True,
                    if_true=[],
                    if_false=[FunctionTask(fn=lambda r: "no")],
                ),
            ],
            state=s,
        )
        assert len(pr.results) == 0

    @pytest.mark.asyncio
    async def test_multi_step_branch(self) -> None:
        """分支内多个步骤依次执行。"""
        s = State()
        s._set_output("mode", "analyze")
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [
                Condition(
                    check=lambda state: state.mode == "analyze",
                    if_true=[
                        FunctionTask(fn=lambda r: "step1", output_key="a"),
                        FunctionTask(fn=lambda r: "step2", output_key="b"),
                    ],
                ),
            ],
            state=s,
        )
        assert s.a == "step1"  # type: ignore[attr-defined]
        assert s.b == "step2"  # type: ignore[attr-defined]
        assert len(pr.results) == 2

    @pytest.mark.asyncio
    async def test_condition_task_index(self) -> None:
        """Condition 子步骤的 task_index 格式正确。"""
        s = State()
        s._set_output("go", True)
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [
                FunctionTask(fn=lambda r: "before"),
                Condition(
                    check=lambda state: state.go,
                    if_true=[
                        FunctionTask(fn=lambda r: "a"),
                        FunctionTask(fn=lambda r: "b"),
                    ],
                ),
            ],
            state=s,
        )
        # 第一个是普通步骤 "0"，Condition 在 index 1
        # if_true 子步骤应为 "1.c0", "1.c1"
        assert pr.results[1].task_index == "1.c0"
        assert pr.results[2].task_index == "1.c1"

    @pytest.mark.asyncio
    async def test_false_branch_task_index(self) -> None:
        """if_false 分支的 task_index 使用 'f' 前缀。"""
        s = State()
        s._set_output("go", False)
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [
                Condition(
                    check=lambda state: state.go,
                    if_true=[FunctionTask(fn=lambda r: "yes")],
                    if_false=[FunctionTask(fn=lambda r: "no")],
                ),
            ],
            state=s,
        )
        assert pr.results[0].task_index == "0.f0"

    @pytest.mark.asyncio
    async def test_condition_with_llm_task(self) -> None:
        """Condition 分支内可以包含 LLMTask。"""
        s = State()
        s._set_output("need_llm", True)
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [
                Condition(
                    check=lambda state: state.need_llm,
                    if_true=[LLMTask(prompt="分析", output_key="analysis")],
                ),
            ],
            state=s,
        )
        assert s.analysis == "mock"  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_condition_with_v2_check(self) -> None:
        """check 函数使用 v2 State API。"""
        class MyState(State):
            score: float = 0.0

        s = MyState(score=0.95)
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [
                Condition(
                    check=lambda state: state.score > 0.8,
                    if_true=[FunctionTask(fn=lambda r: "passed")],
                ),
            ],
            state=s,
        )
        assert pr.results[0].output == "passed"


class TestConditionInParallelBlocked:
    @pytest.mark.asyncio
    async def test_condition_in_parallel_raises(self) -> None:
        """Condition 嵌入 Parallel 应在入口校验时抛错。"""
        from harness._internal.exceptions import InvalidPipelineError

        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        with pytest.raises(InvalidPipelineError, match="Condition/Loop"):
            await h.pipeline([
                Parallel(tasks=[
                    Condition(check=lambda s: True, if_true=[]),  # type: ignore[arg-type]
                ]),
            ])

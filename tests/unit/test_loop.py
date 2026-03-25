"""tests/unit/test_loop.py — Loop 测试。"""

from __future__ import annotations

import pytest

from harness import Harness, Loop, State
from harness.tasks import FunctionTask, LLMTask, Parallel
from harness.runners.base import RunnerResult
from tests.conftest import make_mock_storage, patch_storage


class _MockRunner:
    async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
        return RunnerResult(text="mock", tokens_used=5, session_id="s1")


class TestLoopBasic:
    def test_instantiation(self) -> None:
        loop = Loop(
            body=[FunctionTask(fn=lambda r: "step")],
            until=lambda state: True,
            max_iterations=3,
        )
        assert len(loop.body) == 1
        assert loop.max_iterations == 3

    def test_default_until_returns_true(self) -> None:
        """默认 until 返回 True，意味着只执行一次。"""
        loop = Loop()
        s = State()
        assert loop.until(s) is True

    def test_default_max_iterations(self) -> None:
        loop = Loop()
        assert loop.max_iterations == 5


class TestLoopPipeline:
    @pytest.mark.asyncio
    async def test_loop_terminates_on_until(self) -> None:
        """until 返回 True 时 Loop 终止。"""
        call_count = 0

        def increment(state: State):
            nonlocal call_count
            call_count += 1
            state._set_output("count", call_count)
            return call_count

        s = State()
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [
                Loop(
                    body=[FunctionTask(fn=increment)],
                    until=lambda state: state.count >= 3,
                    max_iterations=10,
                ),
            ],
            state=s,
        )
        assert call_count == 3
        assert len(pr.results) == 3

    @pytest.mark.asyncio
    async def test_loop_respects_max_iterations(self) -> None:
        """即使 until 不返回 True，也会在 max_iterations 后停止。"""
        call_count = 0

        def increment(state: State):
            nonlocal call_count
            call_count += 1
            return call_count

        s = State()
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [
                Loop(
                    body=[FunctionTask(fn=increment)],
                    until=lambda state: False,  # 永不满足
                    max_iterations=4,
                ),
            ],
            state=s,
        )
        assert call_count == 4
        assert len(pr.results) == 4

    @pytest.mark.asyncio
    async def test_loop_single_iteration(self) -> None:
        """until 立即满足时只执行一次。"""
        s = State()
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline(
            [
                Loop(
                    body=[FunctionTask(fn=lambda r: "done", output_key="result")],
                    until=lambda state: True,  # 第一次就满足
                    max_iterations=5,
                ),
            ],
            state=s,
        )
        assert s.result == "done"  # type: ignore[attr-defined]
        assert len(pr.results) == 1

    @pytest.mark.asyncio
    async def test_loop_multi_step_body(self) -> None:
        """body 内多个步骤每次迭代都执行。"""
        s = State()
        s._set_output("iter_count", 0)
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        def count_up(state: State):
            state._set_output("iter_count", state.iter_count + 1)
            return state.iter_count

        def check(state: State):
            return f"iter={state.iter_count}"

        pr = await h.pipeline(
            [
                Loop(
                    body=[
                        FunctionTask(fn=count_up),
                        FunctionTask(fn=check, output_key="status"),
                    ],
                    until=lambda state: state.iter_count >= 2,
                    max_iterations=5,
                ),
            ],
            state=s,
        )
        assert s.iter_count == 2  # type: ignore[attr-defined]
        # 2 iterations × 2 steps = 4 results
        assert len(pr.results) == 4

    @pytest.mark.asyncio
    async def test_loop_task_index_format(self) -> None:
        """Loop 子步骤的 task_index 格式：{outer}.i{iter}.{child}。"""
        s = State()
        s._set_output("n", 0)
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        def inc(state: State):
            state._set_output("n", state.n + 1)
            return state.n

        pr = await h.pipeline(
            [
                Loop(
                    body=[
                        FunctionTask(fn=inc),
                        FunctionTask(fn=lambda r: "b"),
                    ],
                    until=lambda state: state.n >= 2,
                    max_iterations=5,
                ),
            ],
            state=s,
        )
        # Outer index = 0, iteration 0: "0.i0.0", "0.i0.1"
        # Outer index = 0, iteration 1: "0.i1.0", "0.i1.1"
        assert pr.results[0].task_index == "0.i0.0"
        assert pr.results[1].task_index == "0.i0.1"
        assert pr.results[2].task_index == "0.i1.0"
        assert pr.results[3].task_index == "0.i1.1"

    @pytest.mark.asyncio
    async def test_loop_with_llm_task(self) -> None:
        """Loop body 内可以包含 LLMTask。"""
        s = State()
        s._set_output("attempts", 0)
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        def track(state: State):
            state._set_output("attempts", state.attempts + 1)
            return state.attempts

        pr = await h.pipeline(
            [
                Loop(
                    body=[
                        LLMTask(prompt="generate", output_key="draft"),
                        FunctionTask(fn=track),
                    ],
                    until=lambda state: state.attempts >= 2,
                    max_iterations=5,
                ),
            ],
            state=s,
        )
        assert s.draft == "mock"  # type: ignore[attr-defined]
        assert s.attempts == 2  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_loop_state_modification_across_iterations(self) -> None:
        """Loop 内的 state 修改在迭代间可见。"""
        s = State()
        s._set_output("items", [])
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        def append_item(state: State):
            current = list(state.items)
            current.append(f"item{len(current)}")
            state._set_output("items", current)
            return current

        pr = await h.pipeline(
            [
                Loop(
                    body=[FunctionTask(fn=append_item)],
                    until=lambda state: len(state.items) >= 3,
                    max_iterations=5,
                ),
            ],
            state=s,
        )
        assert s.items == ["item0", "item1", "item2"]  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_loop_after_other_steps(self) -> None:
        """Loop 在 pipeline 中其他步骤之后执行。"""
        s = State()
        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        call_count = 0
        def counter(state: State):
            nonlocal call_count
            call_count += 1
            state._set_output("c", call_count)
            return call_count

        pr = await h.pipeline(
            [
                FunctionTask(fn=lambda r: "init", output_key="phase"),
                Loop(
                    body=[FunctionTask(fn=counter)],
                    until=lambda state: state.c >= 2,
                    max_iterations=5,
                ),
            ],
            state=s,
        )
        assert s.phase == "init"  # type: ignore[attr-defined]
        assert s.c == 2  # type: ignore[attr-defined]
        # 1 init + 2 loop = 3 results
        assert len(pr.results) == 3


class TestLoopInParallelBlocked:
    @pytest.mark.asyncio
    async def test_loop_in_parallel_raises(self) -> None:
        """Loop 嵌入 Parallel 应在入口校验时抛错。"""
        from harness._internal.exceptions import InvalidPipelineError

        h = Harness(project_path="/tmp/test", runner=_MockRunner())
        patch_storage(h, make_mock_storage())

        with pytest.raises(InvalidPipelineError, match="Condition/Loop"):
            await h.pipeline([
                Parallel(tasks=[
                    Loop(body=[], until=lambda s: True),  # type: ignore[arg-type]
                ]),
            ])

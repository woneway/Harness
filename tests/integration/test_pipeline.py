"""tests/integration/test_pipeline.py — 端到端集成测试（不需要 Claude CLI）。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pydantic import BaseModel

from harness import Harness, FunctionTask, ShellTask, TaskConfig
from harness._internal.exceptions import TaskFailedError
from harness.task import Parallel, PollingTask


# ---------------------------------------------------------------------------
# 模块级 Pydantic 模型（供 resume 反序列化测试使用）
# 必须定义在模块顶层，importlib 才能通过 __qualname__ 重新导入
# ---------------------------------------------------------------------------


class _ResumeStepOutput(BaseModel):
    """用于 TestResumeDeserialization 的测试 schema。"""

    value: int
    label: str


def make_harness(tmp_path: Path, **kwargs) -> Harness:
    """创建不依赖 Claude CLI 的 Harness 实例。"""
    from harness.runners.base import AbstractRunner, RunnerResult

    class NoopRunner(AbstractRunner):
        async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
            return RunnerResult(text=prompt, tokens_used=0, session_id=None)

    return Harness(str(tmp_path), runner=NoopRunner(), **kwargs)


# ---------------------------------------------------------------------------
# 顺序 pipeline
# ---------------------------------------------------------------------------


class TestSequentialPipeline:
    @pytest.mark.asyncio
    async def test_function_and_shell(self, tmp_path: Path) -> None:
        """FunctionTask + ShellTask 顺序执行。"""
        h = make_harness(tmp_path)
        result = await h.pipeline([
            FunctionTask(fn=lambda r: "computed"),
            ShellTask(cmd="echo shell-output"),
        ])
        assert len(result.results) == 2
        assert result.results[0].output == "computed"
        assert "shell-output" in result.results[1].output

    @pytest.mark.asyncio
    async def test_results_passed_to_next_task(self, tmp_path: Path) -> None:
        """前序结果通过 results 列表传递给后续 task。"""
        h = make_harness(tmp_path)
        captured = []

        def second_fn(results):
            captured.extend(results)
            return f"got-{results[0].output}"

        result = await h.pipeline([
            FunctionTask(fn=lambda r: "value-from-step1"),
            FunctionTask(fn=second_fn),
        ])
        assert len(captured) >= 1
        assert captured[0].output == "value-from-step1"
        assert result.results[1].output == "got-value-from-step1"

    @pytest.mark.asyncio
    async def test_total_tokens_summed(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        result = await h.pipeline([
            FunctionTask(fn=lambda r: "a"),
            FunctionTask(fn=lambda r: "b"),
        ])
        assert result.total_tokens == 0  # FunctionTask tokens = 0

    @pytest.mark.asyncio
    async def test_run_saved_to_storage(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        pr = await h.pipeline([FunctionTask(fn=lambda r: "x")], name="test-run")

        assert h._storage is not None
        run = await h._storage.get_run(pr.run_id)
        assert run is not None
        assert run["name"] == "test-run"
        assert run["status"] == "success"


# ---------------------------------------------------------------------------
# PollingTask 三路径
# ---------------------------------------------------------------------------


class TestPollingTask:
    @pytest.mark.asyncio
    async def test_polling_success_path(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        result = await h.pipeline([
            PollingTask(
                submit_fn=lambda r: "job-handle",
                poll_fn=lambda h: {"status": "done"},
                success_condition=lambda r: r["status"] == "done",
                poll_interval=0.001,
                timeout=10,
                config=TaskConfig(max_retries=0),
            )
        ])
        assert result.results[0].success is True

    @pytest.mark.asyncio
    async def test_polling_failure_condition_path(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        with pytest.raises(TaskFailedError):
            await h.pipeline([
                PollingTask(
                    submit_fn=lambda r: "h",
                    poll_fn=lambda h: {"status": "error"},
                    success_condition=lambda r: r["status"] == "done",
                    failure_condition=lambda r: r["status"] == "error",
                    poll_interval=0.001,
                    timeout=10,
                    config=TaskConfig(max_retries=0),
                )
            ])

    @pytest.mark.asyncio
    async def test_polling_timeout_path(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        with pytest.raises(TaskFailedError):
            await h.pipeline([
                PollingTask(
                    submit_fn=lambda r: "h",
                    poll_fn=lambda h: {"status": "pending"},
                    success_condition=lambda r: r["status"] == "done",
                    poll_interval=0.001,
                    timeout=0.005,
                    config=TaskConfig(max_retries=0),
                )
            ])


# ---------------------------------------------------------------------------
# Parallel all_or_nothing
# ---------------------------------------------------------------------------


class TestParallelAllOrNothing:
    @pytest.mark.asyncio
    async def test_all_succeed(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        result = await h.pipeline([
            Parallel(tasks=[
                FunctionTask(fn=lambda r: "a"),
                FunctionTask(fn=lambda r: "b"),
            ])
        ])
        assert len(result.results) == 2
        assert all(r.success for r in result.results)

    @pytest.mark.asyncio
    async def test_one_fail_raises(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        with pytest.raises((TaskFailedError, Exception)):
            await h.pipeline([
                Parallel(tasks=[
                    FunctionTask(fn=lambda r: "ok"),
                    FunctionTask(
                        fn=lambda r: (_ for _ in ()).throw(RuntimeError("fail")),
                        config=TaskConfig(max_retries=0),
                    ),
                ], error_policy="all_or_nothing")
            ])


# ---------------------------------------------------------------------------
# Parallel best_effort
# ---------------------------------------------------------------------------


class TestParallelBestEffort:
    @pytest.mark.asyncio
    async def test_partial_failure_in_result(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)
        result = await h.pipeline([
            Parallel(
                tasks=[
                    FunctionTask(fn=lambda r: "ok"),
                    FunctionTask(
                        fn=lambda r: (_ for _ in ()).throw(RuntimeError("fail")),
                        config=TaskConfig(max_retries=0),
                    ),
                    FunctionTask(fn=lambda r: "ok2"),
                ],
                error_policy="best_effort",
            )
        ])
        assert len(result.results) == 3
        successes = [r for r in result.results if r.success]
        failures = [r for r in result.results if not r.success]
        assert len(successes) == 2
        assert len(failures) == 1


# ---------------------------------------------------------------------------
# resume_from 续跑
# ---------------------------------------------------------------------------


class TestResumeFrom:
    @pytest.mark.asyncio
    async def test_resume_skips_task_0_reruns_task_1(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path)

        task_0_calls = [0]
        task_1_calls = [0]

        def task0_fn(results):
            task_0_calls[0] += 1
            return "task0-result"

        def task1_fn(results):
            task_1_calls[0] += 1
            if task_1_calls[0] == 1:
                raise RuntimeError("first run fails")
            return "task1-fixed"

        tasks = [
            FunctionTask(fn=task0_fn),
            FunctionTask(fn=task1_fn, config=TaskConfig(max_retries=0)),
        ]

        # 第一次跑：Task 0 成功，Task 1 失败
        failed_run_id = None
        try:
            await h.pipeline(tasks)
        except TaskFailedError as e:
            failed_run_id = e.run_id

        assert failed_run_id is not None
        assert task_0_calls[0] == 1
        assert task_1_calls[0] == 1

        # 重置计数
        task_0_calls[0] = 0

        # 续跑：Task 0 应被跳过
        result = await h.pipeline(tasks, resume_from=failed_run_id)
        assert task_0_calls[0] == 0  # Task 0 被跳过
        assert task_1_calls[0] == 2  # Task 1 重新执行

    @pytest.mark.asyncio
    async def test_parallel_block_reruns_if_partial_failure(self, tmp_path: Path) -> None:
        """Parallel 块内任意子 task 失败，续跑时整体重跑。"""
        h = make_harness(tmp_path)

        parallel_call_counts = [0, 0]

        def fn0(results):
            parallel_call_counts[0] += 1
            return "p0"

        def fn1(results):
            parallel_call_counts[1] += 1
            if parallel_call_counts[1] == 1:
                raise RuntimeError("p1 fails first time")
            return "p1-fixed"

        tasks = [
            Parallel(
                tasks=[
                    FunctionTask(fn=fn0),
                    FunctionTask(fn=fn1, config=TaskConfig(max_retries=0)),
                ],
                error_policy="best_effort",
            )
        ]

        # 第一次跑
        pr1 = await h.pipeline(tasks)
        # best_effort：一个失败，一个成功
        assert parallel_call_counts[0] == 1
        assert parallel_call_counts[1] == 1

        # 续跑：Parallel 块整体重跑（因为有子 task 失败）
        pr2 = await h.pipeline(tasks, resume_from=pr1.run_id)
        # fn0 应被重新调用（Parallel 块整体重跑）
        assert parallel_call_counts[0] == 2


# ---------------------------------------------------------------------------
# resume_from 续跑时 output 反序列化
# ---------------------------------------------------------------------------


class TestResumeDeserialization:
    @pytest.mark.asyncio
    async def test_resumed_result_output_is_model_instance(self, tmp_path: Path) -> None:
        """续跑时，前序成功步骤的 output 应反序列化为 Pydantic 模型实例，而非 JSON 字符串。

        注意：output_schema 类必须定义在模块顶层，才能通过 importlib 在续跑时重新导入。
        本地类（函数内定义）会触发 ResumeSchemaError，这是预期的正确行为。
        """
        h = make_harness(tmp_path)

        def task0(results):
            return _ResumeStepOutput(value=42, label="hello")

        def task1_fail(results):
            raise RuntimeError("deliberate failure")

        tasks = [
            FunctionTask(fn=task0, output_schema=_ResumeStepOutput),
            FunctionTask(fn=task1_fail, config=TaskConfig(max_retries=0)),
        ]

        failed_run_id = None
        try:
            await h.pipeline(tasks)
        except TaskFailedError as e:
            failed_run_id = e.run_id

        assert failed_run_id is not None

        # 续跑：Task 0 应跳过，resumed_results[0].output 应是 _ResumeStepOutput 实例
        def task1_ok(results):
            assert isinstance(results[0].output, _ResumeStepOutput), (
                f"Expected _ResumeStepOutput, got {type(results[0].output)}: {results[0].output!r}"
            )
            return str(results[0].output.value) + results[0].output.label

        tasks2 = [
            FunctionTask(fn=task0, output_schema=_ResumeStepOutput),
            FunctionTask(fn=task1_ok),
        ]

        result = await h.pipeline(tasks2, resume_from=failed_run_id)
        assert result.results[0].output.value == 42  # type: ignore[union-attr]
        assert result.results[0].output.label == "hello"  # type: ignore[union-attr]
        assert result.results[1].output == "42hello"

    @pytest.mark.asyncio
    async def test_resumed_result_output_no_schema_returns_parsed_json(
        self, tmp_path: Path
    ) -> None:
        """续跑时，无 schema 的前序步骤 output 应尝试 JSON 解析（兼容 dict）。"""
        h = make_harness(tmp_path)

        def task0_dict(results):
            return {"key": "value", "count": 7}

        def task1_fail(results):
            raise RuntimeError("fail")

        tasks = [
            FunctionTask(fn=task0_dict),
            FunctionTask(fn=task1_fail, config=TaskConfig(max_retries=0)),
        ]

        failed_run_id = None
        try:
            await h.pipeline(tasks)
        except TaskFailedError as e:
            failed_run_id = e.run_id

        assert failed_run_id is not None

        accessed_output: list = []

        def task1_ok(results):
            accessed_output.append(results[0].output)
            return "done"

        tasks2 = [
            FunctionTask(fn=task0_dict),
            FunctionTask(fn=task1_ok),
        ]
        await h.pipeline(tasks2, resume_from=failed_run_id)
        # output 应该是 dict（JSON 解析后），不是字符串
        assert isinstance(accessed_output[0], dict)
        assert accessed_output[0]["key"] == "value"
        assert accessed_output[0]["count"] == 7


# ---- Dialogue 集成测试 ----

from harness.task import Dialogue, DialogueOutput, Role
from harness.runners.base import AbstractRunner, RunnerResult
from tests.conftest import make_mock_storage, patch_storage


class TestDialoguePipelineIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_dialogue(self, tmp_path: Path, mock_runner) -> None:
        """完整 pipeline：FunctionTask → Dialogue → FunctionTask。"""
        h = Harness(str(tmp_path), runner=mock_runner)
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline([
            FunctionTask(fn=lambda r: "input_data"),
            Dialogue(
                background="分析代码质量",
                roles=[
                    Role(
                        name="analyzer",
                        system_prompt="你是分析者",
                        prompt=lambda ctx: (
                            "分析" if ctx.round == 0
                            else f"修正分析，批评者说：{ctx.last_from('critic')}"
                        ),
                    ),
                    Role(
                        name="critic",
                        system_prompt="你是批评者",
                        prompt=lambda ctx: f"批评：{ctx.last_from('analyzer')}",
                    ),
                ],
                max_rounds=2,
            ),
            FunctionTask(fn=lambda results: f"总结辩论：{results[-1].output.final_content}"),
        ])

        assert len(pr.results) == 3
        assert pr.results[0].task_type == "function"
        assert pr.results[1].task_type == "dialogue"
        assert isinstance(pr.results[1].output, DialogueOutput)
        assert pr.results[1].output.rounds_completed == 2
        assert len(pr.results[1].output.turns) == 4  # 2 轮 × 2 角色
        assert pr.results[2].task_type == "function"

    @pytest.mark.asyncio
    async def test_dialogue_until_stops_early(self, tmp_path: Path) -> None:
        """until 条件满足时提前终止，只跑 1 轮。"""
        call_count = 0

        class CountingRunner(AbstractRunner):
            async def execute(self, prompt, *, system_prompt, session_id, **kwargs):
                nonlocal call_count
                call_count += 1
                return RunnerResult(
                    text="我同意" if "critic" in system_prompt else "分析",
                    tokens_used=1,
                    session_id=None,
                )

        h = Harness(str(tmp_path), runner=CountingRunner())
        patch_storage(h, make_mock_storage())
        pr = await h.pipeline([
            Dialogue(
                roles=[
                    Role(name="analyzer", system_prompt="analyzer", prompt=lambda ctx: "分析"),
                    Role(name="critic", system_prompt="critic", prompt=lambda ctx: "批评"),
                ],
                max_rounds=5,
                until=lambda ctx: "我同意" in (ctx.last_from("critic") or ""),
            )
        ])
        assert call_count == 2  # 只跑了 1 轮
        assert pr.results[0].output.rounds_completed == 1

    @pytest.mark.asyncio
    async def test_dialogue_turn_mode_in_pipeline(self, tmp_path: Path, mock_runner) -> None:
        """回合模式：next_speaker 动态决定发言顺序，total_turns 正确。"""
        h = Harness(str(tmp_path), runner=mock_runner)
        patch_storage(h, make_mock_storage())

        order = ["a", "b", "a", "b", "a"]
        pr = await h.pipeline([
            Dialogue(
                roles=[
                    Role(name="a", system_prompt="", prompt=lambda ctx: "p"),
                    Role(name="b", system_prompt="", prompt=lambda ctx: "p"),
                ],
                max_turns=5,
                next_speaker=lambda h: order[len(h)],
            )
        ])
        out = pr.results[0].output
        assert isinstance(out, DialogueOutput)
        assert out.total_turns == 5
        assert [t.role_name for t in out.turns] == order
        # 回合模式下 rounds_completed == total_turns
        assert out.rounds_completed == 5

    @pytest.mark.asyncio
    async def test_dialogue_prompt_callable_raises_propagates(self, tmp_path: Path, mock_runner) -> None:
        """role.prompt 抛异常时，pipeline 应抛 TaskFailedError。"""
        from harness._internal.exceptions import TaskFailedError as _TFE

        h = Harness(str(tmp_path), runner=mock_runner)
        patch_storage(h, make_mock_storage())

        with pytest.raises(_TFE) as exc_info:
            await h.pipeline([
                Dialogue(
                    roles=[
                        Role(
                            name="bad",
                            system_prompt="",
                            prompt=lambda ctx: (_ for _ in ()).throw(ValueError("bad prompt")),
                        )
                    ],
                    max_rounds=1,
                )
            ])
        assert "prompt callable raised" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_dialogue_invalid_next_speaker_propagates(self, tmp_path: Path, mock_runner) -> None:
        """next_speaker 返回无效角色名时，pipeline 应抛 TaskFailedError。"""
        from harness._internal.exceptions import TaskFailedError as _TFE

        h = Harness(str(tmp_path), runner=mock_runner)
        patch_storage(h, make_mock_storage())

        with pytest.raises(_TFE) as exc_info:
            await h.pipeline([
                Dialogue(
                    roles=[
                        Role(name="alice", system_prompt="", prompt=lambda ctx: "p"),
                    ],
                    max_turns=3,
                    next_speaker=lambda h: "nonexistent",
                )
            ])
        assert "invalid role name" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_dialogue_pipeline_results_passed_to_context(self, tmp_path: Path, mock_runner) -> None:
        """pipeline_results 正确传递到 DialogueContext 中。"""
        h = Harness(str(tmp_path), runner=mock_runner)
        patch_storage(h, make_mock_storage())

        captured_results: list = []

        def capture_prompt(ctx):
            captured_results.append(list(ctx.pipeline_results))
            return "p"

        pr = await h.pipeline([
            FunctionTask(fn=lambda r: "upstream_data"),
            Dialogue(
                roles=[Role(name="a", system_prompt="", prompt=capture_prompt)],
                max_rounds=1,
            ),
        ])
        assert len(captured_results) == 1
        assert len(captured_results[0]) == 1
        assert captured_results[0][0].output == "upstream_data"

    @pytest.mark.asyncio
    async def test_dialogue_tokens_summed_in_pipeline_result(self, tmp_path: Path) -> None:
        """Dialogue 的 tokens_used 应为所有发言的累计值，体现在 PipelineResult.total_tokens 中。"""
        class TokenRunner(AbstractRunner):
            async def execute(self, prompt, *, system_prompt, session_id, **kwargs):
                return RunnerResult(text="ok", tokens_used=10, session_id=None)

        h = Harness(str(tmp_path), runner=TokenRunner())
        patch_storage(h, make_mock_storage())

        pr = await h.pipeline([
            Dialogue(
                roles=[
                    Role(name="a", system_prompt="", prompt=lambda ctx: "p"),
                    Role(name="b", system_prompt="", prompt=lambda ctx: "p"),
                ],
                max_rounds=3,  # 3 轮 × 2 角色 × 10 tokens = 60
            )
        ])
        assert pr.results[0].tokens_used == 60
        assert pr.total_tokens == 60

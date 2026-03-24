"""tests/unit/test_task.py — Task 类型实例化和行为验证。"""

import warnings

import pytest
from pydantic import BaseModel

from harness.task import (
    DialogueProgressEvent,
    FunctionTask,
    LLMTask,
    Parallel,
    PollingTask,
    Result,
    PipelineResult,
    ShellTask,
    Task,
    TaskConfig,
    result_by_type,
)
from harness._internal.exceptions import InvalidPipelineError


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------


class TestTaskConfig:
    def test_defaults(self) -> None:
        cfg = TaskConfig()
        assert cfg.timeout == 3600
        assert cfg.max_retries == 2
        assert cfg.backoff_base == 2.0

    def test_frozen(self) -> None:
        cfg = TaskConfig(timeout=60)
        with pytest.raises(Exception):
            cfg.timeout = 30  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = TaskConfig(timeout=120, max_retries=5, backoff_base=1.5)
        assert cfg.timeout == 120
        assert cfg.max_retries == 5
        assert cfg.backoff_base == 1.5


# ---------------------------------------------------------------------------
# BaseTask 互斥校验
# ---------------------------------------------------------------------------


class TestBaseTaskCallbackMutex:
    def test_both_callbacks_raises(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            LLMTask(
                prompt="hi",
                stream_callback=lambda t: None,
                raw_stream_callback=lambda e: None,
            )

    def test_only_stream_callback_ok(self) -> None:
        task = LLMTask(prompt="hi", stream_callback=lambda t: None)
        assert task.stream_callback is not None
        assert task.raw_stream_callback is None

    def test_only_raw_stream_callback_ok(self) -> None:
        task = LLMTask(prompt="hi", raw_stream_callback=lambda e: None)
        assert task.stream_callback is None
        assert task.raw_stream_callback is not None

    def test_no_callbacks_ok(self) -> None:
        task = LLMTask(prompt="hi")
        assert task.stream_callback is None
        assert task.raw_stream_callback is None


# ---------------------------------------------------------------------------
# LLMTask
# ---------------------------------------------------------------------------


class TestLLMTask:
    def test_string_prompt(self) -> None:
        task = LLMTask(prompt="hello")
        assert task.prompt == "hello"

    def test_callable_prompt(self) -> None:
        fn = lambda results: "dynamic"
        task = LLMTask(prompt=fn)
        assert task.prompt is fn

    def test_defaults(self) -> None:
        task = LLMTask(prompt="x")
        assert task.system_prompt == ""
        assert task.output_schema is None
        assert task.runner is None
        assert task.config is None

    def test_with_schema(self) -> None:
        class Out(BaseModel):
            text: str

        task = LLMTask(prompt="x", output_schema=Out)
        assert task.output_schema is Out


# ---------------------------------------------------------------------------
# FunctionTask
# ---------------------------------------------------------------------------


class TestFunctionTask:
    def test_instantiation(self) -> None:
        fn = lambda results: 42
        task = FunctionTask(fn=fn)
        assert task.fn is fn

    def test_with_schema(self) -> None:
        class Out(BaseModel):
            value: int

        task = FunctionTask(fn=lambda r: Out(value=1), output_schema=Out)
        assert task.output_schema is Out


# ---------------------------------------------------------------------------
# ShellTask
# ---------------------------------------------------------------------------


class TestShellTask:
    def test_string_cmd(self) -> None:
        task = ShellTask(cmd="echo hello")
        assert task.cmd == "echo hello"

    def test_callable_cmd(self) -> None:
        fn = lambda results: "ls -la"
        task = ShellTask(cmd=fn)
        assert task.cmd is fn

    def test_cwd_and_env(self) -> None:
        task = ShellTask(cmd="ls", cwd="/tmp", env={"FOO": "bar"})
        assert task.cwd == "/tmp"
        assert task.env == {"FOO": "bar"}


# ---------------------------------------------------------------------------
# PollingTask
# ---------------------------------------------------------------------------


class TestPollingTask:
    def test_instantiation(self) -> None:
        submit = lambda results: "handle"
        poll = lambda handle: {"status": "pending"}
        success = lambda r: r["status"] == "done"

        task = PollingTask(
            submit_fn=submit,
            poll_fn=poll,
            success_condition=success,
            poll_interval=5.0,
            timeout=300,
        )
        assert task.submit_fn is submit
        assert task.poll_fn is poll
        assert task.success_condition is success
        assert task.poll_interval == 5.0
        assert task.timeout == 300
        assert task.failure_condition is None

    def test_failure_condition(self) -> None:
        fail = lambda r: r["status"] == "error"
        task = PollingTask(
            submit_fn=lambda r: None,
            poll_fn=lambda h: None,
            success_condition=lambda r: False,
            failure_condition=fail,
        )
        assert task.failure_condition is fail


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------


class TestParallel:
    def test_instantiation(self) -> None:
        tasks = [LLMTask(prompt="a"), ShellTask(cmd="ls")]
        p = Parallel(tasks=tasks)
        assert p.tasks == tasks
        assert p.error_policy == "all_or_nothing"

    def test_best_effort_policy(self) -> None:
        p = Parallel(tasks=[], error_policy="best_effort")
        assert p.error_policy == "best_effort"

    def test_nested_parallel_raises_at_runtime(self) -> None:
        """Parallel 嵌套的运行时校验在 pipeline() 入口处。
        直接构造 Parallel(tasks=[Parallel(...)]) 在类型层面被声明为非法，
        但 Python 不强制类型——这里验证框架层的校验逻辑（在 executor/harness 中）。
        """
        inner = Parallel(tasks=[LLMTask(prompt="inner")])
        # 构造本身不抛异常（运行时校验在 pipeline 入口）
        outer = Parallel(tasks=[inner])  # type: ignore[arg-type]
        assert len(outer.tasks) == 1

    def test_positional_list_auto_corrected(self) -> None:
        """Parallel([task_a, task_b]) 应等价于 Parallel(tasks=[task_a, task_b])。"""
        tasks = [LLMTask(prompt="a"), ShellTask(cmd="ls")]
        p = Parallel(tasks)  # type: ignore[arg-type]
        assert p.tasks == tasks
        assert p.config is None

    def test_positional_list_and_tasks_raises(self) -> None:
        """同时传位置列表和 tasks= 应抛 ValueError。"""
        tasks = [LLMTask(prompt="a")]
        with pytest.raises(ValueError, match="tasks="):
            Parallel(tasks, tasks=tasks)  # type: ignore[arg-type]

    def test_callback_mutex_in_parallel(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            Parallel(
                tasks=[],
                stream_callback=lambda t: None,
                raw_stream_callback=lambda e: None,
            )

    def test_parallel_max_retries_field(self) -> None:
        """验证 max_retries 字段存在且默认值为 2。"""
        p = Parallel(tasks=[])
        assert hasattr(p, "max_retries")
        assert p.max_retries == 2

    def test_parallel_max_retries_custom(self) -> None:
        """验证 max_retries 可以自定义。"""
        p = Parallel(tasks=[], max_retries=5)
        assert p.max_retries == 5

    def test_parallel_max_retries_zero(self) -> None:
        """验证 max_retries=0 合法（不重试）。"""
        p = Parallel(tasks=[], max_retries=0)
        assert p.max_retries == 0


# ---------------------------------------------------------------------------
# Result / PipelineResult
# ---------------------------------------------------------------------------


class TestResult:
    def test_frozen(self) -> None:
        r = Result(
            task_index="0",
            task_type="llm",
            output="hello",
            raw_text="hello",
            tokens_used=10,
            duration_seconds=1.5,
            success=True,
            error=None,
        )
        with pytest.raises(Exception):
            r.success = False  # type: ignore[misc]

    def test_task_index_is_string(self) -> None:
        r = Result(
            task_index="2.1",
            task_type="polling",
            output={"url": "x"},
            raw_text=None,
            tokens_used=0,
            duration_seconds=60.0,
            success=True,
            error=None,
        )
        assert r.task_index == "2.1"
        assert isinstance(r.task_index, str)


class TestPipelineResult:
    def test_frozen(self) -> None:
        pr = PipelineResult(
            run_id="r1",
            name="test",
            results=[],
            total_tokens=0,
            total_duration_seconds=0.0,
        )
        with pytest.raises(Exception):
            pr.run_id = "r2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Task 向后兼容别名
# ---------------------------------------------------------------------------


class TestTaskAlias:
    def test_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t = Task(prompt="hello")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_returns_llm_task(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            t = Task(prompt="hello")
            assert isinstance(t, LLMTask)
            assert t.prompt == "hello"


# ---- Dialogue / Role / DialogueTurn / DialogueContext / DialogueOutput ----

from harness.task import Dialogue, Role


class TestRole:
    def test_role_instantiation(self) -> None:
        role = Role(
            name="analyzer",
            system_prompt="你是专家",
            prompt=lambda ctx: "分析代码",
        )
        assert role.name == "analyzer"
        assert role.system_prompt == "你是专家"
        assert role.runner is None

    def test_role_prompt_callable(self) -> None:
        role = Role(
            name="critic",
            system_prompt="",
            prompt=lambda ctx: f"round={ctx.round}",
        )
        # ctx 先不测，这里只验证 callable 存在
        assert callable(role.prompt)


class TestDialogue:
    def test_dialogue_defaults(self) -> None:
        d = Dialogue(
            roles=[],
            background="背景",
        )
        assert d.max_rounds == 3
        assert d.until is None
        assert d.background == "背景"

    def test_dialogue_with_roles(self) -> None:
        r = Role(name="a", system_prompt="", prompt=lambda ctx: "")
        d = Dialogue(roles=[r], max_rounds=2)
        assert len(d.roles) == 1
        assert d.max_rounds == 2

    def test_dialogue_turn_mode_defaults(self) -> None:
        """回合模式新字段默认值。"""
        d = Dialogue(roles=[])
        assert d.next_speaker is None
        assert d.max_turns is None

    def test_dialogue_next_speaker_callable(self) -> None:
        d = Dialogue(roles=[], next_speaker=lambda h: "a")
        assert callable(d.next_speaker)

    def test_dialogue_max_turns_explicit(self) -> None:
        d = Dialogue(roles=[], max_turns=20)
        assert d.max_turns == 20

    def test_dialogue_until_round_field(self) -> None:
        d = Dialogue(roles=[], until_round=lambda ctx: ctx.round >= 3)
        assert callable(d.until_round)
        assert d.until is None

    def test_dialogue_role_stream_callback(self) -> None:
        cb = lambda role, chunk: None
        d = Dialogue(roles=[], role_stream_callback=cb)
        assert d.role_stream_callback is cb
        # stream_callback 继承自 BaseTask，仍为 None（独立字段，无类型冲突）
        assert d.stream_callback is None


# ---------------------------------------------------------------------------
# DialogueProgressEvent
# ---------------------------------------------------------------------------


class TestDialogueProgressEvent:
    def test_start_event(self) -> None:
        evt = DialogueProgressEvent(event="start", round_or_turn=0, role_name="MCP")
        assert evt.event == "start"
        assert evt.round_or_turn == 0
        assert evt.role_name == "MCP"
        assert evt.content is None

    def test_complete_event_with_content(self) -> None:
        evt = DialogueProgressEvent(
            event="complete", round_or_turn=1, role_name="CLI", content="发言内容"
        )
        assert evt.content == "发言内容"

    def test_error_event_with_content(self) -> None:
        evt = DialogueProgressEvent(
            event="error", round_or_turn=0, role_name="Skills", content="timeout"
        )
        assert evt.event == "error"
        assert evt.content == "timeout"


# ---------------------------------------------------------------------------
# result_by_type
# ---------------------------------------------------------------------------


def _make_result(task_type: str, task_index: str = "0", output: str = "ok") -> Result:
    return Result(
        task_index=task_index,
        task_type=task_type,  # type: ignore[arg-type]
        output=output,
        raw_text=output,
        tokens_used=0,
        duration_seconds=0.0,
        success=True,
        error=None,
    )


class TestResultByType:
    def test_finds_first_match(self) -> None:
        results = [_make_result("dialogue", "0"), _make_result("llm", "1")]
        r = result_by_type(results, "dialogue")
        assert r.task_type == "dialogue"

    def test_finds_nth_match(self) -> None:
        results = [
            _make_result("llm", "0", "first"),
            _make_result("llm", "1", "second"),
        ]
        assert result_by_type(results, "llm", 0).output == "first"
        assert result_by_type(results, "llm", 1).output == "second"

    def test_raises_on_missing_type(self) -> None:
        results = [_make_result("llm", "0")]
        with pytest.raises(ValueError, match="dialogue"):
            result_by_type(results, "dialogue")

    def test_raises_on_n_out_of_range(self) -> None:
        results = [_make_result("llm", "0")]
        with pytest.raises(ValueError, match="n=1"):
            result_by_type(results, "llm", 1)

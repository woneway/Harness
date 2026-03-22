# Dialogue 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 Harness pipeline 中支持多角色辩论/批评循环（`Dialogue`），每个角色有独立 Claude session，知道背景、自己说过什么、对方说过什么。

**Architecture:** 新增 `Role`、`Dialogue`、`DialogueTurn`、`DialogueContext`、`DialogueOutput` 数据类；新增 `harness/_internal/dialogue.py` 实现 `execute_dialogue()`；在 `harness/harness.py` 的 pipeline 主循环中添加 `Dialogue` 分支；零破坏性，所有现有测试继续通过。

**Tech Stack:** Python asyncio, dataclasses, pytest-asyncio, Claude CLI `--resume`

**设计文档:** `docs/plans/2026-03-22-dialogue-design.md`

---

## Task 1：数据模型

**Files:**
- Modify: `harness/task.py`（在 `Parallel` 之后追加）
- Test: `tests/unit/test_task.py`（追加）

### Step 1：写失败测试

在 `tests/unit/test_task.py` 末尾追加：

```python
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
```

### Step 2：运行测试，确认失败

```bash
cd /Users/lianwu/ai/projects/Harness
uv run pytest tests/unit/test_task.py::TestRole tests/unit/test_task.py::TestDialogue -v
```

期望：`ImportError: cannot import name 'Dialogue'`

### Step 3：在 `task.py` 中添加数据模型

在 `harness/task.py` 的 `Parallel` 类之后、`PipelineStep` 类型别名之前，追加：

```python
# ---------------------------------------------------------------------------
# Dialogue / Role
# ---------------------------------------------------------------------------


@dataclass
class DialogueTurn:
    """一次角色发言记录。"""

    round: int       # 轮次，从 0 开始
    role_name: str
    content: str


@dataclass
class DialogueOutput:
    """Dialogue 执行结果，作为 Result.output 存储。"""

    turns: list[DialogueTurn]
    rounds_completed: int
    final_speaker: str   # 最后发言的角色名
    final_content: str   # 最后发言的内容


@dataclass
class Role:
    """Dialogue 中的一个参与者。"""

    name: str
    system_prompt: str
    prompt: Callable[["DialogueContext"], str]
    runner: Any | None = None  # None 时继承 Harness 默认 runner


@dataclass
class Dialogue(BaseTask):
    """多角色循环对话，每个角色维护独立 Claude session。

    执行流程：
    1. 按 roles 顺序，每个角色依次发言（一轮）
    2. 整轮结束后检查 until 条件
    3. 达到 max_rounds 或 until 返回 True 时终止
    4. 向 pipeline results 追加单个 Result，output 为 DialogueOutput

    v1 限制：不支持嵌套在 Parallel 内部。
    """

    roles: list[Role] = field(default_factory=list)
    background: str = ""
    max_rounds: int = 3
    until: Callable[["DialogueContext"], bool] | None = None
```

同时在 `PipelineStep` 类型别名中加入 `Dialogue`：

```python
PipelineStep = LLMTask | FunctionTask | ShellTask | PollingTask | Parallel | Dialogue
```

### Step 4：运行测试，确认通过

```bash
uv run pytest tests/unit/test_task.py::TestRole tests/unit/test_task.py::TestDialogue -v
```

期望：PASS

### Step 5：运行全量测试，确认无回归

```bash
uv run pytest tests/ -v --tb=short
```

期望：所有原有测试继续通过

### Step 6：提交

```bash
git add harness/task.py tests/unit/test_task.py
git commit -m "feat: 新增 Dialogue/Role/DialogueTurn/DialogueOutput 数据模型"
```

---

## Task 2：DialogueContext

**Files:**
- Create: `harness/_internal/dialogue.py`
- Test: `tests/unit/test_dialogue.py`（新建）

### Step 1：写失败测试

新建 `tests/unit/test_dialogue.py`：

```python
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
```

### Step 2：运行测试，确认失败

```bash
uv run pytest tests/unit/test_dialogue.py -v
```

期望：`ModuleNotFoundError: No module named 'harness._internal.dialogue'`

### Step 3：创建 `harness/_internal/dialogue.py`，只实现 `DialogueContext`

```python
"""dialogue.py — Dialogue 多角色对话执行逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.task import DialogueTurn, Result


@dataclass
class DialogueContext:
    """每次调用 Role.prompt 时传入的上下文。"""

    round: int                              # 当前轮次（从 0 开始）
    role_name: str                          # 当前发言角色名
    background: str                         # Dialogue.background
    history: list["DialogueTurn"]           # 本次发言前的所有历史
    pipeline_results: list["Result"]        # 上游 pipeline 结果

    def last_from(self, role_name: str) -> str | None:
        """获取指定角色最近一次发言内容，无则返回 None。"""
        for turn in reversed(self.history):
            if turn.role_name == role_name:
                return turn.content
        return None

    def all_from(self, role_name: str) -> list[str]:
        """获取指定角色所有历史发言，按时间顺序。"""
        return [t.content for t in self.history if t.role_name == role_name]
```

### Step 4：运行测试，确认通过

```bash
uv run pytest tests/unit/test_dialogue.py -v
```

期望：全部 PASS

### Step 5：提交

```bash
git add harness/_internal/dialogue.py tests/unit/test_dialogue.py
git commit -m "feat: 实现 DialogueContext 及 last_from/all_from 方法"
```

---

## Task 3：execute_dialogue 核心逻辑

**Files:**
- Modify: `harness/_internal/dialogue.py`（追加 `execute_dialogue`）
- Modify: `tests/unit/test_dialogue.py`（追加测试）

### Step 1：写失败测试

在 `tests/unit/test_dialogue.py` 末尾追加：

```python
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
```

### Step 2：运行测试，确认失败

```bash
uv run pytest tests/unit/test_dialogue.py::TestExecuteDialogueBasic -v
```

期望：`ImportError: cannot import name 'execute_dialogue'`

### Step 3：在 `dialogue.py` 中实现 `execute_dialogue`

在 `harness/_internal/dialogue.py` 末尾追加（`DialogueContext` 之后）：

```python
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.runners.base import AbstractRunner
    from harness.storage.base import StorageProtocol
    from harness.task import DialogueTurn, Result

from harness.task import DialogueOutput, DialogueTurn, Result, Dialogue, TaskConfig


async def execute_dialogue(
    dialogue: Dialogue,
    outer_index: int,
    pipeline_results: list[Result],
    run_id: str,
    *,
    harness_system_prompt: str,
    harness_runner: AbstractRunner,
    harness_config: TaskConfig | None,
    storage: StorageProtocol | None = None,
) -> Result:
    """执行 Dialogue：多角色顺序轮转，每角色独立 session。

    Returns:
        单个 Result，output 为 DialogueOutput，task_type="dialogue"。
    """
    start_time = time.monotonic()

    # 每个角色独立的 session_id（None 表示首次调用，由 Claude CLI 自动生成）
    role_sessions: dict[str, str | None] = {role.name: None for role in dialogue.roles}

    history: list[DialogueTurn] = []

    for round_num in range(dialogue.max_rounds):
        for role_idx, role in enumerate(dialogue.roles):
            task_index = f"{outer_index}.r{round_num}.{role_idx}"

            # 构造 DialogueContext
            ctx = DialogueContext(
                round=round_num,
                role_name=role.name,
                background=dialogue.background,
                history=list(history),         # 快照，本轮发言前的历史
                pipeline_results=list(pipeline_results),
            )

            # 构造 prompt
            prompt_text = role.prompt(ctx)

            # 合并 system_prompt：background + role.system_prompt + harness.system_prompt
            system_parts = []
            if dialogue.background:
                system_parts.append(dialogue.background)
            if role.system_prompt:
                system_parts.append(role.system_prompt)
            if harness_system_prompt:
                system_parts.append(harness_system_prompt)
            merged_system = "\n\n".join(system_parts)

            # 选择 runner
            runner = role.runner or harness_runner

            # 调用 runner
            runner_result = await runner.execute(
                prompt_text,
                system_prompt=merged_system,
                session_id=role_sessions[role.name],
            )

            # 更新该角色的 session_id
            if runner_result.session_id:
                role_sessions[role.name] = runner_result.session_id

            # 记录发言
            turn = DialogueTurn(
                round=round_num,
                role_name=role.name,
                content=runner_result.text,
            )
            history.append(turn)

            # 持久化单次发言
            if storage is not None:
                await storage.save_task_log(
                    run_id,
                    task_index,
                    "dialogue",
                    output=runner_result.text,
                    output_schema_class=None,
                    raw_text=runner_result.text,
                    tokens_used=runner_result.tokens_used,
                    duration_seconds=0.0,
                    success=True,
                    error=None,
                )

        # 整轮结束后检查 until 条件
        if dialogue.until is not None:
            check_ctx = DialogueContext(
                round=round_num,
                role_name=dialogue.roles[-1].name,
                background=dialogue.background,
                history=list(history),
                pipeline_results=list(pipeline_results),
            )
            if dialogue.until(check_ctx):
                rounds_completed = round_num + 1
                break
    else:
        rounds_completed = dialogue.max_rounds

    duration = time.monotonic() - start_time
    final_turn = history[-1]
    output = DialogueOutput(
        turns=history,
        rounds_completed=rounds_completed,
        final_speaker=final_turn.role_name,
        final_content=final_turn.content,
    )

    return Result(
        task_index=str(outer_index),
        task_type="dialogue",
        output=output,
        raw_text=final_turn.content,
        tokens_used=sum(0 for _ in history),  # 累计 tokens 在 storage 层已记录
        duration_seconds=duration,
        success=True,
        error=None,
    )
```

> **注意**：`tokens_used` 在 `Result` 层设为 0，实际 token 消耗已在每个 turn 的 `save_task_log` 中记录。`total_tokens` 由 harness.py 汇总时，需要从 sub-turn logs 累加（Task 5 处理）。

### Step 4：运行测试，确认通过

```bash
uv run pytest tests/unit/test_dialogue.py -v
```

期望：全部 PASS

### Step 5：运行全量测试，确认无回归

```bash
uv run pytest tests/ -v --tb=short
```

### Step 6：提交

```bash
git add harness/_internal/dialogue.py tests/unit/test_dialogue.py
git commit -m "feat: 实现 execute_dialogue 核心逻辑（多角色独立 session + until 终止）"
```

---

## Task 4：接入 harness.py pipeline 主循环

**Files:**
- Modify: `harness/harness.py`
- Test: `tests/unit/test_harness.py`（追加）

### Step 1：写失败测试

在 `tests/unit/test_harness.py` 末尾追加（参考文件中已有的 mock 模式）：

```python
# ---- Dialogue 集成 ----

from harness.task import Dialogue, DialogueOutput, Role


class TestHarnessDialogue:
    @pytest.mark.asyncio
    async def test_pipeline_with_dialogue_produces_result(
        self, tmp_path, mock_runner
    ) -> None:
        """pipeline 中的 Dialogue 产出单个 Result，output 为 DialogueOutput。"""
        h = Harness(
            project_path=str(tmp_path),
            runner=mock_runner,
        )
        dialogue = Dialogue(
            roles=[
                Role(name="a", system_prompt="", prompt=lambda ctx: "prompt a"),
                Role(name="b", system_prompt="", prompt=lambda ctx: "prompt b"),
            ],
            max_rounds=1,
        )
        pr = await h.pipeline([dialogue])
        assert len(pr.results) == 1
        assert pr.results[0].task_type == "dialogue"
        assert isinstance(pr.results[0].output, DialogueOutput)
        assert pr.results[0].output.rounds_completed == 1

    @pytest.mark.asyncio
    async def test_pipeline_dialogue_then_llm_can_access_output(
        self, tmp_path, mock_runner
    ) -> None:
        """Dialogue 之后的 LLMTask 可以通过 results[-1].output 访问对话记录。"""
        received: list = []

        def capture_prompt(results):
            received.append(results[-1].output)
            return "总结"

        h = Harness(project_path=str(tmp_path), runner=mock_runner)
        pr = await h.pipeline([
            Dialogue(
                roles=[
                    Role(name="a", system_prompt="", prompt=lambda ctx: "x"),
                ],
                max_rounds=1,
            ),
            LLMTask(prompt=capture_prompt),
        ])
        assert len(received) == 1
        assert isinstance(received[0], DialogueOutput)
```

> **提示**：查看 `tests/unit/test_harness.py` 中已有的 `mock_runner` fixture 写法，直接复用。

### Step 2：运行测试，确认失败

```bash
uv run pytest tests/unit/test_harness.py::TestHarnessDialogue -v
```

期望：`TypeError: Unknown task type: <class 'harness.task.Dialogue'>`

### Step 3：修改 `harness/harness.py`

**3a. 在文件顶部 import 块中加入：**

```python
# 在 harness.py 第 20 行附近，现有 execute_parallel import 下方添加：
from harness._internal.dialogue import execute_dialogue
```

**3b. 在 task 导入中加入 `Dialogue`（第 28-38 行附近）：**

```python
from harness.task import (
    Dialogue,          # ← 新增
    FunctionTask,
    LLMTask,
    Parallel,
    PipelineResult,
    PipelineStep,
    PollingTask,
    Result,
    ShellTask,
    TaskConfig,
)
```

**3c. 在 pipeline 主循环中，`elif isinstance(task, PollingTask):` 之后、`else: raise TypeError` 之前插入：**

```python
elif isinstance(task, Dialogue):
    r = await execute_dialogue(
        dialogue=task,
        outer_index=outer_index,
        pipeline_results=results,
        run_id=run_id,
        harness_system_prompt=self._system_prompt,
        harness_runner=self._runner,
        harness_config=self._default_config,
        storage=self._storage,
    )
```

确保 `r` 之后的 `save_task_log` 和 `results.append(r)` 正常执行（Dialogue 的 Result 也会走这段通用逻辑）。

### Step 4：运行测试，确认通过

```bash
uv run pytest tests/unit/test_harness.py::TestHarnessDialogue -v
```

### Step 5：运行全量测试

```bash
uv run pytest tests/ -v --tb=short
```

### Step 6：提交

```bash
git add harness/harness.py tests/unit/test_harness.py
git commit -m "feat: 在 pipeline 主循环中接入 Dialogue 任务类型"
```

---

## Task 5：更新公开 API 导出

**Files:**
- Modify: `harness/__init__.py`

### Step 1：修改 `harness/__init__.py`

```python
from harness.task import (
    Dialogue,       # ← 新增
    FunctionTask,
    LLMTask,
    Parallel,
    PipelineResult,
    PollingTask,
    Result,
    Role,           # ← 新增
    ShellTask,
    Task,
    TaskConfig,
)

__all__ = [
    "Harness",
    "LLMTask",
    "FunctionTask",
    "ShellTask",
    "PollingTask",
    "Parallel",
    "Dialogue",     # ← 新增
    "Role",         # ← 新增
    "Task",
    "Result",
    "PipelineResult",
    "TaskConfig",
    "Memory",
]
```

### Step 2：验证导入

```bash
uv run python -c "from harness import Dialogue, Role; print('ok')"
```

期望：`ok`

### Step 3：运行全量测试

```bash
uv run pytest tests/ -v --tb=short
```

### Step 4：提交

```bash
git add harness/__init__.py
git commit -m "feat: 导出 Dialogue 和 Role 到公开 API"
```

---

## Task 6：integration test

**Files:**
- Modify: `tests/integration/test_pipeline.py`（追加）

### Step 1：写集成测试

在 `tests/integration/test_pipeline.py` 末尾追加（参考已有 mock runner 写法）：

```python
# ---- Dialogue 集成测试 ----

from harness.task import Dialogue, DialogueOutput, Role


class TestDialoguePipelineIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_dialogue(self, tmp_path, mock_runner) -> None:
        """完整 pipeline：FunctionTask → Dialogue → LLMTask。"""
        h = Harness(project_path=str(tmp_path), runner=mock_runner)

        collected = []

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
            LLMTask(prompt=lambda results: f"总结辩论：{results[-1].output.final_content}"),
        ])

        assert len(pr.results) == 3
        assert pr.results[0].task_type == "function"
        assert pr.results[1].task_type == "dialogue"
        assert isinstance(pr.results[1].output, DialogueOutput)
        assert pr.results[1].output.rounds_completed == 2
        assert len(pr.results[1].output.turns) == 4  # 2 轮 × 2 角色
        assert pr.results[2].task_type == "llm"

    @pytest.mark.asyncio
    async def test_dialogue_until_stops_early(self, tmp_path) -> None:
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

        h = Harness(project_path=str(tmp_path), runner=CountingRunner())
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
```

### Step 2：运行集成测试

```bash
uv run pytest tests/integration/test_pipeline.py -v --tb=short
```

期望：全部 PASS

### Step 3：运行全量测试 + 覆盖率

```bash
uv run pytest tests/ --cov=harness --cov-report=term-missing -v
```

期望：覆盖率 ≥ 80%，所有测试通过

### Step 4：提交

```bash
git add tests/integration/test_pipeline.py
git commit -m "test: Dialogue 集成测试（full pipeline + until 早停）"
```

---

## Task 7：更新 CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

在 `CLAUDE.md` 的"公开 API"代码块中加入 `Dialogue` 和 `Role`：

```python
from harness import (
    Harness,
    LLMTask, FunctionTask, ShellTask, PollingTask, Parallel,
    Dialogue, Role,         # 多角色辩论循环
    Task,                   # LLMTask 的已废弃别名（v2 移除）
    Result, PipelineResult,
    TaskConfig, Memory,
)
```

在"包结构"中 `_internal/` 下加一行：

```
  _internal/
    ...
    dialogue.py      # Dialogue 多角色执行 + DialogueContext
```

### 提交

```bash
git add CLAUDE.md
git commit -m "docs: 更新 CLAUDE.md 公开 API 和包结构（Dialogue/Role）"
```

---

## 验收标准

```bash
# 全量测试通过
uv run pytest tests/ -v

# 覆盖率 ≥ 80%
uv run pytest tests/ --cov=harness --cov-report=term-missing

# 导入验证
uv run python -c "
from harness import Dialogue, Role
d = Dialogue(
    roles=[Role(name='a', system_prompt='', prompt=lambda ctx: 'hi')],
    max_rounds=1,
)
print('Dialogue:', d)
print('ok')
"
```

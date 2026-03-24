"""Harness Task 类型体系。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskConfig:
    """Task 级别通用配置，与 runner 无关。

    优先级：Task.config > Harness.default_config > TaskConfig 默认值
    """

    timeout: int = 3600
    max_retries: int = 2
    backoff_base: float = 2.0


# ---------------------------------------------------------------------------
# Result / PipelineResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Result:
    """单个 Task 的执行结果。"""

    task_index: str  # 顺序步骤："0","1"; Parallel 子任务："2.0","2.1"
    task_type: Literal["llm", "function", "shell", "polling", "dialogue"]
    output: BaseModel | str | Any
    raw_text: str | None
    tokens_used: int
    duration_seconds: float
    success: bool
    error: str | None


@dataclass(frozen=True)
class PipelineResult:
    """整条 pipeline 的执行结果。"""

    run_id: str
    name: str | None
    results: list[Result]
    total_tokens: int
    total_duration_seconds: float


# ---------------------------------------------------------------------------
# BaseTask
# ---------------------------------------------------------------------------


@dataclass
class DialogueProgressEvent:
    """progress_callback 接收的结构化事件。

    Attributes:
        event:         "start" | "complete" | "error"
        round_or_turn: 当前轮次（从 0 开始）
        role_name:     当前发言角色名
        content:       发言内容（event="complete"）或错误信息（event="error"），
                       event="start" 时为 None
    """

    event: Literal["start", "complete", "error"]
    round_or_turn: int
    role_name: str
    content: str | None = None


@dataclass
class BaseTask:
    """所有 Task 类型的公共基类，用户不直接实例化。

    stream_callback 和 raw_stream_callback 互斥，同时设置时抛 ValueError。
    """

    config: TaskConfig | None = None
    stream_callback: Callable[[str], None] | None = None
    raw_stream_callback: Callable[[dict], None] | None = None

    def __post_init__(self) -> None:
        if self.stream_callback is not None and self.raw_stream_callback is not None:
            raise ValueError(
                "stream_callback and raw_stream_callback are mutually exclusive. "
                "Set only one of them."
            )


# ---------------------------------------------------------------------------
# Task 类型
# ---------------------------------------------------------------------------


@dataclass
class LLMTask(BaseTask):
    """调用语言模型，默认用 ClaudeCliRunner。"""

    prompt: str | Callable[[list[Result]], str] = ""
    system_prompt: str = ""
    output_schema: type[BaseModel] | None = None
    runner: Any | None = None  # AbstractRunner | None，避免循环导入


@dataclass
class FunctionTask(BaseTask):
    """执行纯 Python 函数，框架直接调用，不经过任何 runner。

    output_schema 校验失败时抛 OutputSchemaError，不触发重试。
    """

    fn: Callable[[list[Result]], Any] = field(default=lambda results: None)
    output_schema: type[BaseModel] | None = None


@dataclass
class ShellTask(BaseTask):
    """执行 Shell 命令，框架用 asyncio.create_subprocess_shell 调用。

    .. warning::
        ``cmd`` 支持 ``Callable[[list[Result]], str]``，意味着前序任务（尤其是
        LLMTask）的输出可动态构造 shell 命令，存在**命令注入风险**。若前序为
        LLMTask，建议对输出做严格校验后再传入。
    """

    cmd: str | Callable[[list[Result]], str] = ""
    cwd: str | None = None
    env: dict[str, str] | None = None


@dataclass
class PollingTask(BaseTask):
    """提交异步任务后轮询结果，适用于视频生成、TTS、图像生成等外部 AI API。"""

    submit_fn: Callable[[list[Result]], Any] = field(default=lambda results: None)
    poll_fn: Callable[[Any], Any] = field(default=lambda handle: None)
    success_condition: Callable[[Any], bool] = field(default=lambda r: False)
    failure_condition: Callable[[Any], bool] | None = None
    poll_interval: float = 10.0
    timeout: int = 900
    output_schema: type[BaseModel] | None = None


@dataclass
class Parallel(BaseTask):
    """并发执行一组 Task（asyncio.gather），全部成功后继续。

    v1 不支持嵌套 Parallel，框架在 pipeline() 入口做运行时校验。

    支持两种等价写法：
        Parallel(tasks=[task_a, task_b])   # 推荐，明确
        Parallel([task_a, task_b])          # 便捷，自动修正

    Attributes:
        max_retries: 整块 Parallel 的最大重试次数（all_or_nothing 策略生效）。
            默认值 2，与 TaskConfig.max_retries 保持一致。0 表示不重试。
    """

    tasks: list[LLMTask | FunctionTask | ShellTask | PollingTask] = field(
        default_factory=list
    )
    error_policy: Literal["all_or_nothing", "best_effort"] = "all_or_nothing"
    max_retries: int = 2

    def __post_init__(self) -> None:
        # 用户写 Parallel([task_a, task_b]) 时，list 会落到继承自 BaseTask 的
        # config 字段（dataclass 字段顺序：config, ..., tasks）。自动修正。
        if isinstance(self.config, list):
            if self.tasks:
                raise ValueError(
                    "Parallel 收到了位置参数列表（config=list）和关键字参数 tasks= 同时存在，"
                    "请只使用 Parallel(tasks=[...])。"
                )
            self.tasks = self.config  # type: ignore[assignment]
            self.config = None
        super().__post_init__()


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
    """Dialogue 执行结果，作为 Result.output 存储。

    Attributes:
        turns: 所有发言记录，按时间顺序。
        rounds_completed: 轮次模式下表示已完成（含部分）轮数；
            回合模式下与 total_turns 相同（无"轮"的概念）。
        total_turns: 所有模式下的实际发言总次数（len(turns)）。
        final_speaker: 最后发言的角色名。
        final_content: 最后发言的内容。
    """

    turns: list[DialogueTurn]
    rounds_completed: int
    total_turns: int     # 实际发言总次数，语义明确，不依赖模式
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
    """多角色对话，支持两种模式：

    **轮次模式**（默认，next_speaker=None）：
        按 roles 顺序每轮各发言一次，until 在每次发言后检查。
        适合：专家小组各自陈述、每人都要发言一次的场景。

    **回合模式**（设置 next_speaker）：
        由 next_speaker(history) 动态决定每次谁发言，
        until 在每次发言后检查，支持点名回应、随机发言顺序等。
        适合：真正的辩论、角色互相点名、动态参与的场景。

    v1 限制：不支持嵌套在 Parallel 内部。
    """

    roles: list[Role] = field(default_factory=list)
    background: str = ""
    max_rounds: int = 3
    until: Callable[["DialogueContext"], bool] | None = None
    # 轮次模式专用：每轮所有角色发言完毕后检查，比 until 更直观。
    # until_round(ctx) 返回 True 时结束，此时 ctx.role_name 为最后一个角色名。
    until_round: Callable[["DialogueContext"], bool] | None = None

    # 回合模式专用：设置后启用动态发言顺序
    next_speaker: Callable[[list["DialogueTurn"]], str] | None = None
    # 回合模式最大发言次数；None 时默认 max_rounds × len(roles)
    max_turns: int | None = None

    # 进度回调：每次发言开始/结束时调用，用于进度显示。
    # 签名：(event: DialogueProgressEvent) → None
    progress_callback: Callable[["DialogueProgressEvent"], None] | None = None
    # 多角色 streaming 回调：实时接收 runner 输出的文本片段，携带角色名。
    # 签名：(role_name: str, chunk: str)
    # 注：不覆盖 BaseTask.stream_callback（Callable[[str], None]），两者语义不同。
    role_stream_callback: Callable[[str, str], None] | None = None


def result_by_type(results: "list[Result]", task_type: str, n: int = 0) -> "Result":
    """从 pipeline results 按 task_type 取第 n 个结果（默认第 0 个）。

    在 FunctionTask.fn 中替代脆弱的整数下标访问，当 pipeline 顺序变化时报错更明确。

    Args:
        results:   FunctionTask.fn 接收的 list[Result]。
        task_type: "llm" | "function" | "shell" | "polling" | "dialogue"
        n:         第 n 个匹配（从 0 开始），默认 0。

    Raises:
        ValueError: 没有找到匹配的 task_type 或 n 超出范围。
    """
    matches = [r for r in results if r.task_type == task_type]
    if not matches:
        raise ValueError(
            f"No result with task_type={task_type!r}. "
            f"Available types: {[r.task_type for r in results]}"
        )
    if n >= len(matches):
        raise ValueError(
            f"result_by_type: n={n} out of range for task_type={task_type!r} "
            f"(found {len(matches)} match(es))"
        )
    return matches[n]


# Forward reference placeholder for type checking only
if TYPE_CHECKING:
    from harness._internal.dialogue import DialogueContext


# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

PipelineStep = LLMTask | FunctionTask | ShellTask | PollingTask | Parallel | Dialogue


# ---------------------------------------------------------------------------
# 向后兼容
# ---------------------------------------------------------------------------


def Task(*args: Any, **kwargs: Any) -> LLMTask:
    """LLMTask 的已废弃别名，v2 将移除。"""
    warnings.warn(
        "Task is deprecated, use LLMTask instead. Task will be removed in v2.",
        DeprecationWarning,
        stacklevel=2,
    )
    return LLMTask(*args, **kwargs)

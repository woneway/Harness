"""Harness Task 类型体系。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

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
    task_type: Literal["llm", "function", "shell", "polling"]
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
    """

    tasks: list[LLMTask | FunctionTask | ShellTask | PollingTask] = field(
        default_factory=list
    )
    error_policy: Literal["all_or_nothing", "best_effort"] = "all_or_nothing"

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
# 类型别名
# ---------------------------------------------------------------------------

PipelineStep = LLMTask | FunctionTask | ShellTask | PollingTask | Parallel


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

"""Parallel — 并发执行任务组。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from harness.tasks.base import BaseTask
from harness.tasks.function import FunctionTask
from harness.tasks.llm import LLMTask
from harness.tasks.polling import PollingTask
from harness.tasks.shell import ShellTask


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

"""TaskConfig — Task 级别通用配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    """Task 级别通用配置，与 runner 无关。

    优先级：Task.config > Harness.default_config > TaskConfig 默认值
    """

    timeout: int = 3600
    max_retries: int = 2
    backoff_base: float = 2.0

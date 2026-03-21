"""AbstractRunner 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RunnerResult:
    """Runner 执行结果。"""

    text: str
    tokens_used: int
    session_id: str | None


class AbstractRunner(ABC):
    """所有 runner 的抽象基类。"""

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        *,
        system_prompt: str,
        session_id: str | None,
        **kwargs: object,
    ) -> RunnerResult:
        """执行 prompt 并返回结果。

        Args:
            prompt: 用户 prompt。
            system_prompt: 系统 prompt。
            session_id: 会话 ID，用于 Claude Code session 复用。
            **kwargs: runner 专用参数。

        Returns:
            RunnerResult 实例。
        """

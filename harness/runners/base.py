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
            **kwargs: runner 专用参数，常见的包括：
                env_overrides (dict[str, str] | None): 环境变量覆写，
                    由 Harness 传入。空字符串值表示删除该变量。
                    ClaudeCliRunner 将其应用到子进程环境；
                    REST API runner 可忽略（无子进程）。
                output_schema_json (str | None): JSON Schema 字符串，
                    用于结构化输出。
                stream_callback (Callable[[str], None] | None): 流式文本回调。
                raw_stream_callback (Callable[[dict], None] | None): 原始流事件回调。

        Returns:
            RunnerResult 实例。
        """

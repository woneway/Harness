"""AgentLeader — ClaudeCliRunner 的封装，用于控制 agent 可见性。

.. warning::
    这是 **best-effort 约束，非安全边界**。Claude Code 仍然可以调用
    白名单之外的 agent；此约束仅通过 system_prompt 软提示实现。
    v2 计划通过隔离 HOME 目录实现硬约束（待 Claude Code 支持后替换）。
"""

from __future__ import annotations

from harness.runners.base import AbstractRunner, RunnerResult
from harness.runners.claude_cli import ClaudeCliRunner


class AgentLeader(AbstractRunner):
    """限制 Claude Code 可调用的 agent 白名单。

    Args:
        agents: 允许调用的 agent 名称列表。
        runner: 底层 runner，None 时内部创建 ClaudeCliRunner()。
    """

    def __init__(
        self,
        agents: list[str],
        runner: ClaudeCliRunner | None = None,
    ) -> None:
        self.agents = agents
        self._runner = runner or ClaudeCliRunner()

    async def execute(
        self,
        prompt: str,
        *,
        system_prompt: str,
        session_id: str | None,
        **kwargs: object,
    ) -> RunnerResult:
        """在 system_prompt 末尾追加 agent 白名单约束后执行。"""
        constraint = (
            f"\n\n可用 agent 白名单：{', '.join(self.agents)}。"
            "不要调用列表之外的 agent。"
        )
        return await self._runner.execute(
            prompt,
            system_prompt=system_prompt + constraint,
            session_id=session_id,
            **kwargs,
        )

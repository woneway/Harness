"""Agent — class-based 持久化角色，对齐 CrewAI/AutoGen/ADK 主流范式。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from harness.runners.base import AbstractRunner

if TYPE_CHECKING:
    from pydantic import BaseModel

    from harness._internal.dialogue import DialogueContext
    from harness.state import State
    from harness.tasks.dialogue import Role


@dataclass
class Agent:
    """持久化角色定义。

    与 Role 的关系：
    - Role = name + system_prompt + prompt callable（轻量，仅 Dialogue 用）
    - Agent = Role 的超集，可独立执行、有 runner、有 tools 预留

    三种用法：
    1. 独立执行：``await agent.run("分析走势")``
    2. 降级为 Role：``agent.as_role(lambda ctx: "...")``
    3. 在 pipeline 中：``FunctionTask(fn=lambda s: agent.run(...))``
    """

    name: str
    system_prompt: str
    runner: AbstractRunner | None = None

    # 预留字段，本次不实现执行逻辑
    tools: list[Any] = field(default_factory=list)

    async def run(
        self,
        prompt: str | Callable[["State"], str],
        *,
        state: "State | None" = None,
        output_schema: "type[BaseModel] | None" = None,
    ) -> str:
        """单次执行：system_prompt + prompt → runner → 文本结果。

        Args:
            prompt: 用户 prompt（str 或接受 State 的 callable）。
            state: 可选 State，传给 prompt callable。
            output_schema: 结构化输出 schema（透传给 runner）。

        Returns:
            LLM 输出文本。

        Raises:
            ValueError: runner 未设置，或 prompt 为 callable 但 state 为 None。
        """
        if self.runner is None:
            raise ValueError(
                f"Agent '{self.name}' has no runner. "
                "Set agent.runner or pass runner= to constructor."
            )

        # 解析 prompt
        if callable(prompt):
            if state is None:
                raise ValueError(
                    "prompt is callable but state is None. "
                    "Pass state= when using callable prompt."
                )
            prompt_text = prompt(state)
        else:
            prompt_text = prompt

        # 构建 runner kwargs
        kwargs: dict[str, object] = {}
        if output_schema is not None:
            kwargs["output_schema_json"] = output_schema.model_json_schema()

        result = await self.runner.execute(
            prompt_text,
            system_prompt=self.system_prompt,
            session_id=None,
            **kwargs,
        )
        return result.text

    def as_role(self, prompt: Callable[["DialogueContext"], str]) -> "Role":
        """降级为 Dialogue Role。

        Agent 的 system_prompt 和 runner 透传给 Role，
        用户只需提供 prompt callable。
        """
        from harness.tasks.dialogue import Role

        return Role(
            name=self.name,
            system_prompt=self.system_prompt,
            prompt=prompt,
            runner=self.runner,
        )

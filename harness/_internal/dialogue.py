"""dialogue.py — Dialogue 多角色对话执行逻辑。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.runners.base import AbstractRunner
    from harness.storage.base import StorageProtocol
    from harness.task import Result

from harness.task import Dialogue, DialogueOutput, DialogueTurn, TaskConfig


@dataclass
class DialogueContext:
    """每次调用 Role.prompt 时传入的上下文。"""

    round: int                              # 当前轮次（从 0 开始）
    role_name: str                          # 当前发言角色名
    background: str                         # Dialogue.background
    history: list[DialogueTurn]             # 本次发言前的所有历史
    pipeline_results: list[Result]          # 上游 pipeline 结果

    def last_from(self, role_name: str) -> str | None:
        """获取指定角色最近一次发言内容，无则返回 None。"""
        for turn in reversed(self.history):
            if turn.role_name == role_name:
                return turn.content
        return None

    def all_from(self, role_name: str) -> list[str]:
        """获取指定角色所有历史发言，按时间顺序。"""
        return [t.content for t in self.history if t.role_name == role_name]


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
    from harness.task import Result

    start_time = time.monotonic()

    # 每个角色独立的 session_id（None 表示首次调用，由 Claude CLI 自动生成）
    role_sessions: dict[str, str | None] = {role.name: None for role in dialogue.roles}

    history: list[DialogueTurn] = []

    rounds_completed = 0
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
        rounds_completed = round_num + 1
        if dialogue.until is not None:
            check_ctx = DialogueContext(
                round=round_num,
                role_name=dialogue.roles[-1].name,
                background=dialogue.background,
                history=list(history),
                pipeline_results=list(pipeline_results),
            )
            if dialogue.until(check_ctx):
                break

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
        tokens_used=0,  # 累计 tokens 在 storage 层已记录
        duration_seconds=duration,
        success=True,
        error=None,
    )

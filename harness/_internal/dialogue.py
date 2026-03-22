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


async def _execute_turn(
    role_name: str,
    task_index: str,
    round_or_turn: int,
    dialogue: "Dialogue",
    history: list["DialogueTurn"],
    pipeline_results: list["Result"],
    role_sessions: dict[str, str | None],
    harness_system_prompt: str,
    harness_runner: "AbstractRunner",
    storage: "StorageProtocol | None",
    run_id: str,
) -> "DialogueTurn":
    """执行单次角色发言，更新 session，持久化，返回 DialogueTurn。"""
    role_map = {r.name: r for r in dialogue.roles}
    role = role_map[role_name]

    ctx = DialogueContext(
        round=round_or_turn,
        role_name=role_name,
        background=dialogue.background,
        history=list(history),
        pipeline_results=list(pipeline_results),
    )
    prompt_text = role.prompt(ctx)

    system_parts = []
    if dialogue.background:
        system_parts.append(dialogue.background)
    if role.system_prompt:
        system_parts.append(role.system_prompt)
    if harness_system_prompt:
        system_parts.append(harness_system_prompt)
    merged_system = "\n\n".join(system_parts)

    runner = role.runner or harness_runner
    runner_result = await runner.execute(
        prompt_text,
        system_prompt=merged_system,
        session_id=role_sessions[role_name],
    )
    if runner_result.session_id:
        role_sessions[role_name] = runner_result.session_id

    turn = DialogueTurn(round=round_or_turn, role_name=role_name, content=runner_result.text)

    if storage is not None:
        await storage.save_task_log(
            run_id, task_index, "dialogue",
            output=runner_result.text,
            output_schema_class=None,
            raw_text=runner_result.text,
            tokens_used=runner_result.tokens_used,
            duration_seconds=0.0,
            success=True,
            error=None,
        )
    return turn


async def execute_dialogue(
    dialogue: "Dialogue",
    outer_index: int,
    pipeline_results: list["Result"],
    run_id: str,
    *,
    harness_system_prompt: str,
    harness_runner: "AbstractRunner",
    harness_config: "TaskConfig | None",
    storage: "StorageProtocol | None" = None,
) -> "Result":
    """执行 Dialogue，支持轮次模式和回合模式。

    轮次模式（next_speaker=None）：
        所有角色按顺序轮流，until 在每次发言后检查。

    回合模式（next_speaker 已设置）：
        next_speaker(history) 决定每次谁发言，until 在每次发言后检查。
    """
    from harness.task import Result

    start_time = time.monotonic()
    role_sessions: dict[str, str | None] = {role.name: None for role in dialogue.roles}
    history: list[DialogueTurn] = []
    rounds_completed = 0

    if dialogue.next_speaker is None:
        # ── 轮次模式 ──
        done = False
        for round_num in range(dialogue.max_rounds):
            for role_idx, role in enumerate(dialogue.roles):
                task_index = f"{outer_index}.r{round_num}.{role_idx}"
                turn = await _execute_turn(
                    role.name, task_index, round_num,
                    dialogue, history, pipeline_results,
                    role_sessions, harness_system_prompt, harness_runner,
                    storage, run_id,
                )
                history.append(turn)

                # until 在每次发言后立即检查
                if dialogue.until is not None:
                    check_ctx = DialogueContext(
                        round=round_num,
                        role_name=role.name,
                        background=dialogue.background,
                        history=list(history),
                        pipeline_results=list(pipeline_results),
                    )
                    if dialogue.until(check_ctx):
                        done = True
                        break

            rounds_completed = round_num + 1
            if done:
                break
    else:
        # ── 回合模式 ──
        max_turns = dialogue.max_turns or (dialogue.max_rounds * len(dialogue.roles))
        for turn_num in range(max_turns):
            next_role_name = dialogue.next_speaker(list(history))
            task_index = f"{outer_index}.t{turn_num}"
            turn = await _execute_turn(
                next_role_name, task_index, turn_num,
                dialogue, history, pipeline_results,
                role_sessions, harness_system_prompt, harness_runner,
                storage, run_id,
            )
            history.append(turn)
            rounds_completed = turn_num + 1

            # until 在每次发言后检查
            if dialogue.until is not None:
                check_ctx = DialogueContext(
                    round=turn_num,
                    role_name=next_role_name,
                    background=dialogue.background,
                    history=list(history),
                    pipeline_results=list(pipeline_results),
                )
                if dialogue.until(check_ctx):
                    break

    duration = time.monotonic() - start_time
    final_turn = history[-1]
    return Result(
        task_index=str(outer_index),
        task_type="dialogue",
        output=DialogueOutput(
            turns=history,
            rounds_completed=rounds_completed,
            final_speaker=final_turn.role_name,
            final_content=final_turn.content,
        ),
        raw_text=final_turn.content,
        tokens_used=0,
        duration_seconds=duration,
        success=True,
        error=None,
    )

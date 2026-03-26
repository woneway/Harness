"""discussion.py — Discussion 多 Agent 结构化讨论执行逻辑。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

    from harness.agent import Agent
    from harness.runners.base import AbstractRunner
    from harness.state import State
    from harness.storage.base import StorageProtocol
    from harness.tasks import Result

from harness._internal.exceptions import TaskFailedError
from harness._internal.task_index import TaskIndex
from harness.tasks.discussion import (
    Discussion,
    DiscussionOutput,
    DiscussionProgressEvent,
    DiscussionTurn,
)
from harness.tasks.config import TaskConfig

logger = logging.getLogger(__name__)


# ── DiscussionContext ───────────────────────────────────────────────────────


@dataclass
class DiscussionContext:
    """每次构建 Agent prompt 时传入的上下文。"""

    round: int
    agent_name: str
    topic: str
    background: str                         # 已解析的背景（str，不是 callable）
    state: "State | None" = None

    # 讨论历史
    history: list[DiscussionTurn] = None  # type: ignore[assignment]

    # 立场数据
    my_position: Any | None = None
    positions: dict[str, Any] = None  # type: ignore[assignment]
    position_history: dict[str, list[Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []
        if self.positions is None:
            self.positions = {}
        if self.position_history is None:
            self.position_history = {}

    def last_response_from(self, name: str) -> str | None:
        """获取指定 Agent 最近一次发言文字。"""
        for turn in reversed(self.history):
            if turn.agent_name == name:
                return turn.response
        return None

    def all_responses_from(self, name: str) -> list[str]:
        """获取指定 Agent 所有历史发言文字，按时间顺序。"""
        return [t.response for t in self.history if t.agent_name == name]

    def position_of(self, name: str) -> Any | None:
        """获取某 Agent 最新立场。"""
        return self.positions.get(name)

    def did_change(self, name: str) -> bool:
        """某 Agent 上轮是否改变了立场。"""
        for turn in reversed(self.history):
            if turn.agent_name == name:
                return turn.position_changed
        return False


# ── 默认提示词模板 ──────────────────────────────────────────────────────────


def _format_dict(d: dict) -> str:
    """格式化 dict 为可读文本。"""
    return ", ".join(f"{k}: {v}" for k, v in d.items())


def _default_prompt_template(agent: "Agent", ctx: DiscussionContext) -> str:
    """框架默认的讨论提示词模板。"""
    parts = []

    # 话题
    parts.append(f"## 讨论话题\n{ctx.topic}")

    # 背景
    if ctx.background:
        parts.append(f"## 背景信息\n{ctx.background}")

    # 自己的当前立场
    if ctx.my_position is not None:
        pos_dict = ctx.my_position.model_dump() if hasattr(ctx.my_position, "model_dump") else ctx.my_position
        parts.append(f"## 你的当前立场\n{_format_dict(pos_dict)}")

    # 其他人的最新立场 + 上轮发言
    others_parts = []
    for name, pos in ctx.positions.items():
        if name == ctx.agent_name:
            continue
        pos_dict = pos.model_dump() if hasattr(pos, "model_dump") else pos
        entry = f"### {name}\n立场：{_format_dict(pos_dict)}"
        last_resp = ctx.last_response_from(name)
        if last_resp:
            entry += f"\n分析：{last_resp[:500]}"
        if ctx.did_change(name):
            entry += "\n（上轮已调整立场）"
        others_parts.append(entry)
    if others_parts:
        parts.append("## 其他参与者\n" + "\n\n".join(others_parts))

    # 指令
    if ctx.round == 0:
        parts.append("## 指令\n请基于你的角色和背景信息，给出你的初始分析和立场。")
    else:
        parts.append("## 指令\n请基于讨论进展，更新或坚持你的立场，并说明理由。")

    return "\n\n".join(parts)


# ── Prompt 解析 ─────────────────────────────────────────────────────────────


def _resolve_prompt(
    agent: "Agent",
    ctx: DiscussionContext,
    discussion: Discussion,
) -> str:
    """三级优先级解析 prompt。

    1. agent_prompts[agent.name] — per-agent 定制
    2. prompt_template — 全局模板
    3. _default_prompt_template — 框架默认
    """
    if discussion.agent_prompts and agent.name in discussion.agent_prompts:
        return discussion.agent_prompts[agent.name](ctx)
    if discussion.prompt_template is not None:
        return discussion.prompt_template(agent, ctx)
    return _default_prompt_template(agent, ctx)


# ── System prompt 合并 ──────────────────────────────────────────────────────


def _merge_system_prompt(
    agent: "Agent",
    harness_sp: str,
) -> str:
    """合并 system_prompt：agent.build_system_prompt() + harness_sp。"""
    parts = []
    agent_sp = agent.build_system_prompt()
    if agent_sp:
        parts.append(agent_sp)
    if harness_sp:
        parts.append(harness_sp)
    return "\n\n".join(parts)


# ── Phase 2: 立场提取 ──────────────────────────────────────────────────────


def _build_extraction_prompt(response_text: str, position_schema: type["BaseModel"]) -> str:
    """构建 Phase 2 立场提取 prompt。"""
    schema_json = json.dumps(position_schema.model_json_schema(), ensure_ascii=False, indent=2)
    return (
        "请从以下发言中提取结构化立场，严格按 JSON schema 输出。\n\n"
        f"## 发言原文\n{response_text}\n\n"
        f"## JSON Schema\n```json\n{schema_json}\n```\n\n"
        "请只输出一个 JSON 对象，不要包含其他文字。"
    )


def _extract_position(text: str, position_schema: type["BaseModel"]) -> "Any | None":
    """三级 JSON 提取容错：直接解析 → 代码块 → 正则提取 → None。"""
    import re

    # Level 1: 直接解析
    try:
        return position_schema.model_validate_json(text)
    except Exception:
        pass

    # Level 2: 提取 ```json ... ``` 代码块
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block:
        try:
            return position_schema.model_validate_json(code_block.group(1).strip())
        except Exception:
            pass

    # Level 3: 正则提取第一个 { ... }
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return position_schema.model_validate_json(brace_match.group(0))
        except Exception:
            pass

    return None


# ── 单次发言执行 ────────────────────────────────────────────────────────────


async def _execute_agent_turn(
    agent: "Agent",
    task_index: str,
    round_num: int,
    discussion: Discussion,
    ctx: DiscussionContext,
    agent_sessions: dict[str, str | None],
    harness_system_prompt: str,
    harness_runner: "AbstractRunner",
    storage: "StorageProtocol | None",
    run_id: str,
    config: TaskConfig,
) -> "tuple[DiscussionTurn, int, Any]":
    """两阶段执行单次 Agent 发言，返回 (DiscussionTurn, tokens, parsed_position)。

    Phase 1: 自由文本分析（agent session，流式输出，无 JSON 约束）
    Phase 2: 轻量立场提取（fresh session，output_schema，无流式）

    Phase 2 解析失败时降级：保留上一轮立场（与旧行为一致）。
    """
    # 解析 prompt
    try:
        prompt_text = _resolve_prompt(agent, ctx, discussion)
    except Exception as e:
        raise TaskFailedError(
            run_id, task_index, "discussion",
            f"Agent '{agent.name}' prompt callable raised: {e}",
        )

    system = _merge_system_prompt(agent, harness_system_prompt)
    runner = agent.runner or harness_runner

    last_error = ""
    attempt = 0
    turn_start = time.monotonic()

    # 封装 role_stream_callback
    role_stream_cb = None
    if discussion.role_stream_callback is not None:
        _cb = discussion.role_stream_callback
        role_stream_cb = lambda chunk: _cb(agent.name, chunk)  # noqa: E731

    # 自动流式进度
    effective_stream_cb = role_stream_cb
    if effective_stream_cb is None and discussion.progress_callback is not None:
        _pcb = discussion.progress_callback
        _agent_name = agent.name

        def _auto_stream_cb(chunk: str) -> None:
            _pcb(
                DiscussionProgressEvent(
                    event="streaming",
                    round=round_num,
                    agent_name=_agent_name,
                    content=chunk,
                )
            )

        effective_stream_cb = _auto_stream_cb

    while attempt <= config.max_retries:
        # 进度回调：发言开始
        if discussion.progress_callback:
            discussion.progress_callback(
                DiscussionProgressEvent(event="start", round=round_num, agent_name=agent.name)
            )

        try:
            # ── Phase 1: 自由文本分析 ──
            phase1_result = await asyncio.wait_for(
                runner.execute(
                    prompt_text,
                    system_prompt=system,
                    session_id=agent_sessions[agent.name],
                    output_schema_json=None,
                    stream_callback=effective_stream_cb,
                ),
                timeout=config.timeout,
            )
        except asyncio.TimeoutError:
            last_error = f"Discussion turn {task_index} timed out after {config.timeout}s"
            logger.warning(last_error)
        except Exception as e:
            last_error = str(e)
            logger.warning("Discussion turn %s attempt %d failed: %s", task_index, attempt, e)
        else:
            # Phase 1 成功：更新 agent session
            if phase1_result.session_id:
                agent_sessions[agent.name] = phase1_result.session_id

            response_text = phase1_result.text
            total_tokens = phase1_result.tokens_used

            # ── Phase 2: 立场提取 ──
            prev_position = ctx.my_position
            extraction_prompt = _build_extraction_prompt(
                response_text, discussion.position_schema,
            )
            extraction_runner = discussion.extraction_runner or runner
            position_schema_json = json.dumps(
                discussion.position_schema.model_json_schema(),
            )

            phase2_timeout = min(config.timeout, 60)
            try:
                phase2_result = await asyncio.wait_for(
                    extraction_runner.execute(
                        extraction_prompt,
                        system_prompt="",
                        session_id=None,
                        output_schema_json=position_schema_json,
                    ),
                    timeout=phase2_timeout,
                )
                total_tokens += phase2_result.tokens_used

                position = _extract_position(phase2_result.text, discussion.position_schema)
                if position is not None:
                    changed = prev_position is None or (
                        position.model_dump() != prev_position.model_dump()
                        if hasattr(prev_position, "model_dump")
                        else position.model_dump() != prev_position
                    )
                else:
                    # Phase 2 提取失败降级
                    logger.warning(
                        "Discussion turn %s Phase 2 extraction failed, degrading",
                        task_index,
                    )
                    position = prev_position
                    changed = False
            except Exception as phase2_err:
                # Phase 2 整体失败降级
                logger.warning(
                    "Discussion turn %s Phase 2 failed, degrading: %s",
                    task_index, phase2_err,
                )
                position = prev_position
                changed = False

            turn = DiscussionTurn(
                round=round_num,
                agent_name=agent.name,
                response=response_text,
                position=position,
                position_changed=changed,
            )

            if storage is not None:
                await storage.save_task_log(
                    run_id, task_index, "discussion",
                    output=response_text,
                    output_schema_class=None,
                    raw_text=response_text,
                    tokens_used=total_tokens,
                    duration_seconds=time.monotonic() - turn_start,
                    success=True,
                    error=None,
                )

            # 进度回调：发言完成
            if discussion.progress_callback:
                discussion.progress_callback(
                    DiscussionProgressEvent(
                        event="complete",
                        round=round_num,
                        agent_name=agent.name,
                        content=response_text,
                    )
                )
            return turn, total_tokens, position

        attempt += 1
        if attempt <= config.max_retries:
            wait = config.backoff_base ** (attempt - 1)
            await asyncio.sleep(wait)

    # 进度回调：发言出错
    if discussion.progress_callback:
        discussion.progress_callback(
            DiscussionProgressEvent(
                event="error",
                round=round_num,
                agent_name=agent.name,
                content=last_error,
            )
        )

    raise TaskFailedError(run_id, task_index, "discussion", last_error)


# ── 执行主循环 ──────────────────────────────────────────────────────────────


async def execute_discussion(
    discussion: Discussion,
    outer_index: int,
    pipeline_results: list["Result"],
    run_id: str,
    *,
    harness_system_prompt: str,
    harness_runner: "AbstractRunner",
    harness_config: "TaskConfig | None",
    storage: "StorageProtocol | None" = None,
    state: "State | None" = None,
) -> "Result":
    """执行 Discussion，追踪立场演变，检测收敛。"""
    from harness.tasks import Result

    if discussion.position_schema is None:
        raise TaskFailedError(
            run_id, str(outer_index), "discussion",
            "Discussion.position_schema is required.",
        )

    config = discussion.config or harness_config or TaskConfig()
    start_time = time.monotonic()

    agent_sessions: dict[str, str | None] = {a.name: None for a in discussion.agents}
    history: list[DiscussionTurn] = []
    position_history: dict[str, list[Any]] = {a.name: [] for a in discussion.agents}
    current_positions: dict[str, Any] = {}
    total_tokens = 0
    converged = False
    convergence_round: int | None = None

    # 解析 background
    resolved_bg = ""
    if discussion.background:
        if callable(discussion.background):
            resolved_bg = discussion.background(state) if state is not None else ""
        else:
            resolved_bg = discussion.background

    for round_num in range(discussion.max_rounds):
        round_done = False
        for agent_idx, agent in enumerate(discussion.agents):
            task_index = str(TaskIndex.disc_round(outer_index, round_num, agent_idx))

            # 构建 DiscussionContext
            ctx = DiscussionContext(
                round=round_num,
                agent_name=agent.name,
                topic=discussion.topic,
                background=resolved_bg,
                state=state,
                history=list(history),
                my_position=current_positions.get(agent.name),
                positions=dict(current_positions),
                position_history={k: list(v) for k, v in position_history.items()},
            )

            turn, turn_tokens, position = await _execute_agent_turn(
                agent, task_index, round_num,
                discussion, ctx,
                agent_sessions, harness_system_prompt, harness_runner,
                storage, run_id, config,
            )

            history.append(turn)
            total_tokens += turn_tokens

            # 更新立场追踪（position 可能为 None 如果是降级且首轮）
            if position is not None:
                current_positions[agent.name] = position
                position_history[agent.name].append(position)

            # 检查 until（传入更新后的 context）
            if discussion.until is not None:
                updated_ctx = DiscussionContext(
                    round=round_num,
                    agent_name=agent.name,
                    topic=discussion.topic,
                    background=resolved_bg,
                    state=state,
                    history=list(history),
                    my_position=current_positions.get(agent.name),
                    positions=dict(current_positions),
                    position_history={k: list(v) for k, v in position_history.items()},
                )
                if discussion.until(updated_ctx):
                    round_done = True
                    break

        if round_done:
            break

        # 检查收敛（每轮结束后）
        if discussion.convergence is not None and discussion.convergence(position_history):
            converged = True
            convergence_round = round_num
            break

    duration = time.monotonic() - start_time
    rounds_completed = min(round_num + 1, discussion.max_rounds) if discussion.agents else 0

    return Result(
        task_index=str(outer_index),
        task_type="discussion",
        output=DiscussionOutput(
            turns=history,
            rounds_completed=rounds_completed,
            total_turns=len(history),
            final_positions=dict(current_positions),
            converged=converged,
            convergence_round=convergence_round,
            position_history={k: list(v) for k, v in position_history.items()},
        ),
        raw_text=history[-1].response if history else "",
        tokens_used=total_tokens,
        duration_seconds=duration,
        success=True,
        error=None,
    )

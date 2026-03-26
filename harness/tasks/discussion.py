"""Discussion — 多 Agent 结构化讨论任务。

与 Dialogue 的根本区别：
- Dialogue: 角色交换文本 → 产出对话记录
- Discussion: Agent 更新结构化立场 → 产出决策记录（立场演变 + 最终决策）

position_schema 是必填项，定义了每个 Agent 维护的立场结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

from harness.tasks.base import BaseTask

if TYPE_CHECKING:
    from pydantic import BaseModel

    from harness.agent import Agent
    from harness.state import State


# ── 数据类 ──────────────────────────────────────────────────────────────────


@dataclass
class DiscussionTurn:
    """一次 Agent 发言记录（含结构化立场）。"""

    round: int           # 轮次，从 0 开始
    agent_name: str
    response: str        # 文字解释（其他 Agent 可见）
    position: Any        # 结构化立场（BaseModel 实例）
    position_changed: bool  # 立场是否相比上轮发生了变化


@dataclass
class DiscussionOutput:
    """Discussion 执行结果，作为 Result.output 存储。

    与 DialogueOutput 的根本区别：
    - position_history — 框架级的立场追踪，不是文本记录
    - final_positions — 结构化的最终决策，不是 final_content: str
    - converged — 框架判断的收敛状态，不是用户回调
    """

    turns: list[DiscussionTurn]
    rounds_completed: int
    total_turns: int

    # 最终决策
    final_positions: dict[str, Any]      # agent_name → 最终 position
    converged: bool                      # 是否达成收敛
    convergence_round: int | None        # 哪一轮收敛（None 表示未收敛）

    # 立场演变
    position_history: dict[str, list[Any]]  # agent_name → [round0_pos, round1_pos, ...]


@dataclass
class DiscussionProgressEvent:
    """Discussion progress_callback 接收的结构化事件。"""

    event: Literal["start", "complete", "error", "streaming"]
    round: int
    agent_name: str
    content: str | None = None


# ── Discussion PipelineStep ─────────────────────────────────────────────────


@dataclass
class Discussion(BaseTask):
    """多 Agent 结构化讨论。

    与 Dialogue 的根本区别：
    - Dialogue: 角色交换文本 → 产出对话记录
    - Discussion: Agent 更新结构化立场 → 产出决策记录

    position_schema 是必填项，定义了每个 Agent 维护的立场结构。
    """

    agents: list["Agent"] = field(default_factory=list)
    position_schema: type["BaseModel"] | None = None  # 必填

    topic: str = ""
    background: str | Callable[["State"], str] = ""
    max_rounds: int = 5

    # 收敛检测（Discussion 独有）
    convergence: Callable[[dict[str, list[Any]]], bool] | None = None

    # 提示词定制（可选，框架有默认模板）
    prompt_template: Callable[["Agent", "DiscussionContext"], str] | None = None
    agent_prompts: dict[str, Callable[["DiscussionContext"], str]] | None = None

    # 自定义终止（与 convergence 互补）
    until: Callable[["DiscussionContext"], bool] | None = None

    output_key: str | None = None
    progress_callback: Callable[[DiscussionProgressEvent], None] | None = None
    role_stream_callback: Callable[[str, str], None] | None = None


# Forward reference placeholder for type checking only
if TYPE_CHECKING:
    from harness._internal.discussion import DiscussionContext


# ── 收敛工具函数 ────────────────────────────────────────────────────────────


def all_agree_on(field_name: str) -> Callable[[dict[str, list[Any]]], bool]:
    """所有 Agent 在某字段上达成一致时收敛。

    Args:
        field_name: position schema 中的字段名。

    Returns:
        convergence 回调函数。
    """
    def _check(position_history: dict[str, list[Any]]) -> bool:
        values = set()
        for positions in position_history.values():
            if not positions:
                return False
            latest = positions[-1]
            val = getattr(latest, field_name, None)
            if val is None:
                return False
            values.add(val)
        return len(values) == 1

    return _check


def positions_stable(rounds: int = 2) -> Callable[[dict[str, list[Any]]], bool]:
    """连续 N 轮所有 Agent 立场不变时收敛。

    Args:
        rounds: 需要稳定的连续轮数。

    Returns:
        convergence 回调函数。
    """
    def _check(position_history: dict[str, list[Any]]) -> bool:
        for positions in position_history.values():
            if len(positions) < rounds:
                return False
            recent = positions[-rounds:]
            first_dump = recent[0].model_dump() if hasattr(recent[0], "model_dump") else recent[0]
            for pos in recent[1:]:
                dump = pos.model_dump() if hasattr(pos, "model_dump") else pos
                if dump != first_dump:
                    return False
        return True

    return _check


def majority_agree_on(
    field_name: str, threshold: float = 0.6
) -> Callable[[dict[str, list[Any]]], bool]:
    """多数 Agent 在某字段上一致时收敛。

    Args:
        field_name: position schema 中的字段名。
        threshold: 一致比例阈值（0-1）。

    Returns:
        convergence 回调函数。
    """
    def _check(position_history: dict[str, list[Any]]) -> bool:
        if not position_history:
            return False
        values: list[Any] = []
        for positions in position_history.values():
            if not positions:
                return False
            latest = positions[-1]
            val = getattr(latest, field_name, None)
            values.append(val)

        total = len(values)
        from collections import Counter
        counts = Counter(v for v in values if v is not None)
        if not counts:
            return False
        most_common_count = counts.most_common(1)[0][1]
        return most_common_count / total >= threshold

    return _check

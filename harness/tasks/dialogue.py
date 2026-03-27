"""Dialogue / Role — 多角色对话任务。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from harness.tasks.base import BaseTask, DialogueProgressEvent


@dataclass
class DialogueTurn:
    """一次角色发言记录。"""

    round: int       # 轮次，从 0 开始
    role_name: str
    content: str


@dataclass
class DialogueOutput:
    """Dialogue 执行结果，作为 Result.output 存储。

    Attributes:
        turns: 所有发言记录，按时间顺序。
        rounds_completed: 轮次模式下表示已完成（含部分）轮数；
            回合模式下与 total_turns 相同（无"轮"的概念）。
        total_turns: 所有模式下的实际发言总次数（len(turns)）。
        final_speaker: 最后发言的角色名。
        final_content: 最后发言的内容。
    """

    turns: list[DialogueTurn]
    rounds_completed: int
    total_turns: int     # 实际发言总次数，语义明确，不依赖模式
    final_speaker: str   # 最后发言的角色名
    final_content: str   # 最后发言的内容


@dataclass
class Role:
    """Dialogue 中的一个参与者。"""

    name: str
    system_prompt: str
    prompt: Callable[["DialogueContext"], str]
    runner: Any | None = None  # None 时继承 Harness 默认 runner


@dataclass
class Dialogue(BaseTask):
    """多角色对话，支持两种模式：

    **轮次模式**（默认，next_speaker=None）：
        按 roles 顺序每轮各发言一次，until 在每次发言后检查。
        适合：专家小组各自陈述、每人都要发言一次的场景。

    **回合模式**（设置 next_speaker）：
        由 next_speaker(history) 动态决定每次谁发言，
        until 在每次发言后检查，支持点名回应、随机发言顺序等。
        适合：真正的辩论、角色互相点名、动态参与的场景。

    v1 限制：不支持嵌套在 Parallel 内部。
    """

    roles: list[Role] = field(default_factory=list)
    background: str = ""
    max_rounds: int = 3
    until: Callable[["DialogueContext"], bool] | None = None
    # 轮次模式专用：每轮所有角色发言完毕后检查，比 until 更直观。
    # until_round(ctx) 返回 True 时结束，此时 ctx.role_name 为最后一个角色名。
    until_round: Callable[["DialogueContext"], bool] | None = None

    # 回合模式专用：设置后启用动态发言顺序
    next_speaker: Callable[[list["DialogueTurn"]], str] | None = None
    # 回合模式最大发言次数；None 时默认 max_rounds × len(roles)
    max_turns: int | None = None

    # 进度回调：每次发言开始/结束时调用，用于进度显示。
    # 签名：(event: DialogueProgressEvent) → None
    progress_callback: Callable[["DialogueProgressEvent"], None] | None = None
    # 多角色 streaming 回调：实时接收 runner 输出的文本片段，携带角色名。
    # 签名：(role_name: str, chunk: str)
    # 注：不覆盖 BaseTask.stream_callback（Callable[[str], None]），两者语义不同。
    role_stream_callback: Callable[[str, str], None] | None = None


# Forward reference placeholder for type checking only
if TYPE_CHECKING:
    from harness._internal.dialogue import DialogueContext

"""task_index.py — 结构化任务索引，替代散落各处的字符串拼接与解析。

DB 存储格式（向后兼容）：
    sequential:   "2"
    parallel:     "2.0"
    dlg_round:    "2.r0.1"
    dlg_turn:     "2.t3"
    disc_round:   "2.g0.1"
    cond_true:    "3.c0"
    cond_false:   "3.f0"
    loop_iter:    "4.i2.1"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TaskIndex:
    """任务索引。

    Attributes:
        outer:  该任务在顶层 pipeline 中的位置（整数）。
        kind:   索引类型：
                  "seq"        — 顺序任务
                  "par"        — Parallel 子任务
                  "dlg_round"  — Dialogue 轮次模式子任务
                  "dlg_turn"   — Dialogue 回合模式子任务
                  "disc_round" — Discussion 轮次子任务
                  "cond_true"  — Condition if_true 子步骤
                  "cond_false" — Condition if_false 子步骤
                  "loop_iter"  — Loop 迭代子步骤
        sub:    子索引（par: child_index; dlg_round: role_idx; dlg_turn: turn_num;
                disc_round: agent_idx; cond_true/cond_false: child_index;
                loop_iter: child_index）。
        round_: Dialogue/Discussion 轮次模式专用，round number。
        iter_:  Loop 专用，迭代号（kind=="loop_iter" 时非 None）。
    """

    outer: int
    kind: Literal["seq", "par", "dlg_round", "dlg_turn", "disc_round",
                   "cond_true", "cond_false", "loop_iter"] = "seq"
    sub: int | None = None
    round_: int | None = None
    iter_: int | None = None

    # ── 工厂方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def sequential(outer: int) -> "TaskIndex":
        return TaskIndex(outer=outer, kind="seq")

    @staticmethod
    def parallel_child(outer: int, sub: int) -> "TaskIndex":
        return TaskIndex(outer=outer, kind="par", sub=sub)

    @staticmethod
    def dialogue_turn(outer: int, turn: int) -> "TaskIndex":
        return TaskIndex(outer=outer, kind="dlg_turn", sub=turn)

    @staticmethod
    def dialogue_round(outer: int, round_: int, role_idx: int) -> "TaskIndex":
        return TaskIndex(outer=outer, kind="dlg_round", sub=role_idx, round_=round_)

    @staticmethod
    def cond_true(outer: int, child: int) -> "TaskIndex":
        return TaskIndex(outer=outer, kind="cond_true", sub=child)

    @staticmethod
    def cond_false(outer: int, child: int) -> "TaskIndex":
        return TaskIndex(outer=outer, kind="cond_false", sub=child)

    @staticmethod
    def disc_round(outer: int, round_: int, agent_idx: int) -> "TaskIndex":
        return TaskIndex(outer=outer, kind="disc_round", sub=agent_idx, round_=round_)

    @staticmethod
    def loop_iter(outer: int, iteration: int, child: int) -> "TaskIndex":
        return TaskIndex(outer=outer, kind="loop_iter", sub=child, iter_=iteration)

    # ── 序列化 ────────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        if self.kind == "seq":
            return str(self.outer)
        if self.kind == "par":
            return f"{self.outer}.{self.sub}"
        if self.kind == "dlg_turn":
            return f"{self.outer}.t{self.sub}"
        if self.kind == "dlg_round":
            return f"{self.outer}.r{self.round_}.{self.sub}"
        if self.kind == "disc_round":
            return f"{self.outer}.g{self.round_}.{self.sub}"
        if self.kind == "cond_true":
            return f"{self.outer}.c{self.sub}"
        if self.kind == "cond_false":
            return f"{self.outer}.f{self.sub}"
        if self.kind == "loop_iter":
            return f"{self.outer}.i{self.iter_}.{self.sub}"
        raise ValueError(f"Unknown TaskIndex kind: {self.kind!r}")

    # ── 反序列化 ──────────────────────────────────────────────────────────────

    @classmethod
    def parse(cls, s: str) -> "TaskIndex":
        """从字符串解析 TaskIndex。

        Raises:
            ValueError: 字符串格式不符合已知任何格式。
        """
        if "." not in s:
            return cls(outer=int(s), kind="seq")

        outer_str, rest = s.split(".", 1)
        outer = int(outer_str)

        if rest.startswith("r"):
            # dlg_round: "r{round}.{role_idx}"
            inner = rest[1:]
            round_str, role_str = inner.split(".", 1)
            return cls(outer=outer, kind="dlg_round", sub=int(role_str), round_=int(round_str))

        if rest.startswith("g"):
            # disc_round: "g{round}.{agent_idx}"
            inner = rest[1:]
            round_str, agent_str = inner.split(".", 1)
            return cls(outer=outer, kind="disc_round", sub=int(agent_str), round_=int(round_str))

        if rest.startswith("t"):
            # dlg_turn: "t{turn_num}"
            return cls(outer=outer, kind="dlg_turn", sub=int(rest[1:]))

        if rest.startswith("c"):
            # cond_true: "c{child}"
            return cls(outer=outer, kind="cond_true", sub=int(rest[1:]))

        if rest.startswith("f"):
            # cond_false: "f{child}"
            return cls(outer=outer, kind="cond_false", sub=int(rest[1:]))

        if rest.startswith("i"):
            # loop_iter: "i{iter}.{child}"
            inner = rest[1:]
            iter_str, child_str = inner.split(".", 1)
            return cls(outer=outer, kind="loop_iter", sub=int(child_str), iter_=int(iter_str))

        # par: "{child_int}"
        return cls(outer=outer, kind="par", sub=int(rest))

    # ── 查询属性 ──────────────────────────────────────────────────────────────

    @property
    def is_child(self) -> bool:
        """True 当此索引表示子任务（parallel, dialogue, condition, or loop child）。"""
        return self.kind != "seq"

    @property
    def outer_key(self) -> str:
        """外层任务的字符串键，用于 resume 分组。"""
        return str(self.outer)

    def par_child_int(self) -> int:
        """Parallel 子任务的整数索引（仅 kind=="par" 时有效）。

        Raises:
            ValueError: 索引不是 parallel child。
        """
        if self.kind != "par" or self.sub is None:
            raise ValueError(f"TaskIndex {self!r} is not a parallel child")
        return self.sub

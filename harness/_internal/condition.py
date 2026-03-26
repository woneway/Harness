"""condition.py — Condition 执行逻辑。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from harness._internal.task_index import TaskIndex
from harness.tasks.condition import Condition
from harness.tasks.result import Result

if TYPE_CHECKING:
    from harness.state import State


async def execute_condition(
    condition: Condition,
    outer_index: int,
    state: "State",
    *,
    execute_step: Callable,
) -> list[Result]:
    """执行 Condition：评估 check(state)，执行对应分支。

    Args:
        condition: Condition 任务。
        outer_index: pipeline 中的位置索引。
        state: 当前 pipeline State。
        execute_step: 回调函数，签名 (step, task_index, state) -> Result，
            由 harness.py 提供，用于递归执行子步骤。

    Returns:
        分支内所有子步骤的 Result 列表。
    """
    branch_taken = condition.check(state)
    steps = condition.if_true if branch_taken else condition.if_false

    results: list[Result] = []
    for child_idx, step in enumerate(steps):
        if branch_taken:
            task_index = str(TaskIndex.cond_true(outer_index, child_idx))
        else:
            task_index = str(TaskIndex.cond_false(outer_index, child_idx))

        r = await execute_step(step, task_index, state)
        results.append(r)

    return results

"""loop.py — Loop 执行逻辑。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from harness._internal.task_index import TaskIndex
from harness.tasks.loop import Loop
from harness.tasks.result import Result

if TYPE_CHECKING:
    from harness.state import State


async def execute_loop(
    loop: Loop,
    outer_index: int,
    state: "State",
    *,
    execute_step: Callable,
) -> list[Result]:
    """执行 Loop：重复执行 body 直到 until(state) 返回 True 或达到 max_iterations。

    Args:
        loop: Loop 任务。
        outer_index: pipeline 中的位置索引。
        state: 当前 pipeline State。
        execute_step: 回调函数，签名 (step, task_index, state) -> Result，
            由 harness.py 提供，用于递归执行子步骤。

    Returns:
        所有迭代中所有子步骤的 Result 列表。
    """
    all_results: list[Result] = []

    for iteration in range(loop.max_iterations):
        iter_results: list[Result] = []
        for child_idx, step in enumerate(loop.body):
            task_index = str(TaskIndex.loop_iter(outer_index, iteration, child_idx))
            r = await execute_step(step, task_index, state)
            iter_results.append(r)

        all_results.extend(iter_results)

        # 检查终止条件
        if loop.until(state):
            break

    return all_results

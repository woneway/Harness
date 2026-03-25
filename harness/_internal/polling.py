"""polling.py — PollingTask 轮询执行逻辑。"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from harness._internal.exceptions import TaskFailedError
from harness.tasks import PollingTask, Result, TaskConfig

if TYPE_CHECKING:
    from harness.state import State


async def execute_polling(
    task: PollingTask,
    task_index: str,
    results: list[Result],
    run_id: str,
    *,
    harness_config: TaskConfig | None = None,
    state: "State | None" = None,
) -> Result:
    """执行 PollingTask：submit → 循环 poll → 返回最终结果。

    失败时触发 max_retries 重试（重试时重新 submit）。
    """
    from harness._internal.executor import _effective_config

    config = _effective_config(task.config, harness_config)

    last_error = "Max retries exceeded"
    attempt = 0

    while attempt <= config.max_retries:
        # config.timeout 作为每次 attempt 的整体上界（含 submit + poll 循环）。
        # task.timeout 是 _run_once 内部 poll 循环的专属 deadline。
        # 若 config.timeout < task.timeout，asyncio.wait_for 优先触发。
        try:
            result, error = await asyncio.wait_for(
                _run_once(task, task_index, results, run_id, config, state=state),
                timeout=config.timeout,
            )
        except asyncio.TimeoutError:
            result = None
            error = (
                f"polling attempt {attempt} timed out after {config.timeout}s "
                "(TaskConfig.timeout)"
            )
        if result is not None:
            return result
        last_error = error
        attempt += 1
        if attempt <= config.max_retries:
            wait = config.backoff_base ** (attempt - 1)
            await asyncio.sleep(wait)

    raise TaskFailedError(
        run_id, task_index, "polling", last_error, partial_results=list(results)
    )


async def _run_once(
    task: PollingTask,
    task_index: str,
    results: list[Result],
    run_id: str,
    config: TaskConfig,
    *,
    state: "State | None" = None,
) -> tuple[Result | None, str]:
    """执行一次 submit + poll 循环，成功返回 (Result, "")，失败返回 (None, error_msg)。"""
    from pydantic import ValidationError

    from harness._internal.compat import call_with_compat

    start_time = time.monotonic()

    # submit（支持 v1 和 v2 callable 模式）
    try:
        if state is not None:
            handle: Any = call_with_compat(task.submit_fn, state)
        else:
            handle = task.submit_fn(results)
    except Exception as e:
        return None, f"submit_fn raised: {e}"

    # poll 循环
    deadline = time.monotonic() + task.timeout

    while True:
        if time.monotonic() > deadline:
            return None, f"polling timed out after {task.timeout}s"

        await asyncio.sleep(task.poll_interval)

        try:
            response = task.poll_fn(handle)
        except Exception as e:
            return None, f"poll_fn raised: {e}"

        # success_condition
        if task.success_condition(response):
            duration = time.monotonic() - start_time
            output: Any = response

            # output_schema 校验
            if task.output_schema is not None:
                try:
                    output = task.output_schema.model_validate(response)
                except (ValidationError, Exception) as e:
                    return None, f"output_schema validation failed: {e}"

            return Result(
                task_index=task_index,
                task_type="polling",
                output=output,
                raw_text=None,
                tokens_used=0,
                duration_seconds=duration,
                success=True,
                error=None,
            ), ""

        # failure_condition
        if task.failure_condition is not None and task.failure_condition(response):
            return None, "failure_condition triggered"

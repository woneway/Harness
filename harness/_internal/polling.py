"""polling.py — PollingTask 轮询执行逻辑。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from harness._internal.exceptions import TaskFailedError
from harness.task import PollingTask, Result, TaskConfig


async def execute_polling(
    task: PollingTask,
    task_index: str,
    results: list[Result],
    run_id: str,
    *,
    harness_config: TaskConfig | None = None,
) -> Result:
    """执行 PollingTask：submit → 循环 poll → 返回最终结果。

    失败时触发 max_retries 重试（重试时重新 submit）。
    """
    from harness._internal.executor import _effective_config

    config = _effective_config(task.config, harness_config)

    last_error = ""
    attempt = 0

    while attempt <= config.max_retries:
        result = await _run_once(task, task_index, results, run_id, config)
        if result is not None:
            return result
        # _run_once 返回 None 表示失败，需要重试
        attempt += 1
        if attempt <= config.max_retries:
            wait = config.backoff_base ** (attempt - 1)
            await asyncio.sleep(wait)

    raise TaskFailedError(
        run_id, task_index, "polling", "Max retries exceeded", partial_results=list(results)
    )


async def _run_once(
    task: PollingTask,
    task_index: str,
    results: list[Result],
    run_id: str,
    config: TaskConfig,
) -> Result | None:
    """执行一次 submit + poll 循环，成功返回 Result，失败返回 None。"""
    from pydantic import ValidationError

    start_time = time.monotonic()

    # submit
    try:
        handle: Any = task.submit_fn(results)
    except Exception as e:
        return None  # submit 失败，触发重试

    # poll 循环
    deadline = time.monotonic() + task.timeout

    while True:
        if time.monotonic() > deadline:
            return None  # 超时，触发重试

        await asyncio.sleep(task.poll_interval)

        try:
            response = task.poll_fn(handle)
        except Exception as e:
            return None  # poll 失败，触发重试

        # success_condition
        if task.success_condition(response):
            duration = time.monotonic() - start_time
            output: Any = response

            # output_schema 校验
            if task.output_schema is not None:
                try:
                    if isinstance(response, dict):
                        output = task.output_schema.model_validate(response)
                    else:
                        output = task.output_schema.model_validate(response)
                except (ValidationError, Exception):
                    return None  # 校验失败，触发重试

            return Result(
                task_index=task_index,
                task_type="polling",
                output=output,
                raw_text=None,
                tokens_used=0,
                duration_seconds=duration,
                success=True,
                error=None,
            )

        # failure_condition
        if task.failure_condition is not None and task.failure_condition(response):
            return None  # 生成失败，触发重试

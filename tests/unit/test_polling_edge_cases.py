"""tests/unit/test_polling_edge_cases.py — PollingTask 边界情况覆盖。"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from harness._internal.exceptions import TaskFailedError
from harness._internal.polling import execute_polling
from harness.task import PollingTask, TaskConfig


def fast_config(max_retries: int = 0, timeout: float = 60) -> TaskConfig:
    return TaskConfig(max_retries=max_retries, backoff_base=0.01, timeout=timeout)


class TestPollingTimeout:
    @pytest.mark.asyncio
    async def test_config_timeout_triggers_asyncio_timeout(self) -> None:
        """config.timeout 触发 asyncio.wait_for TimeoutError 路径。"""

        def submit(results):
            return "h"

        async_sleep_called = False

        def poll(handle):
            return {"status": "pending"}

        task = PollingTask(
            submit_fn=submit,
            poll_fn=poll,
            success_condition=lambda r: r["status"] == "done",
            poll_interval=0.001,
            timeout=100,  # task.timeout 足够大
        )
        # 使用极小的 config.timeout 让 asyncio.wait_for 超时
        with pytest.raises(TaskFailedError) as exc_info:
            await execute_polling(
                task, "0", [], "run-1",
                harness_config=fast_config(max_retries=0, timeout=0.01),
            )
        assert "timed out" in exc_info.value.error


class TestPollingSubmitError:
    @pytest.mark.asyncio
    async def test_submit_fn_raises(self) -> None:
        """submit_fn 抛异常返回 (None, error_msg)，触发重试后最终失败。"""

        def bad_submit(results):
            raise RuntimeError("submit failed")

        task = PollingTask(
            submit_fn=bad_submit,
            poll_fn=lambda h: None,
            success_condition=lambda r: True,
            poll_interval=0.001,
            timeout=10,
        )
        with pytest.raises(TaskFailedError) as exc_info:
            await execute_polling(
                task, "0", [], "run-1",
                harness_config=fast_config(max_retries=1),
            )
        assert "submit_fn raised" in exc_info.value.error


class TestPollingPollError:
    @pytest.mark.asyncio
    async def test_poll_fn_raises(self) -> None:
        """poll_fn 抛异常返回 (None, error_msg)。"""

        def poll(handle):
            raise ValueError("poll error")

        task = PollingTask(
            submit_fn=lambda r: "handle",
            poll_fn=poll,
            success_condition=lambda r: True,
            poll_interval=0.001,
            timeout=10,
        )
        with pytest.raises(TaskFailedError) as exc_info:
            await execute_polling(
                task, "0", [], "run-1",
                harness_config=fast_config(max_retries=0),
            )
        assert "poll_fn raised" in exc_info.value.error


class TestPollingOutputSchemaFailure:
    @pytest.mark.asyncio
    async def test_output_schema_validation_failure(self) -> None:
        """output_schema 校验失败时返回错误，触发重试。"""

        class Expected(BaseModel):
            value: int

        task = PollingTask(
            submit_fn=lambda r: "h",
            poll_fn=lambda h: {"bad": "data"},  # 不符合 schema
            success_condition=lambda r: True,
            poll_interval=0.001,
            timeout=10,
            output_schema=Expected,
        )
        with pytest.raises(TaskFailedError) as exc_info:
            await execute_polling(
                task, "0", [], "run-1",
                harness_config=fast_config(max_retries=0),
            )
        assert "output_schema validation failed" in exc_info.value.error

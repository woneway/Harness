"""tests/unit/test_polling.py — PollingTask 执行逻辑验证。"""

from __future__ import annotations

import asyncio

import pytest

from harness._internal.exceptions import TaskFailedError
from harness._internal.polling import execute_polling
from harness.task import PollingTask, TaskConfig


def fast_config(max_retries: int = 2) -> TaskConfig:
    return TaskConfig(max_retries=max_retries, backoff_base=0.01)


class TestPollingSuccess:
    @pytest.mark.asyncio
    async def test_immediate_success(self) -> None:
        """poll 立即返回 success。"""
        submit_count = [0]
        poll_count = [0]

        def submit(results):
            submit_count[0] += 1
            return "handle"

        def poll(handle):
            poll_count[0] += 1
            return {"status": "done", "url": "http://example.com"}

        task = PollingTask(
            submit_fn=submit,
            poll_fn=poll,
            success_condition=lambda r: r["status"] == "done",
            poll_interval=0.001,
            timeout=10,
            config=fast_config(),
        )
        result = await execute_polling(task, "0", [], "run-1")
        assert result.success is True
        assert result.output == {"status": "done", "url": "http://example.com"}
        assert result.task_type == "polling"
        assert submit_count[0] == 1

    @pytest.mark.asyncio
    async def test_multiple_polls_before_success(self) -> None:
        """poll 需要多次才成功。"""
        poll_calls = [0]

        def poll(handle):
            poll_calls[0] += 1
            if poll_calls[0] >= 3:
                return {"status": "done"}
            return {"status": "pending"}

        task = PollingTask(
            submit_fn=lambda r: "h",
            poll_fn=poll,
            success_condition=lambda r: r["status"] == "done",
            poll_interval=0.001,
            timeout=10,
            config=fast_config(),
        )
        result = await execute_polling(task, "1", [], "run-1")
        assert result.success is True
        assert poll_calls[0] == 3


class TestPollingFailureCondition:
    @pytest.mark.asyncio
    async def test_failure_condition_triggers_retry(self) -> None:
        """failure_condition=True 时触发重试（重新 submit）。"""
        submit_count = [0]
        poll_count = [0]

        def submit(results):
            submit_count[0] += 1
            return f"handle-{submit_count[0]}"

        def poll(handle):
            poll_count[0] += 1
            return {"status": "error"}

        task = PollingTask(
            submit_fn=submit,
            poll_fn=poll,
            success_condition=lambda r: r["status"] == "done",
            failure_condition=lambda r: r["status"] == "error",
            poll_interval=0.001,
            timeout=10,
            config=fast_config(max_retries=2),
        )
        with pytest.raises(TaskFailedError):
            await execute_polling(task, "0", [], "run-1")

        # 失败后重试：submit 应被调用 3 次（1 + 2 retries）
        assert submit_count[0] == 3

    @pytest.mark.asyncio
    async def test_failure_condition_then_success(self) -> None:
        """第一次 failure，第二次 submit 后成功。"""
        attempt = [0]

        def submit(results):
            attempt[0] += 1
            return attempt[0]

        def poll(handle):
            if handle == 1:
                return {"status": "error"}
            return {"status": "done"}

        task = PollingTask(
            submit_fn=submit,
            poll_fn=poll,
            success_condition=lambda r: r["status"] == "done",
            failure_condition=lambda r: r["status"] == "error",
            poll_interval=0.001,
            timeout=10,
            config=fast_config(max_retries=2),
        )
        result = await execute_polling(task, "0", [], "run-1")
        assert result.success is True
        assert attempt[0] == 2


class TestPollingTimeout:
    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(self) -> None:
        """poll 始终 pending，超时后触发重试。"""
        submit_count = [0]

        def submit(results):
            submit_count[0] += 1
            return "h"

        task = PollingTask(
            submit_fn=submit,
            poll_fn=lambda h: {"status": "pending"},
            success_condition=lambda r: r["status"] == "done",
            poll_interval=0.001,
            timeout=0.005,  # 极短超时
            config=fast_config(max_retries=1),
        )
        with pytest.raises(TaskFailedError):
            await execute_polling(task, "0", [], "run-1")

        # timeout 后重试：submit 应被调用 2 次
        assert submit_count[0] == 2

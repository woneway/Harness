"""parallel.py — Parallel 并发执行逻辑。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable

from harness._internal.exceptions import InvalidPipelineError, TaskFailedError
from harness._internal.task_index import TaskIndex
from harness.tasks import (
    FunctionTask,
    LLMTask,
    Parallel,
    PollingTask,
    Result,
    ShellTask,
    TaskConfig,
)

if TYPE_CHECKING:
    from harness._internal.session import SessionManager
    from harness.runners.base import AbstractRunner
    from harness.state import State
    from harness.storage.base import StorageProtocol


async def execute_parallel(
    parallel: Parallel,
    outer_index: int,
    results: list[Result],
    run_id: str,
    *,
    harness_system_prompt: str,
    harness_runner: "AbstractRunner",
    harness_config: TaskConfig | None,
    session_manager: "SessionManager",
    memory_injection: str,
    storage: "StorageProtocol | None" = None,
    is_new_session: bool = False,
    harness_stream_callback: "Callable[[str], None] | None" = None,
    harness_raw_stream_callback: "Callable[[dict], None] | None" = None,
    env_overrides: "dict[str, str] | None" = None,
    state: "State | None" = None,
) -> list[Result]:
    """并发执行 Parallel 块内的所有 Task。

    Args:
        parallel: Parallel 任务块。
        outer_index: 该 Parallel 在 pipeline 中的位置索引（整数）。
        results: 前序任务结果列表。
        run_id: 当前 run ID。
        state: v2 State 对象（Parallel 子任务接收只读快照）。

    Returns:
        所有子任务的 Result 列表。

    Raises:
        InvalidPipelineError: 如果 tasks 中包含嵌套的 Parallel。
        TaskFailedError: all_or_nothing 策略下任一任务失败时。
    """
    # 运行时校验：不允许嵌套 Parallel
    for t in parallel.tasks:
        if isinstance(t, Parallel):
            raise InvalidPipelineError(
                "Nested Parallel is not supported in v1. "
                f"Found Parallel inside Parallel at index {outer_index}."
            )

    if parallel.error_policy == "all_or_nothing":
        return await _run_all_or_nothing_with_retry(
            parallel,
            outer_index,
            results,
            run_id,
            harness_system_prompt=harness_system_prompt,
            harness_runner=harness_runner,
            harness_config=harness_config,
            session_manager=session_manager,
            memory_injection=memory_injection,
            storage=storage,
            is_new_session=is_new_session,
            harness_stream_callback=harness_stream_callback,
            harness_raw_stream_callback=harness_raw_stream_callback,
            env_overrides=env_overrides,
            state=state,
        )
    else:
        # best_effort：一次性构建协程并等待全部完成
        task_indices: list[str] = []
        task_types: list[str] = []
        coros = []
        for inner_index, task in enumerate(parallel.tasks):
            task_index = str(TaskIndex.parallel_child(outer_index, inner_index))
            task_indices.append(task_index)
            task_types.append(_task_type_str(task))
            coro = _execute_one(
                task,
                task_index,
                results,
                run_id,
                harness_system_prompt=harness_system_prompt,
                harness_runner=harness_runner,
                harness_config=harness_config,
                session_manager=session_manager,
                memory_injection=memory_injection,
                storage=storage,
                is_new_session=is_new_session,
                harness_stream_callback=harness_stream_callback,
                harness_raw_stream_callback=harness_raw_stream_callback,
                env_overrides=env_overrides,
                state=state,
            )
            coros.append(coro)
        return await _run_best_effort(coros, task_indices, task_types)


async def _run_all_or_nothing_with_retry(
    parallel: Parallel,
    outer_index: int,
    results: list[Result],
    run_id: str,
    *,
    harness_system_prompt: str,
    harness_runner: "AbstractRunner",
    harness_config: TaskConfig | None,
    session_manager: "SessionManager",
    memory_injection: str,
    storage: "StorageProtocol | None",
    is_new_session: bool,
    harness_stream_callback: "Callable[[str], None] | None" = None,
    harness_raw_stream_callback: "Callable[[dict], None] | None" = None,
    env_overrides: "dict[str, str] | None" = None,
    state: "State | None" = None,
) -> list[Result]:
    """all_or_nothing 策略：带整块重试上限（parallel.max_retries）的执行。

    重试计数语义与 TaskConfig.max_retries 一致：
        max_retries=0 → 只执行 1 次，不重试
        max_retries=2 → 最多执行 3 次（初始 + 2 次重试）

    每次重试前重新构建协程列表（子 Task 状态无关）。
    重试间退避：固定 1 秒（Parallel 块级，不依赖 TaskConfig.backoff_base）。
    """
    attempt = 0
    last_exc: Exception | None = None

    while attempt <= parallel.max_retries:
        # 每次尝试重新构建协程，避免复用已消耗的协程
        coros = []
        for inner_index, task in enumerate(parallel.tasks):
            task_index = str(TaskIndex.parallel_child(outer_index, inner_index))
            coro = _execute_one(
                task,
                task_index,
                results,
                run_id,
                harness_system_prompt=harness_system_prompt,
                harness_runner=harness_runner,
                harness_config=harness_config,
                session_manager=session_manager,
                memory_injection=memory_injection,
                storage=storage,
                is_new_session=is_new_session,
                harness_stream_callback=harness_stream_callback,
                harness_raw_stream_callback=harness_raw_stream_callback,
                env_overrides=env_overrides,
                state=state,
            )
            coros.append(coro)

        try:
            return await _run_all_or_nothing(coros, run_id, outer_index)
        except Exception as e:
            last_exc = e
            attempt += 1
            if attempt <= parallel.max_retries:
                await asyncio.sleep(1)

    # 所有尝试均已耗尽，传播最后一次异常
    assert last_exc is not None
    raise last_exc


async def _run_all_or_nothing(
    coros: list,
    run_id: str,
    outer_index: int,
) -> list[Result]:
    """任一失败立刻取消其余，抛 TaskFailedError。"""
    tasks = [asyncio.create_task(c) for c in coros]

    # gather 遇到第一个异常时返回异常（return_exceptions=False）
    try:
        results_raw = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results_raw)
    except Exception as e:
        # 取消所有剩余任务
        for t in tasks:
            if not t.done():
                t.cancel()
        # 等待取消完成
        await asyncio.gather(*tasks, return_exceptions=True)

        if isinstance(e, TaskFailedError):
            raise
        raise TaskFailedError(
            run_id,
            str(outer_index),
            "parallel",
            str(e),
        )


async def _run_best_effort(
    coros: list,
    task_indices: list[str],
    task_types: list[str],
) -> list[Result]:
    """等待全部完成，失败的标记 success=False，不抛异常。"""
    results_raw = await asyncio.gather(*[asyncio.create_task(c) for c in coros], return_exceptions=True)

    results: list[Result] = []
    for task_index, task_type, r in zip(task_indices, task_types, results_raw):
        if isinstance(r, Exception):
            results.append(
                Result(
                    task_index=task_index,
                    task_type=task_type,
                    output=None,
                    raw_text=None,
                    tokens_used=0,
                    duration_seconds=0.0,
                    success=False,
                    error=str(r),
                )
            )
        else:
            results.append(r)
    return results


def _task_type_str(task: LLMTask | FunctionTask | ShellTask | PollingTask) -> str:
    """返回 task 对应的 task_type 字符串，与 executor 保持一致。"""
    if isinstance(task, LLMTask):
        return "llm"
    if isinstance(task, FunctionTask):
        return "function"
    if isinstance(task, ShellTask):
        return "shell"
    if isinstance(task, PollingTask):
        return "polling"
    return "unknown"


async def _execute_one(
    task: LLMTask | FunctionTask | ShellTask | PollingTask,
    task_index: str,
    results: list[Result],
    run_id: str,
    *,
    harness_system_prompt: str,
    harness_runner: "AbstractRunner",
    harness_config: TaskConfig | None,
    session_manager: "SessionManager",
    memory_injection: str,
    storage: "StorageProtocol | None",
    is_new_session: bool,
    harness_stream_callback: "Callable[[str], None] | None" = None,
    harness_raw_stream_callback: "Callable[[dict], None] | None" = None,
    env_overrides: "dict[str, str] | None" = None,
    state: "State | None" = None,
) -> Result:
    """按任务类型派发到对应的 executor 函数。

    LLMTask 的流回调优先级：Task 级 > Harness 级。
    LLMTask 使用独立的 SessionManager，避免并发竞争外部共享 session。
    """
    from harness._internal.executor import (
        execute_function_task,
        execute_llm_task,
        execute_shell_task,
    )
    from harness._internal.polling import execute_polling
    from harness._internal.session import SessionManager

    if isinstance(task, LLMTask):
        # 合并 Harness 级回调：Task 级优先，否则降级到 Harness 级
        effective_cb = task.stream_callback or harness_stream_callback
        effective_raw_cb = task.raw_stream_callback or harness_raw_stream_callback
        if effective_cb is not task.stream_callback or effective_raw_cb is not task.raw_stream_callback:
            task = replace(task, stream_callback=effective_cb, raw_stream_callback=effective_raw_cb)
        # Bug2 修复：每个并发 LLMTask 使用独立 session，避免多个任务竞争同一
        # session_manager 导致 last-writer-wins 和潜在的 session 混乱。
        # 外部 session_manager 不被修改；调用方在 Parallel 完成后统一 mark_broken。
        local_session = SessionManager()
        return await execute_llm_task(
            task,
            task_index,
            results,
            run_id,
            harness_system_prompt=harness_system_prompt,
            harness_runner=harness_runner,
            harness_config=harness_config,
            session_manager=local_session,
            memory_injection=memory_injection,
            storage=storage,
            is_new_session=is_new_session,
            state=state,
        )
    elif isinstance(task, FunctionTask):
        return await execute_function_task(
            task, task_index, results, run_id,
            harness_config=harness_config,
            env_overrides=env_overrides,
            state=state,
        )
    elif isinstance(task, ShellTask):
        return await execute_shell_task(
            task, task_index, results, run_id,
            harness_config=harness_config,
            env_overrides=env_overrides,
            state=state,
        )
    elif isinstance(task, PollingTask):
        return await execute_polling(
            task, task_index, results, run_id,
            harness_config=harness_config,
            state=state,
        )
    else:
        raise TypeError(f"Unknown task type: {type(task)}")

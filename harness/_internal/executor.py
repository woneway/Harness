"""executor.py — 单个 PipelineStep 执行入口，按 task_type 派发。"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable

logger = logging.getLogger(__name__)

from pydantic import BaseModel, ValidationError

from harness._internal.compat import call_with_compat
from harness._internal.exceptions import (
    InvalidPipelineError,
    OutputSchemaError,
    TaskFailedError,
)
from harness._internal.session import SessionManager
from harness.runners.base import AbstractRunner
from harness.state import State
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
    from harness.storage.base import StorageProtocol

# 保护 os.environ 并发修改，防止 Parallel 内多个 FunctionTask 竞态。
# 每个事件循环绑定独立 Lock，避免跨 loop 共享导致 RuntimeError。
# 使用 (id, weakref) 对来自动清理已销毁的事件循环对应的 Lock。
import weakref

_env_locks: dict[int, tuple[weakref.ref, asyncio.Lock]] = {}


def _get_env_lock() -> asyncio.Lock:
    """获取当前事件循环绑定的 env_lock，自动清理已销毁循环的条目。"""
    loop = asyncio.get_running_loop()
    loop_id = id(loop)

    # 清理已销毁的事件循环对应的 Lock（避免内存泄漏）
    stale = [k for k, (ref, _) in _env_locks.items() if ref() is None]
    for k in stale:
        del _env_locks[k]

    entry = _env_locks.get(loop_id)
    if entry is not None and entry[0]() is not None:
        return entry[1]

    lock = asyncio.Lock()
    _env_locks[loop_id] = (weakref.ref(loop), lock)
    return lock


def _effective_config(
    task_config: TaskConfig | None,
    harness_config: TaskConfig | None,
) -> TaskConfig:
    """TaskConfig 三级合并：task.config > harness.default_config > TaskConfig()"""
    return task_config or harness_config or TaskConfig()


def _build_prior_context(task_logs: list[dict[str, Any]]) -> str:
    """从前序成功 Task 日志构建 prompt 注入的上下文字符串。"""
    lines = ["=== 前序任务输出 ==="]
    for log in task_logs:
        idx = log.get("task_index", "?")
        ttype = log.get("task_type", "?")
        output = log.get("output") or ""
        # 限制长度
        preview = str(output)[:300]
        lines.append(f"Task {idx} [{ttype}]: {preview}")
    return "\n".join(lines)


async def execute_llm_task(
    task: LLMTask,
    task_index: str,
    results: list[Result],
    run_id: str,
    *,
    harness_system_prompt: str,
    harness_runner: AbstractRunner,
    harness_config: TaskConfig | None,
    session_manager: SessionManager,
    memory_injection: str,
    storage: "StorageProtocol | None" = None,
    is_new_session: bool = False,
    env_overrides: dict[str, str] | None = None,
    state: State | None = None,
) -> Result:
    """执行单个 LLMTask，含重试逻辑。"""
    config = _effective_config(task.config, harness_config)
    runner = task.runner or harness_runner

    # system_prompt 合并
    system_parts = [harness_system_prompt]
    if task.system_prompt:
        system_parts.append(task.system_prompt)
    if memory_injection:
        system_parts.append(memory_injection)
    merged_system = "\n\n".join(p for p in system_parts if p)

    # 输出分隔标识
    stream_cb = task.stream_callback
    raw_cb = task.raw_stream_callback

    _emit_separator(task_index, "llm", stream_cb, raw_cb)

    # 解析 prompt（支持 v1 和 v2 callable 模式）
    try:
        if callable(task.prompt):
            if state is not None:
                base_prompt_text = call_with_compat(
                    task.prompt, state, in_state_pipeline=True,
                )
            else:
                base_prompt_text = task.prompt(results)
        else:
            base_prompt_text = task.prompt
    except Exception as e:
        # Callable prompt 抛异常：不重试，直接抛 TaskFailedError
        raise TaskFailedError(
            run_id, task_index, "llm", f"Prompt callable raised: {e}"
        )

    # 准备 JSON schema
    schema_json: str | None = None
    if task.output_schema is not None:
        schema_json = json.dumps(task.output_schema.model_json_schema())

    start_time = time.monotonic()
    last_error: str = ""
    attempt = 0
    last_parse_error: str | None = None

    while attempt <= config.max_retries:
        # schema 校验失败时将错误提示追加到 base prompt（每次只保留最近一次）
        prompt_text = base_prompt_text
        if last_parse_error is not None:
            prompt_text = (
                base_prompt_text
                + f"\n\n[Previous attempt failed schema validation: {last_parse_error}. "
                "Please respond with valid JSON matching the schema.]"
            )

        # session 断开时注入前序上下文（只在进入新 session 的第一次调用时注入）
        current_prompt = prompt_text
        if is_new_session and storage is not None:
            prior_logs = await storage.get_task_logs(run_id, success_only=True)
            if prior_logs:
                context = _build_prior_context(prior_logs)
                current_prompt = context + "\n\n" + prompt_text
            # 标记已消费：同一 session 内后续 attempt（如 schema 校验重试）不重复注入
            is_new_session = False

        try:
            result_obj = await asyncio.wait_for(
                runner.execute(
                    current_prompt,
                    system_prompt=merged_system,
                    session_id=session_manager.current_session_id,
                    output_schema_json=schema_json,
                    stream_callback=stream_cb,
                    raw_stream_callback=raw_cb,
                    env_overrides=env_overrides,
                ),
                timeout=config.timeout,
            )
        except asyncio.TimeoutError:
            last_error = f"LLMTask {task_index} timed out after {config.timeout}s"
            session_manager.mark_broken()
            is_new_session = True
        except Exception as e:
            last_error = str(e)
            session_manager.mark_broken()
            is_new_session = True
        else:
            session_manager.update(result_obj.session_id)
            duration = time.monotonic() - start_time

            # 尝试解析 output_schema
            output: Any = result_obj.text
            parse_error: str | None = None

            if task.output_schema is not None:
                try:
                    output = task.output_schema.model_validate_json(result_obj.text)
                except (ValidationError, Exception) as e:
                    parse_error = str(e)
                    last_parse_error = parse_error
                    last_error = f"Output schema validation failed: {parse_error}"

            if parse_error is None:
                return Result(
                    task_index=task_index,
                    task_type="llm",
                    output=output,
                    raw_text=result_obj.text,
                    tokens_used=result_obj.tokens_used,
                    duration_seconds=duration,
                    success=True,
                    error=None,
                )

        attempt += 1
        if attempt <= config.max_retries:
            wait = config.backoff_base ** (attempt - 1)
            await asyncio.sleep(wait)

    raise TaskFailedError(
        run_id, task_index, "llm", last_error, partial_results=list(results)
    )


async def execute_function_task(
    task: FunctionTask,
    task_index: str,
    results: list[Result],
    run_id: str,
    *,
    harness_config: TaskConfig | None,
    env_overrides: dict[str, str] | None = None,
    state: State | None = None,
) -> Result:
    """执行单个 FunctionTask。

    output_schema 校验失败抛 OutputSchemaError，不触发重试。
    """
    _emit_separator(task_index, "function", task.stream_callback, task.raw_stream_callback)

    config = _effective_config(task.config, harness_config)
    start_time = time.monotonic()
    attempt = 0
    last_error = ""

    # env_overrides 用 asyncio.Lock 保护，防止 Parallel 内多个 FunctionTask 并发修改
    # os.environ 导致竞态条件。Lock 确保同一时刻只有一个 FunctionTask 修改全局环境。
    async with _get_env_lock():
        _orig_env: dict[str, str | None] = {}
        if env_overrides:
            for key, val in env_overrides.items():
                _orig_env[key] = os.environ.get(key)
                if val == "":
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

        try:
            while attempt <= config.max_retries:
                try:
                    if state is not None:
                        raw_output = call_with_compat(
                            task.fn, state, in_state_pipeline=True,
                        )
                    else:
                        raw_output = task.fn(results)
                    # 支持 async 函数：自动 await coroutine
                    if inspect.isawaitable(raw_output):
                        raw_output = await raw_output
                except OutputSchemaError:
                    raise  # 直接向上抛，不重试
                except Exception as e:
                    last_error = str(e)
                    attempt += 1
                    if attempt <= config.max_retries:
                        wait = config.backoff_base ** (attempt - 1)
                        await asyncio.sleep(wait)
                    continue

                # output_schema 校验：失败抛 OutputSchemaError，不重试
                if task.output_schema is not None:
                    if not isinstance(raw_output, task.output_schema):
                        raise OutputSchemaError(task_index, task.output_schema, type(raw_output))

                duration = time.monotonic() - start_time
                return Result(
                    task_index=task_index,
                    task_type="function",
                    output=raw_output,
                    raw_text=None,
                    tokens_used=0,
                    duration_seconds=duration,
                    success=True,
                    error=None,
                )
        finally:
            if env_overrides:
                for key, orig_val in _orig_env.items():
                    if orig_val is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = orig_val

    raise TaskFailedError(
        run_id, task_index, "function", last_error, partial_results=list(results)
    )


async def execute_shell_task(
    task: ShellTask,
    task_index: str,
    results: list[Result],
    run_id: str,
    *,
    harness_config: TaskConfig | None,
    env_overrides: dict[str, str] | None = None,
    state: State | None = None,
) -> Result:
    """执行单个 ShellTask，非零退出码触发重试。"""
    _emit_separator(task_index, "shell", task.stream_callback, task.raw_stream_callback)

    config = _effective_config(task.config, harness_config)

    # 解析 cmd（支持 v1 和 v2 callable 模式）
    try:
        if callable(task.cmd):
            if state is not None:
                cmd_text = call_with_compat(
                    task.cmd, state, in_state_pipeline=True,
                )
            else:
                cmd_text = task.cmd(results)
        else:
            cmd_text = task.cmd
    except Exception as e:
        raise TaskFailedError(run_id, task_index, "shell", f"cmd callable raised: {e}")

    # 构建子进程环境变量
    # task.env 视为对父进程环境的覆写（overlay），而非完全替换。
    # 这样 task.env={"MY_VAR": "x"} 不会丢失 PATH 等继承变量。
    # env_overrides（Harness 级）始终在最后应用，优先级最高。
    if env_overrides or task.env is not None:
        cmd_env = os.environ.copy()
        if task.env is not None:
            cmd_env.update(task.env)
        for key, val in (env_overrides or {}).items():
            if val == "":
                cmd_env.pop(key, None)
            else:
                cmd_env[key] = val
    else:
        cmd_env = None  # inherit from process

    start_time = time.monotonic()
    last_error = ""
    attempt = 0

    while attempt <= config.max_retries:
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_shell(
                    cmd_text,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=task.cwd,
                    env=cmd_env,
                ),
                timeout=config.timeout,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=config.timeout
            )
        except asyncio.TimeoutError:
            last_error = f"ShellTask {task_index} timed out after {config.timeout}s"
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
        except Exception as e:
            last_error = str(e)
        else:
            if proc.returncode == 0:
                duration = time.monotonic() - start_time
                return Result(
                    task_index=task_index,
                    task_type="shell",
                    output=stdout.decode(errors="replace"),
                    raw_text=stdout.decode(errors="replace"),
                    tokens_used=0,
                    duration_seconds=duration,
                    success=True,
                    error=None,
                )
            last_error = stderr.decode(errors="replace") or f"exit code {proc.returncode}"

        attempt += 1
        if attempt <= config.max_retries:
            wait = config.backoff_base ** (attempt - 1)
            await asyncio.sleep(wait)

    raise TaskFailedError(
        run_id, task_index, "shell", last_error, partial_results=list(results)
    )


def _emit_separator(
    task_index: str,
    task_type: str,
    stream_callback: Callable[[str], None] | None,
    raw_stream_callback: Callable[[dict], None] | None,
) -> None:
    """输出每个 Task 执行前的分隔标识。"""
    separator = f"=== Task {task_index} [{task_type}] ==="
    logger.info(separator)
    if stream_callback is not None:
        try:
            stream_callback(separator + "\n")
        except Exception:
            pass

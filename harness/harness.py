"""Harness — 主类，用户的唯一入口点。"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

from harness._internal.deserialize import deserialize_output
from harness._internal.exceptions import InvalidPipelineError, TaskFailedError
from harness._internal.executor import (
    execute_function_task,
    execute_llm_task,
    execute_shell_task,
)
from harness._internal.dialogue import execute_dialogue
from harness._internal.parallel import execute_parallel
from harness._internal.polling import execute_polling
from harness._internal.session import SessionManager
from harness.memory import Memory
from harness.notifier.base import AbstractNotifier
from harness.runners.base import AbstractRunner
from harness.runners.claude_cli import ClaudeCliRunner
from harness.storage.sql import SQLStorage
from harness.task import (
    Dialogue,
    FunctionTask,
    LLMTask,
    Parallel,
    PipelineResult,
    PipelineStep,
    PollingTask,
    Result,
    ShellTask,
    TaskConfig,
)


def _schema_class_path(task: object) -> str | None:
    """返回 task.output_schema 的完整限定名，用于数据库存储。

    若 task 无 output_schema 属性或值为 None，则返回 None。
    """
    schema = getattr(task, "output_schema", None)
    if schema is None:
        return None
    return f"{schema.__module__}.{schema.__qualname__}"


def _extract_run_summary(results: list[Result]) -> str:
    """run summary 提取规则。

    优先取最后一个成功 Task 的 output.summary 或 output['summary']，
    没有则取 str(output)[:300]，Task 失败时写入错误信息。
    """
    for r in reversed(results):
        if r.success:
            output = r.output
            # 尝试获取 output.summary（Pydantic model）
            if hasattr(output, "summary") and output.summary is not None:
                return str(output.summary)
            # 尝试获取 output['summary']（dict）
            if isinstance(output, dict) and "summary" in output:
                return str(output["summary"])
            return str(output)[:300]

    # 所有 task 都失败
    for r in reversed(results):
        if not r.success:
            return f"Task {r.task_index} [{r.task_type}] 失败：{r.error}"

    return ""


class Harness:
    """AI-native 通用自动化流水线框架主类。

    Args:
        project_path: 项目根目录。
        runner: LLMTask 的默认 runner，None 时使用 ClaudeCliRunner()。
        system_prompt: 全局 system_prompt，对所有 LLMTask 生效。
        storage_url: 存储 URL，None 时自动拼 SQLite 路径。
        memory: Memory 配置，None 时不启用记忆注入。
        notifier: 通知器，None 时不发送通知。
        stream_callback: 接收解析后文本片段的回调（与 raw_stream_callback 互斥）。
        raw_stream_callback: 接收原始 event dict 的回调（与 stream_callback 互斥）。
        default_config: 全局默认 TaskConfig。
        env_overrides: 运行 FunctionTask/ShellTask 前覆写的环境变量。
            空字符串值表示清除该变量（如 {'HTTP_PROXY': ''} 清除代理）。
    """

    def __init__(
        self,
        project_path: str,
        *,
        runner: AbstractRunner | None = None,
        system_prompt: str = "",
        storage_url: str | None = None,
        memory: Memory | None = None,
        notifier: AbstractNotifier | None = None,
        stream_callback: Callable[[str], None] | None = None,
        raw_stream_callback: Callable[[dict], None] | None = None,
        default_config: TaskConfig = TaskConfig(),
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        if stream_callback is not None and raw_stream_callback is not None:
            raise ValueError(
                "stream_callback and raw_stream_callback are mutually exclusive. "
                "Set only one of them."
            )

        self._project_path = Path(project_path).resolve()
        self._runner = runner or ClaudeCliRunner()
        self._system_prompt = system_prompt
        self._storage_url = storage_url or (
            f"sqlite+aiosqlite:///{self._project_path}/.harness/harness.db"
        )
        self._memory = memory
        self._notifier = notifier
        self._stream_callback = stream_callback
        self._raw_stream_callback = raw_stream_callback
        self._default_config = default_config
        self._env_overrides: dict[str, str] = env_overrides or {}

        self._storage: SQLStorage | None = None
        self._scheduler = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """幂等初始化：创建目录、追加 .gitignore、创建数据库表。"""
        if self._initialized:
            return

        # 创建 .harness/ 目录
        harness_dir = self._project_path / ".harness"
        harness_dir.mkdir(parents=True, exist_ok=True)

        # 追加 .gitignore（幂等）
        gitignore_path = self._project_path / ".gitignore"
        entry = ".harness/harness.db\n"
        if gitignore_path.exists():
            content = gitignore_path.read_text(encoding="utf-8")
            if ".harness/harness.db" not in content:
                with gitignore_path.open("a", encoding="utf-8") as f:
                    f.write(entry)
        else:
            gitignore_path.write_text(entry, encoding="utf-8")

        # 初始化存储
        self._storage = SQLStorage(self._storage_url)
        await self._storage.init()

        self._initialized = True

    async def run(
        self,
        prompt: str | Callable,
        *,
        output_schema: type | None = None,
        config: TaskConfig | None = None,
    ) -> Result:
        """单次 LLMTask 调用（pipeline 的语法糖）。"""
        pipeline_result = await self.pipeline(
            [LLMTask(prompt=prompt, output_schema=output_schema, config=config)]
        )
        return pipeline_result.results[0]

    async def pipeline(
        self,
        tasks: list[PipelineStep],
        *,
        name: str | None = None,
        resume_from: str | None = None,
    ) -> PipelineResult:
        """执行多步流水线。

        Args:
            tasks: 任务列表。
            name: 可选名称，存入 runs 表。
            resume_from: 从指定 run_id 续跑，跳过已成功的步骤。

        Returns:
            PipelineResult 包含所有步骤结果。

        Raises:
            InvalidPipelineError: 存在嵌套 Parallel 时。
            TaskFailedError: 任何步骤超过 max_retries 时。
        """
        await self._ensure_initialized()

        # 入口校验：嵌套 Parallel
        for i, task in enumerate(tasks):
            if isinstance(task, Parallel):
                for subtask in task.tasks:
                    if isinstance(subtask, Parallel):
                        raise InvalidPipelineError(
                            f"Nested Parallel is not supported. "
                            f"Found at index {i}."
                        )

        run_id = str(uuid.uuid4())
        assert self._storage is not None

        await self._storage.save_run(run_id, str(self._project_path), name)

        # 获取 resume 信息
        skipped_indices: set[str] = set()
        resumed_results: list[Result] = []

        if resume_from is not None:
            # 获取所有日志（含失败）用于 Parallel 原子性判断
            all_prev_logs = await self._storage.get_task_logs(resume_from)
            # 只取成功日志用于顺序步骤跳过
            success_prev_logs = [l for l in all_prev_logs if l["success"]]

            # 顺序步骤：只有成功的才跳过
            for log in success_prev_logs:
                idx = log["task_index"]
                if "." not in idx:
                    skipped_indices.add(idx)

            # Parallel 块：必须所有子 task 都成功才能跳过（原子性）
            parallel_outer: dict[str, list] = {}
            for log in all_prev_logs:
                idx = log["task_index"]
                if "." in idx:
                    outer = idx.split(".")[0]
                    if outer not in parallel_outer:
                        parallel_outer[outer] = []
                    parallel_outer[outer].append(log)

            for outer, logs in parallel_outer.items():
                if all(l["success"] for l in logs):
                    skipped_indices.add(outer)
                # 否则 Parallel 块整体重跑，不加入 skipped_indices

            # 构建续跑的初始 results（只含跳过的步骤的成功结果）
            for log in success_prev_logs:
                idx = log["task_index"]
                if idx in skipped_indices:
                    task_type = log.get("task_type", "function")
                    output_raw = log.get("output")
                    schema_class_path = log.get("output_schema_class")
                    deserialized = deserialize_output(output_raw, schema_class_path)
                    resumed_results.append(
                        Result(
                            task_index=idx,
                            task_type=task_type,
                            output=deserialized,
                            raw_text=log.get("raw_text"),
                            tokens_used=log.get("tokens_used", 0),
                            duration_seconds=log.get("duration_seconds", 0.0),
                            success=True,
                            error=None,
                        )
                    )

        # 找到最后一个顶层 LLMTask 的索引，用于 memory consolidation 注入
        last_llm_index: int | None = None
        for _i in range(len(tasks) - 1, -1, -1):
            if isinstance(tasks[_i], LLMTask):
                last_llm_index = _i
                break

        session_manager = SessionManager()
        if resume_from is not None:
            session_manager.reset()

        results: list[Result] = list(resumed_results)
        total_tokens = sum(r.tokens_used for r in results)
        start_time = time.monotonic()

        # 构建 memory_injection
        memory_injection = ""
        if self._memory is not None:
            try:
                memory_injection = await self._memory.build_injection(
                    self._storage, self._project_path
                )
            except Exception as e:
                logger.warning(
                    f"Memory injection failed, proceeding without it: {e}"
                )

        try:
            for outer_index, task in enumerate(tasks):
                task_index = str(outer_index)

                # 跳过已成功的步骤（续跑）
                if task_index in skipped_indices:
                    continue

                step_result: Result | list[Result]

                if isinstance(task, Parallel):
                    sub_results = await execute_parallel(
                        task,
                        outer_index,
                        results,
                        run_id,
                        harness_system_prompt=self._system_prompt,
                        harness_runner=self._runner,
                        harness_config=self._default_config,
                        session_manager=session_manager,
                        memory_injection=memory_injection,
                        storage=self._storage,
                        is_new_session=session_manager.is_broken,
                        harness_stream_callback=self._stream_callback,
                        harness_raw_stream_callback=self._raw_stream_callback,
                    )
                    for r in sub_results:
                        # 从 task_index（如 "2.1"）解析子任务索引以获取 schema
                        sub_idx_str = r.task_index.split(".", 1)[-1]
                        try:
                            sub_task = task.tasks[int(sub_idx_str)]
                            sub_schema_path = _schema_class_path(sub_task)
                        except (ValueError, IndexError):
                            sub_schema_path = None
                        await self._storage.save_task_log(
                            run_id,
                            r.task_index,
                            r.task_type,
                            output=r.output,
                            output_schema_class=sub_schema_path,
                            raw_text=r.raw_text,
                            tokens_used=r.tokens_used,
                            duration_seconds=r.duration_seconds,
                            success=r.success,
                            error=r.error,
                        )
                        total_tokens += r.tokens_used
                    results.extend(sub_results)
                    continue

                # 解析有效回调（Task 级覆写 Harness 级）
                effective_cb = task.stream_callback or self._stream_callback
                effective_raw_cb = task.raw_stream_callback or self._raw_stream_callback

                if isinstance(task, LLMTask):
                    # memory consolidation：末尾 LLMTask 无 schema 时注入整理提示
                    effective_sp = task.system_prompt
                    if (
                        self._memory is not None
                        and outer_index == last_llm_index
                        and task.output_schema is None
                    ):
                        consolidation = self._memory.consolidation_system_prompt(
                            self._project_path
                        )
                        effective_sp = "\n\n".join(
                            p for p in [task.system_prompt, consolidation] if p
                        )

                    # 注入回调
                    task_with_cb = LLMTask(
                        prompt=task.prompt,
                        system_prompt=effective_sp,
                        output_schema=task.output_schema,
                        runner=task.runner,
                        config=task.config,
                        stream_callback=effective_cb,
                        raw_stream_callback=effective_raw_cb,
                    )
                    r = await execute_llm_task(
                        task_with_cb,
                        task_index,
                        results,
                        run_id,
                        harness_system_prompt=self._system_prompt,
                        harness_runner=self._runner,
                        harness_config=self._default_config,
                        session_manager=session_manager,
                        memory_injection=memory_injection,
                        storage=self._storage,
                        is_new_session=session_manager.is_broken,
                    )
                    # 检查 memory_update
                    if (
                        self._memory is not None
                        and hasattr(r.output, "memory_update")
                        and r.output.memory_update is not None
                    ):
                        self._memory.write_memory_update(
                            self._project_path, r.output.memory_update
                        )

                elif isinstance(task, FunctionTask):
                    r = await execute_function_task(
                        task, task_index, results, run_id,
                        harness_config=self._default_config,
                        env_overrides=self._env_overrides,
                    )
                elif isinstance(task, ShellTask):
                    r = await execute_shell_task(
                        task, task_index, results, run_id,
                        harness_config=self._default_config,
                        env_overrides=self._env_overrides,
                    )
                elif isinstance(task, PollingTask):
                    r = await execute_polling(
                        task, task_index, results, run_id,
                        harness_config=self._default_config,
                    )
                elif isinstance(task, Dialogue):
                    r = await execute_dialogue(
                        dialogue=task,
                        outer_index=outer_index,
                        pipeline_results=results,
                        run_id=run_id,
                        harness_system_prompt=self._system_prompt,
                        harness_runner=self._runner,
                        harness_config=self._default_config,
                        storage=self._storage,
                    )
                else:
                    raise TypeError(f"Unknown task type: {type(task)}")

                await self._storage.save_task_log(
                    run_id,
                    r.task_index,
                    r.task_type,
                    output=r.output,
                    output_schema_class=_schema_class_path(task),
                    raw_text=r.raw_text,
                    tokens_used=r.tokens_used,
                    duration_seconds=r.duration_seconds,
                    success=r.success,
                    error=r.error,
                )
                total_tokens += r.tokens_used
                results.append(r)

        except TaskFailedError as e:
            # 写失败状态
            total_duration = time.monotonic() - start_time
            summary = f"Task {e.task_index} [{e.task_type}] 失败：{e.error}"
            await self._storage.update_run(
                run_id,
                status="failed",
                total_tokens=total_tokens,
                summary=summary,
                error=e.error,
            )
            if self._notifier is not None:
                try:
                    await self._notifier.notify(
                        title=name or "Pipeline Failed",
                        body=summary,
                        success=False,
                    )
                except Exception as e:
                    logger.warning(
                        f"Notifier failed to send notification: {e}"
                    )
            raise

        total_duration = time.monotonic() - start_time
        summary = _extract_run_summary(results)
        await self._storage.update_run(
            run_id,
            status="success",
            total_tokens=total_tokens,
            summary=summary,
        )

        if self._notifier is not None:
            try:
                await self._notifier.notify(
                    title=name or "Pipeline Completed",
                    body=summary,
                    success=True,
                )
            except Exception as e:
                logger.warning(
                    f"Notifier failed to send notification: {e}"
                )

        return PipelineResult(
            run_id=run_id,
            name=name,
            results=results,
            total_tokens=total_tokens,
            total_duration_seconds=total_duration,
        )

    def schedule(
        self,
        tasks: list[PipelineStep],
        *,
        cron: str,
        name: str | None = None,
    ) -> None:
        """添加定时 pipeline 任务。

        Args:
            tasks: 要定时执行的任务列表。
            cron: Cron 表达式。
            name: 可选名称。
        """
        if self._scheduler is None:
            from harness.scheduler.apscheduler import APSchedulerBackend
            self._scheduler = APSchedulerBackend()

        async def _run() -> None:
            await self.pipeline(tasks, name=name)

        self._scheduler.add_job(_run, cron)

    async def start(self) -> None:
        """启动调度器。"""
        await self._ensure_initialized()
        if self._scheduler is not None:
            await self._scheduler.start()

    async def stop(self) -> None:
        """停止调度器。"""
        if self._scheduler is not None:
            await self._scheduler.stop()

    async def __aenter__(self) -> "Harness":
        await self._ensure_initialized()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop()

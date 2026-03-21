# Harness 实现计划

**设计文档**：`design/v1-design.md`
**最后更新**：2026-03-21

状态标记：`[ ]` 待实现 · `[x]` 已完成 · `[-]` 进行中

---

## 阶段 1：基础层（无外部依赖）

> 目标：所有核心数据结构就位，可以被后续阶段引用。

### harness/_internal/exceptions.py

- [x] `TaskFailedError(run_id, task_index, task_type, error, partial_results)`
- [x] `ClaudeNotFoundError`
- [x] `InvalidPipelineError`（嵌套 Parallel 时抛出）
- [x] `OutputSchemaError`（FunctionTask 校验失败，不触发重试）

### harness/task.py

- [x] `BaseTask`：`config: TaskConfig | None`，`stream_callback: Callable[[str], None] | None`，`raw_stream_callback: Callable[[dict], None] | None`（两者互斥，同时设置抛 `ValueError`）
- [x] `LLMTask(BaseTask)`：`prompt: str | Callable[[list[Result]], str]`，`system_prompt: str = ""`，`output_schema: type[BaseModel] | None`，`runner: AbstractRunner | None`
- [x] `FunctionTask(BaseTask)`：`fn: Callable[[list[Result]], Any]`，`output_schema: type[BaseModel] | None`；校验失败抛 `OutputSchemaError`，**不触发重试**
- [x] `ShellTask(BaseTask)`：`cmd: str | Callable[[list[Result]], str]`，`cwd: str | None`，`env: dict[str, str] | None`；文档注明命令注入风险
- [x] `PollingTask(BaseTask)`：`submit_fn: Callable[[list[Result]], Any]`，`poll_fn: Callable[[Any], Any]`，`success_condition: Callable[[Any], bool]`，`failure_condition: Callable[[Any], bool] | None`，`poll_interval: float = 10.0`，`timeout: int = 900`，`output_schema: type[BaseModel] | None`
- [x] `Parallel(BaseTask)`：`tasks: list[LLMTask | FunctionTask | ShellTask | PollingTask]`（类型不含 Parallel，运行时再校验），`error_policy: Literal["all_or_nothing", "best_effort"] = "all_or_nothing"`
- [x] `TaskConfig`（frozen dataclass）：`timeout: int = 3600`，`max_retries: int = 2`，`backoff_base: float = 2.0`；**不含 PermissionMode**
- [x] `Result`（frozen dataclass）：`task_index: str`（字符串路径），`task_type: Literal["llm","function","shell","polling"]`，`output: BaseModel | str | Any`，`raw_text: str | None`，`tokens_used: int`，`duration_seconds: float`，`success: bool`，`error: str | None`
- [x] `PipelineResult`（frozen dataclass）：`run_id: str`，`name: str | None`，`results: list[Result]`，`total_tokens: int`，`total_duration_seconds: float`
- [x] `PipelineStep = LLMTask | FunctionTask | ShellTask | PollingTask | Parallel`
- [x] `Task` 向后兼容别名：包装函数，调用时发出 `DeprecationWarning("Task is deprecated, use LLMTask instead.")`

### harness/runners/base.py

- [x] `AbstractRunner(ABC)`：`execute(prompt: str, *, system_prompt: str, session_id: str | None, **kwargs) -> RunnerResult`
- [x] `RunnerResult`（dataclass）：`text: str`，`tokens_used: int`，`session_id: str | None`

### harness/storage/base.py

- [x] `StorageProtocol(Protocol)`：`save_run / update_run / save_task_log / get_run / list_runs`

### harness/storage/models.py

- [x] `runs` 表：`id, project_path, name VARCHAR, started_at, completed_at, status, total_tokens, summary, error`
- [x] `task_logs` 表：`id, run_id, task_index VARCHAR, task_type, prompt_preview, output, raw_text, tokens_used, duration_seconds, attempt, success, error, created_at`
- [x] `task_index` 列为 VARCHAR，存字符串路径（`"0"`, `"2.0"`, `"2.1"`）

### 测试

- [x] `tests/unit/test_task.py`：每种 Task 类型实例化；DeprecationWarning；Parallel 构造时 tasks 含 Parallel 的运行时校验；BaseTask stream_callback/raw_stream_callback 互斥
- [x] `tests/unit/test_exceptions.py`：各异常的字段

---

## 阶段 2：执行核心（依赖阶段 1）

> 目标：LLMTask / FunctionTask / ShellTask 端到端可执行，结果可写入存储。

### harness/runners/claude_cli.py

- [x] `PermissionMode(StrEnum)`：`BYPASS="bypassPermissions"` / `DEFAULT="default"` / `ACCEPT_EDITS="acceptEdits"` / `DONT_ASK="dontAsk"` / `PLAN="plan"`
- [x] `ClaudeCliRunner(permission_mode: PermissionMode = PermissionMode.BYPASS)`
- [x] `execute()`：`asyncio.create_subprocess_exec`，非阻塞；逐行读 stdout 传给 stream_parser
- [x] CLI 参数组装：`--permission-mode --verbose --output-format stream-json --include-partial-messages --system-prompt [--json-schema] [--session-id] [--resume] -p`
- [x] 超时处理：`asyncio.wait_for` → 超时后 SIGTERM → 等 5s → SIGKILL
- [x] 环境变量清理（仅子进程 env）：移除 `CLAUDECODE / CLAUDE_CODE_ENTRYPOINT / CLAUDE_CODE / CLAUDE_SESSION / CLAUDE_API_KEY`，保留 `ANTHROPIC_API_KEY`
- [x] 启动检查：`which claude` 失败 → `ClaudeNotFoundError`；版本检查失败 → 打印警告继续

### harness/_internal/stream_parser.py

- [x] 逐行解析 JSON Lines（stream-json 格式）
- [x] 碰到 `result` event → 提取最终文本和 `usage.output_tokens`（映射到 `tokens_used`）
- [x] partial message text → 回调 `stream_callback(text: str)` 或 `raw_stream_callback(event: dict)`
- [x] 输出每个 Task 执行前的分隔标识：`=== Task {task_index} [{task_type}] ===`

### harness/_internal/session.py

- [x] `SessionManager`：为 pipeline 内所有 LLMTask 维护共享 `session_id`
- [x] 正常执行时所有 LLMTask 复用同一 session_id
- [x] 重试时生成新 session_id（UUID），后续 LLMTask 继续用新的
- [x] 提供 `current_session_id` 和 `mark_broken()` 方法（断开时记录，供 prompt 注入兜底判断）

### harness/_internal/executor.py

- [x] 单个 `PipelineStep` 执行入口，按 task_type 派发（Parallel 和 PollingTask 委托给各自模块）
- [x] **TaskConfig 三级合并**：`effective_config = task.config or harness.default_config or TaskConfig()`
- [x] **LLMTask 执行流程**：
  1. system_prompt 合并：`Harness.system_prompt + "\n\n" + LLMTask.system_prompt（若非空）+ memory_injection`
  2. 若 `LLMTask.prompt` 是 Callable 且抛异常 → **不重试**，直接抛 `TaskFailedError`
  3. `runner.execute(prompt, system_prompt=..., session_id=...) -> RunnerResult`
  4. 若有 `output_schema`：校验/解析；失败则追加错误信息到 prompt 重试
  5. 执行后检查 `result.output.memory_update`：非 None 时调用 `memory.write_memory_update()`
  6. 若为末尾 LLMTask 且 `output_schema=None` 且使用 ClaudeCliRunner：system_prompt 追加 memory.md 整理提示（兜底机制）
- [x] **FunctionTask 执行流程**：`fn(results)`；output_schema 校验失败抛 `OutputSchemaError`，**不触发重试**
- [x] **ShellTask 执行流程**：`asyncio.create_subprocess_shell`；捕获 stdout（作为 output）和 stderr（作为 error）；非零退出码触发重试
- [x] **重试逻辑**：循环最多 `max_retries` 次，等待 `backoff_base ^ retry_count` 秒（retry_count 从 0 开始），超过后抛 `TaskFailedError`
- [x] **session 断开时 prompt 注入兜底**：触发条件为新 session（重试或续跑）；从 storage 读取前序所有成功 Task 输出，格式化注入 prompt 头部：
  ```
  === 前序任务输出 ===
  Task 0 [llm]: {output.summary 或 raw_text[:300]}
  Task 1 [function]: {str(output)[:200]}
  ```

### harness/storage/sql.py

- [x] SQLAlchemy async 通用实现（SQLite / MySQL / PG）
- [x] SQLite 首次连接后执行 `PRAGMA journal_mode=WAL`
- [x] 实现 `StorageProtocol` 所有方法

### run summary 提取规则（在 harness.py 或 executor.py 中实现）

- [x] 优先取最后一个成功 Task 的 `output.summary` 或 `output["summary"]`（BaseModel 或 dict）
- [x] 没有则取 `str(output)[:300]`
- [x] Task 失败时写入：`"Task {task_index} [{task_type}] 失败：{error_message}"`

### 测试

- [x] `tests/unit/test_claude_cli.py`：mock subprocess；版本检查；环境变量清理；SIGTERM→SIGKILL 超时
- [x] `tests/unit/test_executor.py`：system_prompt 三级合并；TaskConfig 优先级；LLMTask prompt Callable 抛异常不重试；output_schema 校验失败重试（LLMTask）vs 不重试（FunctionTask）；ShellTask 非零退出码重试；session 断开 prompt 注入格式；memory_update 写入触发
- [x] `tests/unit/test_stream_parser.py`：JSON Lines 解析；result event 提取；分隔标识输出

---

## 阶段 3：扩展执行（依赖阶段 2）

> 目标：PollingTask / Parallel / AgentLeader / Memory 可用。

### harness/_internal/polling.py

- [x] `execute_polling(task: PollingTask, results: list[Result], config: TaskConfig) -> Result`
- [x] 流程：`submit_fn(results)` → 循环 `poll_fn(handle)` 每 `poll_interval` 秒
  - `success_condition(response)` → True：返回结果
  - `failure_condition(response)` → True：抛 `TaskFailedError(error="生成失败")`，触发 max_retries 重试（重试时重新 submit）
  - 超过 `timeout`：抛 `TaskFailedError(error="超时")`，触发 max_retries 重试
- [x] 支持 `output_schema` 对最终结果做校验

### harness/_internal/parallel.py

- [x] `execute_parallel(parallel: Parallel, outer_index: int, results: list[Result], ...) -> list[Result]`
- [x] 入口处校验 `parallel.tasks` 不含 `Parallel` 实例，违反时抛 `InvalidPipelineError`
- [x] `task_index` 生成：`f"{outer_index}.{inner_index}"`（字符串，inner_index 从 0 开始）
- [x] `asyncio.gather(*[execute(t) for t in parallel.tasks])` 并发执行
- [x] `error_policy="all_or_nothing"`：任一失败立刻 `cancel` 其余 task，抛 `TaskFailedError`
- [x] `error_policy="best_effort"`：等待全部完成，失败的标记 `success=False`，不抛异常

### harness/runners/agent_leader.py

- [x] `AgentLeader(agents: list[str], runner: ClaudeCliRunner | None = None)`；`runner=None` 时内部创建 `ClaudeCliRunner()`
- [x] `execute()`：在 `system_prompt` 末尾追加：`"\n\n可用 agent 白名单：{', '.join(agents)}。不要调用列表之外的 agent。"`
- [x] docstring 明确标注：**这是 best-effort 约束，非安全边界**

### harness/memory.py

- [x] `Memory(history_runs: int = 3, memory_file: str = ".harness/memory.md", max_tokens: int = 2000)`
- [x] `build_injection(storage: StorageProtocol, project_path: Path) -> str`：
  - 从 storage 读最近 `history_runs` 条 run summary
  - 读取 `memory_file` 内容（不存在时跳过）
  - 拼装格式（见下），超过 `max_tokens` 从头部截断
  - 注入格式：
    ```
    === 最近运行历史 ===
    2026-03-20: {summary}
    ...
    === 项目记忆 ===
    {memory.md 内容}
    ```
- [x] `write_memory_update(project_path: Path, memory_file: str, content: str) -> None`：将 `content` 追加写入 memory_file；写入后若文件超过 `max_tokens` 的 80%，仅做硬截断（v1 不做 LLM 压缩）

### 测试

- [x] `tests/unit/test_polling.py`：success 路径；failure_condition 路径；timeout 路径；重试（重新 submit）
- [x] `tests/unit/test_parallel.py`：all_or_nothing 失败后其余被 cancel；best_effort 部分失败；task_index 字符串生成（`"2.0"`,`"2.1"`）；嵌套 Parallel 抛 InvalidPipelineError
- [x] `tests/unit/test_memory.py`：注入格式；max_tokens 截断；write_memory_update 追加写入

---

## 阶段 4：集成层（依赖阶段 3）

> 目标：用户可以实例化 Harness 并运行完整 pipeline。

### harness/harness.py

- [x] `__init__(project_path, *, runner=None, system_prompt="", storage_url=None, memory=None, notifier=None, stream_callback=None, raw_stream_callback=None, default_config=TaskConfig())`
  - `stream_callback` 和 `raw_stream_callback` 互斥，同时设置抛 `ValueError`
  - `runner=None` 时默认 `ClaudeCliRunner()`
  - `storage_url=None` 时自动拼 `f"sqlite:///{project_path}/.harness/harness.db"`
- [x] **初始化副作用**（幂等）：
  1. 创建 `{project_path}/.harness/` 目录
  2. 在 `{project_path}/.gitignore` 追加 `.harness/harness.db`（检查是否已存在该行）
  3. 创建数据库表
- [x] `run(prompt, *, output_schema=None, config=None) -> Result`：单 LLMTask 语法糖
- [x] `pipeline(tasks, *, name=None, resume_from=None) -> PipelineResult`：
  - 入口校验嵌套 Parallel（抛 `InvalidPipelineError`）
  - `resume_from`：从 storage 读已成功的 `task_index`（字符串），跳过对应步骤；**Parallel 块原子处理**——Parallel 内任意子 task 未全部成功则整个 Parallel 块重跑
  - 执行前写 `save_run()`；每步完成后写 `save_task_log()`
  - TaskConfig 三级合并：`task.config or self.default_config or TaskConfig()`
  - run summary 提取（见阶段 2 规则），写入 `update_run(summary=...)`
  - 全部成功后发通知（notifier）
  - 任一步骤超过 max_retries → 写 storage + 发通知 + 抛 `TaskFailedError`
- [x] `schedule(tasks, *, cron, name=None) -> None`
- [x] `start() / stop()`
- [x] `__aenter__ / __aexit__`：支持 `async with Harness(...) as h:`，退出时调用 `stop()`

### harness/scheduler/base.py + apscheduler.py

- [x] `AbstractScheduler(ABC)`：`add_job(fn: Callable, cron: str) / start() / stop()`
- [x] `APSchedulerBackend`：APScheduler v4 集成

### harness/notifier/base.py + telegram.py

- [x] `AbstractNotifier(ABC)`：`async notify(title: str, body: str, *, success: bool) -> None`
- [x] `TelegramNotifier(bot_token: str, chat_id: str)`：httpx 异步发送；成功 ✅，失败 ❌

### harness/__init__.py

- [x] 导出：`Harness, LLMTask, FunctionTask, ShellTask, PollingTask, Parallel, Task, Result, PipelineResult, TaskConfig, Memory`
- [x] `PermissionMode` **不**在顶层导出（避免暴露 Claude 专用概念），从 `harness.runners.claude_cli` 单独导入

### 测试

- [x] `tests/unit/test_harness.py`：stream_callback/raw_stream_callback 互斥；pipeline 流程（mock executor）；resume_from 跳过逻辑；Parallel 原子续跑语义；async context manager；.gitignore 追加幂等性
- [x] `tests/unit/test_notifier.py`：TelegramNotifier mock httpx 请求

---

## 阶段 5：交付（依赖阶段 4）

> 目标：可发布、可验证、有示例。

### harness/cli.py

- [x] `harness runs`：列出最近 N 条（run_id, name, 时间, 状态, task 数, tokens）
- [x] `harness runs --failed`：只显示失败
- [x] `harness migrate --to <dsn>`：SQLite → MySQL/PG 数据迁移

### pyproject.toml

- [x] 依赖：`sqlalchemy[asyncio] >= 2, apscheduler >= 4, pydantic >= 2, aiosqlite, httpx`
- [x] Python `>= 3.11`；入口点 `harness = "harness.cli:app"`；Apache 2.0

### 集成测试

- [x] `tests/integration/test_pipeline.py`：
  - 顺序 pipeline（FunctionTask + ShellTask，不需要 Claude CLI）
  - PollingTask mock（success / failure_condition / timeout 三路径）
  - Parallel all_or_nothing：一个失败，其余被 cancel
  - Parallel best_effort：部分失败，PipelineResult 包含 success=False 的 Result
  - resume_from：模拟 Task 1 失败，续跑从 Task 1 重试；Parallel 块部分失败时整体重跑
- [x] `tests/integration/test_storage.py`：task_index VARCHAR 存取；runs.name 字段；run summary 提取

### IterationForge 集成验证

- [x] `pip install -e /path/to/harness` 在 IterationForge 引用
- [x] 将原调用替换为 `LLMTask`；用 `Task` 别名验证 DeprecationWarning 触发
- [x] 跑一次完整流程，行为与原版一致

### 示例

- [x] `examples/video_pipeline.py`：`LLMTask + Parallel[PollingTask×2] + FunctionTask + ShellTask`，mock submit_fn/poll_fn
- [x] `examples/iterationforge_style.py`：定时扫描+修复典型用法

---

## 关键设计决策速查

| 事项 | 决策 |
|------|------|
| task_index 类型 | `str`；顺序：`"0"/"1"`；Parallel 内：`"2.0"/"2.1"` |
| PermissionMode 位置 | `ClaudeCliRunner` 构造函数；不在 `TaskConfig`；不在顶层导出 |
| system_prompt 合并 | 追加：`Harness.sp + "\n\n" + Task.sp（非空时）+ memory_injection` |
| submit_fn 签名 | `Callable[[list[Result]], Any]`（与 FunctionTask.fn 一致） |
| stream_callback | 拆为 `stream_callback(str)` 和 `raw_stream_callback(dict)`，互斥；BaseTask 同样有两个字段 |
| Parallel 继承 | 继承 `BaseTask`，有 `config` 和回调字段 |
| FunctionTask 校验失败 | 抛 `OutputSchemaError`，**不重试** |
| LLMTask prompt Callable 异常 | **不重试**，直接抛 `TaskFailedError` |
| TaskConfig 三级合并 | `task.config > harness.default_config > TaskConfig()` |
| memory.md 更新主路径 | 框架检查 `output.memory_update` 字段后写入 |
| memory.md 兜底路径 | 末尾 LLMTask 无 schema + ClaudeCliRunner → system_prompt 追加整理提示 |
| memory_update 调用时机 | LLMTask 执行完成后，在 executor 中检查并调用 `memory.write_memory_update()` |
| run summary 提取 | 优先 `output.summary`，没有则 `str(output)[:300]`，失败时写错误信息 |
| stream 分隔标识 | 每个 Task 执行前输出 `=== Task {index} [{type}] ===` |
| Parallel 续跑语义 | 原子单元；块内任意子 task 未成功则整体重跑 |
| .gitignore 追加 | 初始化时幂等追加 `harness.db`（检查是否已存在该行） |
| pipeline/schedule name | 可选参数，存入 `runs.name` 列，也存入 `PipelineResult.name` |
| Task 别名 | 包装函数 + `DeprecationWarning`，v2 移除 |
| Parallel 嵌套禁止 | 运行时校验（类型声明 + 入口 assert），抛 `InvalidPipelineError` |
| AgentLeader runner 参数 | `runner: ClaudeCliRunner | None = None`，None 时内部创建默认实例 |

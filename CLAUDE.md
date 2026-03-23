# Harness

> Claude CLI 优先、开箱即用、核心可覆写的 AI Workflow 框架

## 项目状态

**阶段：v1 已实现**（275 tests pass，2026-03-23）
设计文档：`design/v1-design.md`

## 背景

从 IterationForge（`~/ai/projects/iterationforge`）提取通用编排能力，作为独立开源项目。
IterationForge 将成为 Harness 的第一个使用方。

## 核心定位

- **Claude CLI 优先**：调用 Claude Code CLI 子进程（asyncio），支持完整工具链、session 持久化、bypassPermissions
- **Batteries-included but overridable**：内置 SQLite 存储、APScheduler v4 调度、Telegram 通知，每个组件都有抽象接口可替换
- **用户只定义业务**：声明 Task、组合 pipeline，基础设施由框架处理

## 公开 API

```python
from harness import (
    Harness,
    LLMTask, FunctionTask, ShellTask, PollingTask, Parallel,
    Dialogue, Role,         # 多角色辩论循环
    Task,                   # LLMTask 的已废弃别名（v2 移除）
    Result, PipelineResult,
    TaskConfig, Memory,
    # Runner
    AbstractRunner, RunnerResult,   # 自定义 runner 基类
    OpenAIRunner,                   # OpenAI-compatible API（MiniMax、DeepSeek 等）
    AnthropicRunner,                # Anthropic Messages API（非 CLI）
)
from harness.runners.claude_cli import PermissionMode  # 不在顶层导出

h = Harness(project_path="/path/to/project")

# 单次 LLM 调用
result = await h.run("分析并修复代码质量问题")

# 多步混合流水线
results = await h.pipeline([
    FunctionTask(fn=collect_data),
    LLMTask("分析数据，给出优化建议"),
    Parallel([
        PollingTask(submit_fn=..., poll_fn=..., success_condition=...),
        ShellTask(cmd="pytest tests/"),
    ]),
])

# 定时运行
h.schedule(tasks=[...], cron="0 2 * * *")
await h.start()
```

## 包结构

```
harness/
  __init__.py          # 公开 API 导出
  harness.py           # Harness 主类（用户入口）
  task.py              # 所有 Task 类型 + TaskConfig + Result + PipelineResult
  memory.py            # Memory（历史运行注入 + memory.md）

  runners/
    base.py            # AbstractRunner + RunnerResult
    claude_cli.py      # ClaudeCliRunner + PermissionMode
    openai.py          # OpenAIRunner（OpenAI-compatible，含 MiniMax/DeepSeek 等）
    anthropic.py       # AnthropicRunner（Anthropic Messages API）
    agent_leader.py    # AgentLeader（白名单约束 runner）

  storage/
    base.py            # StorageProtocol（Protocol）
    sql.py             # SQLAlchemy async 实现（SQLite/MySQL/PG）
    models.py          # ORM 模型（runs + task_logs）

  scheduler/
    base.py            # AbstractScheduler
    apscheduler.py     # APScheduler v4 后端（延迟注册模式）

  notifier/
    base.py            # AbstractNotifier
    telegram.py        # TelegramNotifier（httpx）

  _internal/
    executor.py        # Task 派发 + 重试 + session 管理 + prompt 注入
    parallel.py        # Parallel 并发执行（asyncio.gather + error_policy）
    polling.py         # PollingTask 轮询循环
    dialogue.py        # Dialogue 多角色执行 + DialogueContext
    stream_parser.py   # Claude stream-json 逐行解析
    session.py         # SessionManager（pipeline 内 session 共享）
    exceptions.py      # TaskFailedError / ClaudeNotFoundError / ...

  cli.py               # harness runs / harness migrate（typer + rich）
```

## 关键设计决策

| 事项 | 决策 |
|------|------|
| 异步模型 | asyncio 优先，所有 I/O 非阻塞 |
| 默认存储 | SQLite + WAL，路径 `{project_path}/.harness/harness.db` |
| 默认权限 | `PermissionMode.BYPASS`（无人值守） |
| `run()` | `pipeline([single_LLMTask])` 语法糖 |
| task_index | 字符串，顺序：`"0"/"1"`，Parallel 内：`"2.0"/"2.1"` |
| Session 策略 | pipeline 内 LLMTask 共享 session；重试/续跑时生成新 session，注入前序输出兜底 |
| output_schema | `type[BaseModel]`（LLMTask/PollingTask）或 `type`（FunctionTask isinstance 校验） |
| TaskConfig 优先级 | `task.config > harness.default_config > TaskConfig()` |
| system_prompt 合并 | `Harness.sp + Task.sp（非空）+ memory_injection`，`"\n\n"` 分隔 |
| FunctionTask 校验失败 | 抛 `OutputSchemaError`，**不触发重试** |
| LLMTask prompt callable 异常 | **不触发重试**，直接抛 `TaskFailedError` |
| Parallel 续跑 | 原子单元：块内任意子 task 未成功则整体重跑 |
| APScheduler v4 | 延迟注册模式：`add_job()` 缓存，`start()` 批量 `await add_schedule()` |
| ClaudeCliRunner 超时 | `executor.py` 用 `asyncio.wait_for` 触发取消，runner 内 `except CancelledError` → SIGTERM → 5s → SIGKILL |

## 已知问题（待修复）

### 设计文档偏差

- `design/v1-design.md` Section 11 写 `--session-id`，实际 Claude CLI 使用 `--resume`（见 `claude_cli.py:131`）
- ~~memory.md 兜底整理机制~~（已实现：`Memory.consolidation_system_prompt()`，末尾 LLMTask 无 schema 时自动注入整理提示）

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 含覆盖率
pytest tests/ --cov=harness --cov-report=term-missing
```

测试分布：
- `tests/unit/`：exceptions, executor, harness, memory, parallel, polling, stream_parser, task
- `tests/integration/`：pipeline（端到端，不需要 Claude CLI）、storage（SQLite 读写）

## 示例

```bash
uv run python examples/code_stats/main.py               # 统计代码量（FunctionTask + LLMTask + ShellTask）
uv run python examples/analyze_harness/main.py          # 分析 → 优化 → 复盘（三阶段 pipeline）
uv run python examples/research_report/main.py Clawith  # 联网调研报告（多 LLMTask + FunctionTask）
uv run python examples/video_pipeline/main.py           # LLMTask + Parallel[PollingTask×2] + FunctionTask
```

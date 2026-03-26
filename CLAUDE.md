# Harness

> Claude CLI 优先、开箱即用、核心可覆写的 AI Workflow 框架

## 项目状态

**阶段：v2.1 已实现**（440 tests pass，2026-03-26）
设计文档：`design/v1-design.md`（v1）、`design/v2-architecture.md`（v2 蓝图）、`design/v2-agent-service-plan.md`（v2.1 方案）

v2.0 新增：
- `State` 共享状态（替代 `results[N]`）
- `output_key` 自动写入 state
- `Condition` 条件分支 / `Loop` 循环
- `tasks/` 目录拆分（v1 import 路径仍可用）
- v1 callable 签名 `fn(results)` 自动兼容

v2.1 新增：
- `Agent` class-based 持久化角色（对齐 CrewAI/AutoGen/ADK）
- `Agent.run()` 独立执行 / `Agent.as_role()` 降级为 Dialogue Role
- `CronTrigger` / `EventTrigger` 触发器
- `h.service(name, triggers, handler)` 长驻服务模式
- `h.emit(event, data)` 事件发射
- EventBus（pyee 可选依赖）

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
    Agent,                  # v2.1: class-based 持久化角色
    State,                  # v2: 共享状态基类
    CronTrigger, EventTrigger, TriggerContext,  # v2.1: 触发器
    Condition, Loop,        # v2: 流程控制
    LLMTask, FunctionTask, ShellTask, PollingTask, Parallel,
    Dialogue, Role,         # 多角色辩论循环
    DialogueProgressEvent,  # Dialogue.progress_callback 接收的结构化事件
    Task,                   # LLMTask 的已废弃别名（v2 移除）
    Result, PipelineResult,
    TaskConfig, Memory,
    result_by_type,         # 按 task_type 从 results 取结果，替代 results[N]
    # Runner
    AbstractRunner, RunnerResult,   # 自定义 runner 基类
    OpenAIRunner,                   # OpenAI-compatible API（MiniMax、DeepSeek 等）
    AnthropicRunner,                # Anthropic Messages API（非 CLI）
)
from harness.runners.claude_cli import PermissionMode  # 不在顶层导出

h = Harness(project_path="/path/to/project")

# 单次 LLM 调用
result = await h.run("分析并修复代码质量问题")

# v2.1: Agent 独立使用
analyst = Agent(name="analyst", system_prompt="技术分析师", runner=my_runner)
text = await analyst.run("分析今日走势")

# v2.1: Agent 在 Dialogue 中使用
trader = Agent(name="trader", system_prompt="短线交易员", runner=my_runner)
dialogue = Dialogue(
    roles=[
        analyst.as_role(lambda ctx: f"分析: {ctx.state.market}"),
        trader.as_role(lambda ctx: f"回应: {ctx.last_from('analyst')}"),
    ],
    max_rounds=3,
)

# v2: State 模式 pipeline（推荐）
class MyState(State):
    analysis: str = ""
    report: str = ""

pr = await h.pipeline([
    FunctionTask(fn=lambda state: state._set_output("data", fetch()), output_key="data"),
    LLMTask("分析数据", output_key="analysis"),
    Condition(
        check=lambda state: len(state.analysis) > 100,
        if_true=[LLMTask("生成详细报告", output_key="report")],
        if_false=[FunctionTask(fn=lambda state: "简短报告", output_key="report")],
    ),
    Loop(
        body=[LLMTask("优化报告", output_key="report")],
        until=lambda state: quality_ok(state.report),
        max_iterations=3,
    ),
], state=MyState())

# v2.1: Service 长驻模式
async def on_trigger(ctx: TriggerContext):
    return [LLMTask("处理事件", output_key="result")]

h.service("my-service", triggers=[
    CronTrigger("15 15 * * 1-5"),
    EventTrigger("alert", filter=lambda d: d["level"] > 3),
], handler=on_trigger)
await h.start()

# 外部事件推送
await h.emit("alert", {"level": 5, "msg": "异常检测"})

# v1: results 模式 pipeline（向后兼容）
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
  harness.py           # Harness 主类（pipeline/service/start/stop）
  agent.py             # Agent class-based 角色（v2.1 新增）
  state.py             # State 共享状态基类（v2 新增）
  triggers.py          # CronTrigger, EventTrigger, TriggerContext（v2.1 新增）
  task.py              # re-export shim → harness.tasks（v1 兼容）
  memory.py            # Memory（历史运行注入 + memory.md）

  tasks/               # v2 新增：拆分的任务模块
    __init__.py        # 统一导出
    config.py          # TaskConfig
    result.py          # Result, PipelineResult, result_by_type
    base.py            # BaseTask, DialogueProgressEvent
    llm.py             # LLMTask（含 output_key）
    function.py        # FunctionTask（含 output_key）
    shell.py           # ShellTask（含 output_key）
    polling.py         # PollingTask（含 output_key）
    parallel.py        # Parallel
    dialogue.py        # Dialogue, Role, DialogueTurn, DialogueOutput
    condition.py       # Condition（v2 新增）
    loop.py            # Loop（v2 新增）
    types.py           # PipelineStep union 类型别名

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
    compat.py          # v1/v2 callable 双模式检测（v2 新增）
    condition.py       # Condition 执行逻辑（v2 新增）
    loop.py            # Loop 执行逻辑（v2 新增）
    task_index.py      # TaskIndex 结构化索引（含 cond/loop 格式）
    parallel.py        # Parallel 并发执行（asyncio.gather + error_policy）
    polling.py         # PollingTask 轮询循环
    dialogue.py        # Dialogue 多角色执行 + DialogueContext
    event_bus.py       # EventBus（pyee 封装，v2.1 新增）
    service.py         # ServiceRunner 服务执行核心（v2.1 新增）
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
| task_index | 字符串，顺序：`"0"/"1"`，Parallel：`"2.0"/"2.1"`，Condition：`"3.c0"/"3.f0"`，Loop：`"4.i2.1"` |
| State 模式 | `pipeline(state=State())` 启用共享状态；v1 callable `fn(results)` 自动兼容 |
| output_key | 所有 Task 类型支持 `output_key="field"`，执行后自动 `state._set_output(key, output)` |
| Condition/Loop | 不能嵌入 Parallel 内部（`InvalidPipelineError`），可包含任意其他 PipelineStep |
| Session 策略 | pipeline 内 LLMTask 共享 session；重试/续跑时生成新 session，注入前序输出兜底 |
| output_schema | `type[BaseModel]`（LLMTask/PollingTask）或 `type`（FunctionTask isinstance 校验） |
| TaskConfig 优先级 | `task.config > harness.default_config > TaskConfig()` |
| system_prompt 合并 | `Harness.sp + Task.sp（非空）+ memory_injection`，`"\n\n"` 分隔 |
| FunctionTask 校验失败 | 抛 `OutputSchemaError`，**不触发重试** |
| LLMTask prompt callable 异常 | **不触发重试**，直接抛 `TaskFailedError` |
| Parallel 续跑 | 原子单元：块内任意子 task 未成功则整体重跑 |
| APScheduler v4 | 延迟注册模式：`add_job()` 缓存，`start()` 批量 `await add_schedule()` |
| ClaudeCliRunner 超时 | `executor.py` 用 `asyncio.wait_for` 触发取消，runner 内 `except CancelledError` → SIGTERM → 5s → SIGKILL |
| Agent 定位 | 构建块（不是 PipelineStep），通过 `as_role()` 降级到 Dialogue，通过 FunctionTask 进入 pipeline |
| Service 模式 | `h.service(name, triggers, handler)` 注册；handler 返回 pipeline steps，复用 `h.pipeline()` 执行 |
| EventBus | pyee 可选依赖（`harness-ai[service]`），仅在有 EventTrigger 时懒初始化 |
| Service 错误隔离 | 单次触发执行失败（handler 异常/pipeline 异常）不中断服务 |

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
- `tests/unit/`：exceptions, executor, harness, memory, parallel, polling, stream_parser, task, state, compat, state_executor, task_index, condition, loop, tasks_split, agent, triggers, event_bus, service_runner
- `tests/integration/`：pipeline, storage, state_pipeline, agent_pipeline, service

## 示例

```bash
uv run python examples/code_stats/main.py               # 统计代码量（FunctionTask + LLMTask + ShellTask）
uv run python examples/analyze_harness/main.py          # 分析 → 优化 → 复盘（三阶段 pipeline）
uv run python examples/research_report/main.py Clawith  # 联网调研报告（多 LLMTask + FunctionTask）
uv run python examples/video_pipeline/main.py           # LLMTask + Parallel[PollingTask×2] + FunctionTask
```

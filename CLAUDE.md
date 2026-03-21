# Harness

> Claude CLI 优先、开箱即用、核心可覆写的 AI Workflow 框架

## 项目状态

**阶段：v1 已实现**（122 tests pass）
设计文档：`design/v1-design.md`

## 背景

从 IterationForge（`~/ai/projects/iterationforge`）提取通用编排能力，作为独立开源项目。
IterationForge 将成为 Harness 的第一个使用方。

## 核心定位

- **Claude CLI 优先**：调用 Claude Code CLI 子进程（asyncio），支持完整工具链、session 持久化、bypassPermissions
- **Batteries-included but overridable**：内置 SQLite 存储、APScheduler v4 调度、Telegram 通知，每个组件都有抽象接口可替换
- **用户只定义业务**：声明 Task、组合 pipeline，基础设施由框架处理

## 2 个核心概念

```python
from harness import Harness, Task

h = Harness(project_path="/path/to/project")

# 单次调用
result = await h.run("分析并修复代码质量问题")

# 多步流水线
results = await h.pipeline([
    Task("扫描项目，输出问题列表", output_schema=ScanResult),
    Task(lambda ctx: f"修复问题：{ctx[0].output.issues}"),
    Task("运行测试，验证结果"),
])

# 定时运行
h.schedule(tasks=[...], cron="0 2 * * *")
await h.start()
```

## 包结构

```
harness/
  harness.py / task.py / memory.py       # 用户 API 层
  runners/      # AbstractRunner + ClaudeCliRunner（asyncio）
  storage/      # StorageProtocol + SQLAlchemy async 实现
  scheduler/    # AbstractScheduler + APScheduler v4
  notifier/     # AbstractNotifier + TelegramNotifier
  _internal/    # executor, stream_parser, session, exceptions
  cli.py        # harness migrate / harness runs
```

## 关键设计决策

- 异步优先（asyncio）
- SQLite 默认，存在 `project_path/.harness/harness.db`，WAL 模式
- `bypassPermissions` 默认开启
- `run()` 是 `pipeline([single_task])` 的语法糖
- Session 共享为主，重试/续跑时 prompt 注入兜底
- `output_schema` 支持 `str` 或 `type[BaseModel]`
- 详细设计见 `design/v1-design.md`

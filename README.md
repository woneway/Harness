# Harness

> AI-native 通用自动化流水线框架，内置 Claude Code runner

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

## 核心理念

声明你的流水线，框架负责可靠执行、记录、重试、调度、通知。

- **Claude Code runner 优先**：调用 Claude Code CLI 子进程，完整工具链 + bypassPermissions + session 持久化
- **Task 类型多态**：LLMTask / FunctionTask / ShellTask / PollingTask / Parallel，混合编排
- **开箱即用**：SQLite 存储、APScheduler v4 调度、Telegram 通知，均有抽象接口可替换

## 快速上手

```python
from harness import Harness, LLMTask, FunctionTask, ShellTask

h = Harness(project_path=".")

# 单次 LLM 调用
result = await h.run("分析并修复代码质量问题")

# 多步混合流水线
results = await h.pipeline([
    FunctionTask(fn=collect_data),
    LLMTask("分析数据，给出优化建议"),
    ShellTask(cmd="pytest tests/"),
])
```

## 安装

```bash
pip install harness-ai
```

或从源码安装（开发模式）：

```bash
git clone https://github.com/your-org/harness
cd harness
pip install -e ".[dev]"
```

## Task 类型

| 类型 | 用途 |
|------|------|
| `LLMTask` | 调用语言模型（默认 Claude Code CLI） |
| `FunctionTask` | 执行 Python 函数 |
| `ShellTask` | 执行 Shell 命令 |
| `PollingTask` | 提交异步任务后轮询（视频生成、TTS 等） |
| `Parallel` | 并发执行多个 Task |

## 特性

- **断点续跑**：`pipeline(tasks, resume_from=run_id)` 跳过已成功步骤
- **自动重试**：指数退避，可配置 `max_retries` 和 `backoff_base`
- **定时调度**：`h.schedule(tasks, cron="0 2 * * *"); await h.start()`
- **Memory 注入**：历史运行摘要 + `memory.md` 自动注入给 LLMTask
- **流式输出**：`stream_callback` 实时接收 Claude 输出

## 示例

```bash
python examples/code_stats.py          # 统计代码量
python examples/analyze_harness.py     # 分析 → 优化 → 复盘
python examples/research_report.py Clawith  # 联网调研报告
```

## CLI

```bash
harness runs           # 列出最近运行记录
harness runs --failed  # 只看失败的
harness migrate --to "postgresql+asyncpg://..."  # 迁移数据库
```

## 设计文档

详见 [`design/v1-design.md`](design/v1-design.md)。

## 许可证

Apache 2.0

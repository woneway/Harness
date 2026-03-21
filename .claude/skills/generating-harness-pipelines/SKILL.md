---
name: generating-harness-pipelines
description: Use when user describes a workflow they want to automate and needs a Harness pipeline Python script generated. Triggers: "帮我写一个 Harness pipeline", "用 Harness 实现 XXX", "生成 Harness 脚本", user describes a multi-step AI task needing orchestration.
---

# Generating Harness Pipelines

## Overview

Harness 是一个 Python asyncio AI workflow 框架。用户只需声明 Task 列表传给 `h.pipeline([...])`，框架负责执行、重试、记录、通知。

**项目路径：** `~/ai/projects/Harness`
**示例文件：** `examples/` 目录下有三个可直接运行的参考实现

## 公开 API（准确版本）

```python
from harness import (
    Harness,
    LLMTask, FunctionTask, ShellTask, PollingTask, Parallel,
    Task,           # LLMTask 的已废弃别名，新代码用 LLMTask
    Result, PipelineResult, TaskConfig, Memory,
)
from harness.runners.claude_cli import PermissionMode  # 需要时从此导入
from harness.notifier.telegram import TelegramNotifier
```

## Task 类型选择

| 需求 | Task 类型 | 使用场景 |
|------|-----------|---------|
| 调用 Claude / LLM | `LLMTask(prompt=...)` | 分析、写作、总结 |
| 执行 Python 函数 | `FunctionTask(fn=...)` | 文件 I/O、数据处理、API 调用 |
| 执行 Shell 命令 | `ShellTask(cmd=...)` | CLI 工具、脚本 |
| 并发执行多个任务 | `Parallel(tasks=[...])` | 独立的并发工作 |
| 轮询外部异步任务 | `PollingTask(...)` | 视频生成、TTS、长时间 API |

## 核心模式

### Harness 实例化

```python
h = Harness(
    project_path=".",           # 必填，存储 .harness/harness.db
    stream_callback=lambda text: print(text, end="", flush=True),  # 实时流式输出
    # 可选：
    # runner=ClaudeCliRunner(permission_mode=PermissionMode.DEFAULT),
    # memory=Memory(history_runs=3),
    # notifier=TelegramNotifier(bot_token=..., chat_id=...),
    # default_config=TaskConfig(timeout=1800, max_retries=1),
)
```

### 静态 vs 动态 Prompt

```python
# 静态：LLM 不需要前序结果
LLMTask(prompt="分析 Python 3.11 的新特性")

# 动态：LLM 需要前序输出 — 用闭包
def make_analysis_prompt(topic: str):
    def _prompt(results: list) -> str:
        prior: str = results[0].output   # Task 0 的输出
        return f"基于以下信息：\n{prior}\n\n请对 {topic} 做深度分析"
    return _prompt

LLMTask(prompt=make_analysis_prompt("招商银行"))
```

### 访问前序结果

```python
# 在 FunctionTask 或动态 prompt 中：
results[0].output   # Task 0 的输出
results[1].output   # Task 1 的输出（仅当前 task index > 1 时可用）
results[-1].output  # 最后一个已完成 task 的输出

# 注意：Task N 的 prompt callable 只能访问 results[0..N-1]
```

### CLI 入口（标准写法）

```python
import argparse, asyncio
from pathlib import Path

def main_cli() -> None:
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("topic", help="主题")
    parser.add_argument("--output", default=".", help="输出目录")
    args = parser.parse_args()
    asyncio.run(run(topic=args.topic, output_dir=Path(args.output).resolve()))

if __name__ == "__main__":
    main_cli()
```

### 保存文件模式

```python
def make_save_fn(topic: str, output_dir: Path):
    def _save(results: list) -> Path:
        import re
        content: str = results[-1].output
        safe_name = re.sub(r'[\\/:*?"<>|]', "_", topic).strip()
        path = output_dir / f"{safe_name}.md"
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path
    return _save

FunctionTask(fn=make_save_fn(topic, output_dir))
```

## 调研报告 Pipeline（标准模板）

对于"调研 X"请求，用 **3 阶段 pipeline**：

```
LLMTask [搜索整理]  →  LLMTask [撰写大纲]  →  LLMTask [写完整正文]  →  FunctionTask [保存]
```

**关键：需要真实数据时，必须在 prompt 中明确要求 WebSearch：**

```python
def make_gather_prompt(topic: str) -> str:
    return f"""请使用 WebSearch 工具搜索「{topic}」的以下信息：
1. 官网 / GitHub / 定价页
2. 核心功能与技术架构
3. 用户评价（HackerNews、Reddit、ProductHunt）
4. 主要竞品对比

整理为结构化摘要，每条信息标注来源 URL 和日期。"""
```

参考实现：`examples/research_report.py`（已在项目中）

## 完整示例（骨架）

```python
"""my_pipeline.py — 功能描述。

pipeline：
  LLMTask [步骤1]  →  FunctionTask [步骤2]  →  LLMTask [步骤3]
"""
from __future__ import annotations
import argparse, asyncio
from pathlib import Path
from harness import Harness, LLMTask, FunctionTask

async def run(topic: str, output_dir: Path) -> None:
    h = Harness(
        project_path=".",
        stream_callback=lambda text: print(text, end="", flush=True),
    )
    pr = await h.pipeline(
        [
            LLMTask(prompt=f"研究 {topic}，输出结构化摘要"),
            FunctionTask(fn=lambda results: results[0].output.upper()),
            LLMTask(prompt=lambda results: f"基于：{results[1].output}\n写完整报告"),
        ],
        name=f"my-pipeline-{topic}",
    )
    print(f"\n完成 | tokens: {pr.total_tokens} | run_id: {pr.run_id[:8]}")

def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic")
    parser.add_argument("--output", default=".")
    args = parser.parse_args()
    asyncio.run(run(args.topic, Path(args.output).resolve()))

if __name__ == "__main__":
    main_cli()
```

## 常见错误

| 错误 | 修复 |
|------|------|
| `prompt=lambda results: ...` 在循环中捕获变量 | 用闭包 `def make_prompt(x): def _p(r): return f"...{x}"`  |
| `results[1].output` 在 Task 1 的 prompt 中 | Task 1 只能看到 `results[0]`（零索引，前序任务） |
| 研究类任务 LLM 凭记忆生成 | prompt 中明确写 "请使用 WebSearch 工具搜索" |
| 忘记 `asyncio.run(main_cli())` | async 入口必须 |
| `Task(...)` | 用 `LLMTask(...)`，`Task` 是废弃别名 |
| 模块级获取用户输入 | 放入 `main_cli()` 函数 + argparse |

## 参考文件

- `examples/research_report.py` — 调研报告（搜索 + 大纲 + 正文 + 保存）
- `examples/code_stats.py` — 代码统计（FunctionTask + LLMTask + ShellTask 混用）
- `examples/analyze_harness.py` — 分析优化复盘（三阶段，含前后对比）
- `examples/video_pipeline.py` — 视频生成（LLMTask + Parallel[PollingTask×2] + FunctionTask）
- `examples/iterationforge_style.py` — 定时扫描修复（schedule + Memory + TelegramNotifier）

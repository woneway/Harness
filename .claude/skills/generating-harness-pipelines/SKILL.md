---
name: generating-harness-pipelines
description: Use when user wants to create or modify a Harness pipeline Python script, or debug existing pipeline issues (output_schema, data flow, LLMTask/FunctionTask/PollingTask errors). Triggers: "帮我写一个 Harness pipeline", "用 Harness 实现 XXX", "生成 Harness 脚本", "pipeline 报错", "output_schema 不生效", user describes a multi-step AI task needing orchestration.
---

# Harness Pipeline 生成指南

## 概述

Harness 是一个 Python AI 工作流框架，通过组合 `LLMTask`（调用 Claude CLI）、`FunctionTask`（纯 Python）、`ShellTask`、`Parallel`、`PollingTask` 来构建多步自动化 pipeline。

**本 skill 适用于：**
- 从需求出发设计并生成新 pipeline（`examples/pipeline_name/main.py`）
- 修改现有 pipeline（增删 task、调整 prompt、改数据流）
- 调试 pipeline 问题（`output_schema` 不生效、`results[N]` 索引错误、`Parallel` 数据流、Task 报错等）

---

## 三个阶段

**阶段一 → 澄清需求**
**阶段二 → 设计 pipeline 结构**
**阶段三 → 生成、测试、修复直到跑通**

---

## 阶段一：澄清需求

只问用户尚未回答的问题：

1. **数据来源** — 本地文件 / Web 搜索 / 第三方 API / 数据库？
2. **哪些步骤需要 LLM？** — 哪些纯 Python 就能完成（文件 I/O、格式化、计算）？
3. **输出目标** — 文件（格式/路径）/ Telegram / API 调用 / 打印到控制台？
4. **并行需求** — 有没有可以同时跑的独立步骤？
5. **定时运行** — 一次性还是 cron 定时？
6. **异步 API** — 需要提交 job 后轮询结果吗？（视频生成、TTS、长时间 job）

**何时跳过：** 如果用户消息已覆盖所有相关问题，直接陈述假设并进入阶段二。

---

## 阶段二：设计

### 步骤 0：领域调研（按需）

如果领域陌生（量化金融、特定 API、行业专属流程），先用 WebSearch 再分解步骤。分解错了代价很高。

搜索方向：解决同类问题的开源项目、将要使用的 Python 库 API、行业标准工作流。

### 步骤 1：将需求映射到 Task 类型

| 需求 | Task 类型 | 说明 |
|------|-----------|------|
| LLM 推理 / 写作 / 总结 | `LLMTask` | 调用 Claude Code CLI |
| 确定性逻辑、文件 I/O、数据处理 | `FunctionTask` | 纯 Python，不过 LLM |
| Shell 命令 | `ShellTask` | asyncio 子进程 |
| 可同时运行的独立步骤 | `Parallel([...])` | 等待所有子任务完成 |
| 提交异步 job → 轮询结果 | `PollingTask` | 用于异步外部 API |

**原则：** 确定性操作用 `FunctionTask`，只有需要推理或语言能力时才用 `LLMTask`。

### 步骤 2：规划数据流

```
Task 0 输出 → results[0].output  （Task 1+ 可访问）
Task 1 输出 → results[1].output  （Task 2+ 可访问）
```

**Parallel 块内：** 每个子任务收到的 `results` 来自 Parallel 块**之前**的任务，不包含兄弟任务的输出。

**何时用 `output_schema`：** 后续任务需要访问 LLMTask 输出的特定字段时。使用时，prompt 必须明确要求 LLM 输出符合 schema 字段的 JSON——Harness 负责校验结构，但不会自动注入格式指令。

```python
class AnalysisResult(BaseModel):
    signal: str        # "看涨" / "看跌" / "中性"
    reasons: list[str]

LLMTask(
    prompt="分析以下数据。以 JSON 格式输出，字段：signal（看涨/看跌/中性）、reasons（list[str]）。\n\n数据：...",
    output_schema=AnalysisResult,
)
# 后续任务访问：results[N].output.signal, results[N].output.reasons
```

### 步骤 3：验证外部 API 签名

写任何调用外部库（akshare、httpx、pandas 等）的 `FunctionTask` 之前，先用 WebSearch 确认实际函数签名和返回类型，不要靠记忆猜测。

### 步骤 4：陈述设计，然后开始

```
FunctionTask [数据获取] — akshare.stock_zh_a_hist()，输出 DataFrame
    ↓
LLMTask [分析]         — output_schema=AnalysisResult(signal, reasons)
    ↓
FunctionTask [保存]    — results[1].output.signal → {symbol}_report.md
```

只有当存在会改变整体结构的真实歧义时，才等待用户确认。

---

## 阶段三：生成与测试

### 输出结构

```
examples/pipeline_name/
  main.py
  README.md
```

### 职责边界

- **本 skill 负责：** pipeline 结构、LLMTask prompt、简单 FunctionTask（≤50 行）
- **复杂子系统用占位符**（回测引擎、视频渲染器等）：

```python
def fetch_data(results: list):
    """TODO: 待实现
    输出：pd.DataFrame，列：[date, open, high, low, close, volume]
    建议用：akshare.stock_zh_a_hist(symbol, period='daily', adjust='qfq')
    """
    raise NotImplementedError("fetch_data：详见 docstring 接口说明")
```

### 代码模板

```python
"""main.py — 一句话描述。

pipeline:
  FunctionTask [步骤名] — 做什么，输出什么
      ↓
  LLMTask [步骤名]      — 做什么，输出什么
      ↓
  FunctionTask [保存]   — 写入文件，返回 Path

运行:
    uv run python examples/pipeline_name/main.py <topic> [--output dir]
"""
from __future__ import annotations

import argparse
import asyncio
import re
from datetime import date
from pathlib import Path

from harness import FunctionTask, Harness, LLMTask, Parallel, PollingTask, ShellTask


# ---------------------------------------------------------------------------
# （可选）结构化输出 schema
# ---------------------------------------------------------------------------

# from pydantic import BaseModel
#
# class MyResult(BaseModel):
#     field1: str
#     field2: list[str]


# ---------------------------------------------------------------------------
# Task 函数
# ---------------------------------------------------------------------------

# 模式 A：纯字符串 prompt（Task 0 — 不需要前序结果）
GATHER_PROMPT = """请使用 WebSearch 搜索「{topic}」，至少 3 次查询，覆盖官方信息、技术细节、用户反馈。
每条信息标注来源 URL 和日期。不得凭记忆编造数字或事实。"""


# 模式 B：callable prompt（需要前序结果）
def make_analysis_prompt(topic: str) -> callable:
    def _prompt(results: list) -> str:
        gathered = results[0].output
        return f"基于以下关于「{topic}」的信息：\n\n{gathered}\n\n请给出分析结论。"
    return _prompt


# 模式 C：保存结果到带日期的文件
def make_save_fn(name: str, output_dir: Path, result_index: int) -> callable:
    def _save(results: list) -> Path:
        content = results[result_index].output
        safe = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
        path = output_dir / f"{safe}_{date.today()}.md"
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path
    return _save


# 模式 D：PollingTask 的提交 + 轮询函数
def submit_job(results: list) -> str:
    """提交异步 job，返回 job handle。"""
    raise NotImplementedError
    return "job-handle-001"

def poll_job(handle: str) -> dict:
    """查询 job 状态。"""
    raise NotImplementedError
    return {"status": "done", "url": "https://example.com/result"}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run(topic: str, output_dir: Path) -> None:
    h = Harness(
        project_path=".",
        stream_callback=lambda text: print(text, end="", flush=True),
    )
    pr = await h.pipeline(
        [
            LLMTask(prompt=GATHER_PROMPT.format(topic=topic)),    # Task 0：字符串 prompt
            LLMTask(prompt=make_analysis_prompt(topic)),          # Task 1：callable prompt
            # Parallel 示例（按需取消注释）：
            # Parallel(
            #     tasks=[
            #         PollingTask(
            #             submit_fn=submit_job,
            #             poll_fn=poll_job,
            #             success_condition=lambda r: r.get("status") == "done",
            #             poll_interval=5,
            #             timeout=300,
            #         ),
            #     ],
            #     error_policy="all_or_nothing",
            # ),
            FunctionTask(fn=make_save_fn(topic, output_dir, 1)), # Task 2：保存 Task 1 的输出
        ],
        name=f"pipeline-{topic[:20]}",
    )
    saved_path: Path = pr.results[-1].output
    print(f"\n完成 | 文件：{saved_path} | {pr.total_duration_seconds:.1f}s | tokens: {pr.total_tokens:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main_cli() -> None:
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("topic", help="主要输入参数")
    parser.add_argument("--output", default=".", help="输出目录（默认：当前目录）")
    args = parser.parse_args()
    asyncio.run(run(args.topic, Path(args.output).resolve()))


if __name__ == "__main__":
    main_cli()
```

### README 模板

```markdown
# Pipeline 名称 — 一句话描述

## 环境要求
- 依赖：`uv add <package>`
- 环境变量：`export FOO=...`

## 运行

\```bash
uv run python examples/pipeline_name/main.py "输入参数"
uv run python examples/pipeline_name/main.py "输入参数" --output ./output
\```

## Pipeline 结构

\```
FunctionTask [步骤名] — 描述
    ↓
LLMTask [步骤名]      — 描述
    ↓
FunctionTask [保存]   — 描述
\```
```

### 测试流程

**第一步 — 语法与导入检查（必做）：**
```bash
uv run python -c "import ast; ast.parse(open('examples/pipeline_name/main.py').read()); print('syntax OK')"
uv run python examples/pipeline_name/main.py --help
```

**第二步 — 端到端运行：**
- **有占位符任务** → 跳过。交付 pipeline 骨架 + 接口说明，完成。
- **无占位符** → 运行（含 LLMTask 时提示用户会消耗 Claude CLI tokens）：
```bash
uv run python examples/pipeline_name/main.py "test_input" --output /tmp/harness_test
```

**第三步 — 调试循环（重复直到成功）：**
1. 确认哪个 task 失败（task 序号 + 名称）
2. 精确报告错误：
   ```
   ❌ Task 1 [fetch_data] 失败
   错误：ModuleNotFoundError: No module named 'akshare'
   修复：uv add akshare
   ```
3. 修复后重新运行

**第四步 — 至少端到端成功一次才算完成：**
```
✅ Pipeline 成功 | Run ID: abc12345
Task 0 [搜索]    ✅  2.1s
Task 1 [分析]    ✅  18.4s  (tokens: 1,203)
Task 2 [保存]    ✅  0.1s
总耗时：20.6s
```

---

## 常见错误

| 错误 | 修复方法 |
|------|---------|
| `from harness.task import Parallel, PollingTask` | 应为 `from harness import Parallel, PollingTask`（已在顶层导出） |
| `Parallel([task_a, task_b])` 子任务没有执行 | v1.0 的 bug（位置参数落到 `config`），已修复；两种写法均可，推荐 `Parallel(tasks=[...])` |
| Task 1 里写 `lambda results: results[1].output` | Task 1 只有 `results[0]`，检查索引 |
| 循环里用 `lambda results: f"...{x}"` | 用闭包捕获 `x`：`def make_prompt(x): ...` |
| 设置了 `output_schema` 但 prompt 没要求 JSON | 在 prompt 里明确说明 JSON 格式要求 |
| Parallel 子任务访问兄弟任务的结果 | 子任务只能看到 Parallel 块**之前**的 results |
| 在本 skill 里实现复杂子系统 | 用 `NotImplementedError` 占位符 + 接口说明 |
| 有占位符时跑端到端 | 有占位符就跳过 e2e，只交付骨架 |
| 使用 `Task(...)` | 改用 `LLMTask(...)`，`Task` 已废弃 |

---

## 修改现有 Pipeline

用户要修改已有 pipeline 时：
1. 先读 `examples/pipeline_name/main.py`
2. 定位受影响的 task
3. 精准修改，不动无关 task
4. 重新跑静态检查，无占位符时跑端到端

---

## 参考资料

- Harness 项目：`~/ai/projects/Harness`
- 参考示例：`examples/research_report/main.py`、`examples/video_pipeline/main.py`
- 查看运行历史：`harness runs` / `harness runs --failed`

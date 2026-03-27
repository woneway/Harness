---
name: generating-harness-pipelines
description: Use when user wants to create or modify a Harness pipeline Python script, or debug existing pipeline issues (output_schema, data flow, State, Agent, Discussion errors). Triggers: "帮我写一个 Harness pipeline", "用 Harness 实现 XXX", "生成 Harness 脚本", "pipeline 报错", "output_schema 不生效", user describes a multi-step AI task needing orchestration.
---

# Harness Pipeline 生成指南（v2.2）

## 概述

Harness 是一个 Python AI 工作流框架，通过组合 Task 类型构建多步自动化 pipeline。v2 使用 `State` 共享状态模式管理数据流，支持 `Agent` 角色化、`Discussion` 结构化讨论、`Condition`/`Loop` 流程控制。

**本 skill 适用于：**
- 从需求出发设计并生成新 pipeline（`examples/pipeline_name/main.py`）
- 修改现有 pipeline（增删 task、调整 prompt、改数据流）
- 调试 pipeline 问题（`State` 字段未更新、`output_key` 拼写错误、Task 报错等）

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
5. **条件/循环** — 是否需要分支逻辑（质量检查、分级处理）或迭代优化？
6. **多角色交互** — 需要多个 Agent 讨论/辩论/达成共识吗？
7. **定时运行** — 一次性还是 cron 定时 / 事件触发？
8. **异步 API** — 需要提交 job 后轮询结果吗？（视频生成、TTS、长时间 job）

**何时跳过：** 如果用户消息已覆盖所有相关问题，直接陈述假设并进入阶段二。

---

## 阶段二：设计

### 步骤 0：领域调研（按需）

如果领域陌生（量化金融、特定 API、行业专属流程），先用 WebSearch 再分解步骤。分解错了代价很高。

### 步骤 1：将需求映射到 Task 类型

| 需求 | Task 类型 | 说明 |
|------|-----------|------|
| LLM 推理 / 写作 / 总结 | `LLMTask` | 调用 Claude Code CLI |
| Agent 角色化 LLM 调用 | `Agent.task()` | 返回已配置 system_prompt + runner 的 LLMTask |
| 确定性逻辑、文件 I/O、数据处理 | `FunctionTask` | 纯 Python，不过 LLM |
| Shell 命令 | `ShellTask` | asyncio 子进程 |
| 多角色文本对话 / 辩论 | `Dialogue` + `Role` | 多角色轮流发言，含历史上下文 |
| 多 Agent 结构化讨论（立场追踪、共识） | `Discussion` + `Agent` | 两阶段执行，position_schema 必填 |
| 可同时运行的独立步骤 | `Parallel([...])` | 等待所有子任务完成 |
| 提交异步 job → 轮询结果 | `PollingTask` | 用于异步外部 AI API |
| 条件分支 | `Condition` | `check` → `if_true` / `if_false` 分支 |
| 迭代优化 | `Loop` | `body` + `until` 终止条件 + `max_iterations` |

**原则：** 确定性操作用 `FunctionTask`，只有需要推理或语言能力时才用 `LLMTask`，多角色交互用 `Dialogue`/`Discussion`。

### 步骤 2：规划数据流

**推荐：State 模式（v2）**

```python
from harness import State, LLMTask, FunctionTask

class MyState(State):
    data: str = ""
    analysis: str = ""
    report: str = ""

pr = await h.pipeline([
    FunctionTask(fn=lambda state: fetch(), output_key="data"),
    LLMTask(
        prompt=lambda state: f"分析: {state.data}",
        output_key="analysis",
    ),
    LLMTask(
        prompt=lambda state: f"基于分析生成报告: {state.analysis}",
        output_key="report",
    ),
], state=MyState())

# 访问结果
print(pr.state.report)          # State 字段
print(pr.results[1].output)     # Result 列表
```

**关键规则：**
- 每个 Task 通过 `output_key="field"` 自动写入 State 对应字段
- 后续 Task 的 prompt/fn 通过 `lambda state: state.field` 访问前序输出
- State 字段需要预定义（继承 `State`，声明默认值）

**v1 兼容模式（仍可用）：**

```python
# v1 签名 fn(results) 自动兼容，无需修改
def save(results: list) -> Path:
    content = results[1].output
    ...
```

**Parallel 块内：** 每个子任务的 `state` 是 Parallel 块开始时的快照，兄弟任务的输出互不可见。

**何时用 `output_schema`：** 后续任务需要访问 LLMTask 输出的特定字段时。prompt 必须明确要求输出 JSON。

```python
class AnalysisResult(BaseModel):
    signal: str
    reasons: list[str]

LLMTask(
    prompt="分析数据，以 JSON 输出：signal（看涨/看跌/中性）、reasons。\n\n数据：...",
    output_schema=AnalysisResult,
    output_key="analysis",
)
# 后续: state.analysis.signal, state.analysis.reasons
```

### 步骤 3：验证外部 API 签名

写任何调用外部库（akshare、httpx、pandas 等）的 `FunctionTask` 之前，先用 WebSearch 确认实际函数签名和返回类型。

### 步骤 4：陈述设计，然后开始

```
FunctionTask [数据获取]  → output_key="data"
    ↓
LLMTask [分析]           → output_key="analysis"  (output_schema=AnalysisResult)
    ↓
Condition [质量检查]
  ├─ if_true:  LLMTask [详细报告]  → output_key="report"
  └─ if_false: FunctionTask [简报]  → output_key="report"
    ↓
FunctionTask [保存]      → output_key="path"
```

---

## Agent 角色化

### 基础用法

```python
from harness import Agent

# 方式一：直接 system_prompt
analyst = Agent(name="analyst", system_prompt="你是技术分析师", runner=my_runner)

# 方式二：结构化定义（自动组装 system_prompt）
analyst = Agent(
    name="龙头猎手",
    description="辨识龙头的短线选手。",
    goal="抓住每日龙头股",
    backstory="从涨停板战法起家，擅长辨识市场主线。",
    constraints=["只做龙头", "不碰垃圾股"],
    runner=my_runner,
)

# 独立执行（不需要 pipeline）
text = await analyst.run("分析今日走势")

# 在 pipeline 中使用（通过 Agent.task() 创建 LLMTask）
pr = await h.pipeline([
    analyst.task("分析走势", output_key="analysis"),
], state=MyState())

# 在 Dialogue 中使用（降级为 Role）
role = analyst.as_role(lambda ctx: f"分析: {ctx.last_from('trader')}")
```

**注意：** Agent 不是 PipelineStep，通过 `agent.task()` 或 `agent.as_role()` 进入 pipeline。

---

## Discussion 多 Agent 结构化讨论

### 基础用法

```python
from pydantic import BaseModel
from harness import Agent, Discussion, DiscussionOutput
from harness.tasks.discussion import all_agree_on

class TradingPosition(BaseModel):
    top_pick: str
    direction: str       # "买入" / "卖出" / "观望"
    confidence: float    # 0-1

analyst = Agent(name="技术分析师", description="看K线", runner=my_runner)
trader = Agent(name="短线交易员", description="盘中选股", runner=my_runner)

class MyState(State):
    market: str = ""
    discussion: DiscussionOutput | None = None

pr = await h.pipeline([
    FunctionTask(fn=lambda s: fetch_market(), output_key="market"),
    Discussion(
        agents=[analyst, trader],
        position_schema=TradingPosition,       # 必填！定义立场结构
        topic="下午盘选股",
        background=lambda state: f"行情：{state.market}",
        max_rounds=4,
        convergence=all_agree_on("top_pick"),   # 收敛检测
        output_key="discussion",
    ),
], state=MyState())

output: DiscussionOutput = pr.state.discussion
print(output.converged)          # 是否达成共识
print(output.final_positions)    # {"技术分析师": TradingPosition(...), ...}
print(output.position_history)   # 立场演变
```

### Discussion vs Dialogue 选择

| 场景 | 使用 |
|------|------|
| 自由文本对话、辩论、创意碰撞 | `Dialogue` + `Role` |
| 需要追踪立场变化、检测共识 | `Discussion` + `Agent` + `position_schema` |

### 收敛工具函数

```python
from harness.tasks.discussion import all_agree_on, positions_stable, majority_agree_on

all_agree_on("top_pick")              # 所有 Agent 在 top_pick 字段一致
positions_stable(rounds=2)             # 连续 2 轮所有 Agent 立场不变
majority_agree_on("direction", 0.6)    # 60%+ Agent 在 direction 字段一致
```

### extraction_runner（Phase 2 用更便宜的模型）

```python
from harness.runners.openai import OpenAIRunner

cheap_runner = OpenAIRunner(model="gpt-4o-mini", ...)

Discussion(
    agents=[analyst, trader],
    position_schema=TradingPosition,
    extraction_runner=cheap_runner,  # Phase 2 立场提取用便宜模型
    ...
)
```

---

## Condition / Loop 流程控制

### Condition

```python
from harness import Condition

Condition(
    check=lambda state: len(state.analysis) > 100,
    if_true=[LLMTask("生成详细报告", output_key="report")],
    if_false=[FunctionTask(fn=lambda s: "简短报告", output_key="report")],
)
```

### Loop

```python
from harness import Loop

Loop(
    body=[LLMTask("优化报告", output_key="report")],
    until=lambda state: quality_ok(state.report),
    max_iterations=3,
)
```

**限制：** `Condition`/`Loop`/`Discussion` 不能嵌入 `Parallel`（`InvalidPipelineError`），可嵌入其他 PipelineStep。

---

## Dialogue 多角色对话

### 基础用法

```python
from harness import Dialogue, Role, DialogueProgressEvent

def make_role_a_prompt(ctx) -> str:
    if not ctx.all_from("角色A"):
        return "你是角色A。开场陈述..."
    last_b = ctx.last_from("角色B") or ""
    return f"角色B 说：{last_b[-300:]}\n\n请反驳..."

def on_progress(evt: DialogueProgressEvent) -> None:
    if evt.event == "start":
        print(f"[{evt.role_name}] 第 {evt.round_or_turn + 1} 轮...")

dialogue = Dialogue(
    roles=[
        Role(name="角色A", system_prompt="专业辩手。", prompt=make_role_a_prompt),
        Role(name="角色B", system_prompt="专业辩手。", prompt=make_role_b_prompt),
    ],
    background="辩题：...",
    max_rounds=5,
    until_round=lambda ctx: ctx.round >= 4,
    progress_callback=on_progress,
    role_stream_callback=lambda role, chunk: print(chunk, end="", flush=True),
)
```

### `until` vs `until_round`

| 场景 | 使用 |
|------|------|
| 固定轮数终止 | `until_round=lambda ctx: ctx.round >= N-1` |
| 内容条件终止 | `until=lambda ctx: "同意" in (ctx.last_from("角色B") or "")` |

### Dialogue 的输出

```python
from harness import DiscussionOutput
from harness.task import DialogueOutput

def summarize(state) -> str:
    out: DialogueOutput = state.dialogue_result
    turns_text = "\n\n".join(
        f"【{t.role_name} · 第 {t.round + 1} 轮】\n{t.content}"
        for t in out.turns
    )
    return f"辩论记录：\n{turns_text}"
```

---

## 阶段三：生成与测试

### 输出结构

```
examples/pipeline_name/
  main.py
```

### 职责边界

- **本 skill 负责：** pipeline 结构、LLMTask prompt、简单 FunctionTask（≤50 行）
- **复杂子系统用占位符**（回测引擎、视频渲染器等）：

```python
def fetch_data(state):
    """TODO: 待实现
    输出：pd.DataFrame，列：[date, open, high, low, close, volume]
    建议用：akshare.stock_zh_a_hist(symbol, period='daily', adjust='qfq')
    """
    raise NotImplementedError("fetch_data：详见 docstring 接口说明")
```

### 代码模板（State 模式）

```python
"""main.py — 一句话描述。

pipeline:
  FunctionTask [步骤名] → output_key="data"
      ↓
  LLMTask [步骤名]      → output_key="analysis"
      ↓
  FunctionTask [保存]   → output_key="path"

运行:
    uv run python examples/pipeline_name/main.py <topic> [--output dir]
"""
from __future__ import annotations

import argparse
import asyncio
import re
from datetime import date
from pathlib import Path

from harness import (
    Harness, State,
    LLMTask, FunctionTask, ShellTask, Parallel, Condition, Loop,
)


# ---------------------------------------------------------------------------
# State 定义
# ---------------------------------------------------------------------------

class PipelineState(State):
    data: str = ""
    analysis: str = ""
    report: str = ""
    saved_path: str = ""


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

# FunctionTask — 接收 state（v2 签名）
def fetch_data(state) -> str:
    """获取数据。"""
    return "fetched data..."


# LLMTask prompt — 字符串或 callable(state)
ANALYSIS_PROMPT = "分析以下数据，给出结论。\n\n数据：{data}"

def make_save_fn(output_dir: Path):
    def _save(state) -> str:
        content = state.report
        path = output_dir / f"report_{date.today()}.md"
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)
    return _save


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
            FunctionTask(fn=fetch_data, output_key="data"),
            LLMTask(
                prompt=lambda state: ANALYSIS_PROMPT.format(data=state.data),
                output_key="analysis",
            ),
            LLMTask(
                prompt=lambda state: f"基于分析生成报告：\n{state.analysis}",
                output_key="report",
            ),
            FunctionTask(fn=make_save_fn(output_dir), output_key="saved_path"),
        ],
        state=PipelineState(),
        name=f"pipeline-{topic[:20]}",
    )
    print(f"\n完成 | 文件：{pr.state.saved_path} | {pr.total_duration_seconds:.1f}s | tokens: {pr.total_tokens:,}")


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
2. 精确报告错误并修复
3. 修复后重新运行

**第四步 — 至少端到端成功一次才算完成。**

---

## 常见错误

| 错误 | 修复方法 |
|------|---------|
| `output_key` 拼写与 State 字段名不一致 | State 字段名必须和 `output_key` 完全一致 |
| `pipeline()` 没传 `state=` | 使用 State 模式时必须 `pipeline([...], state=MyState())` |
| State 字段没有默认值 | State 字段必须有默认值（如 `field: str = ""`） |
| `from harness.task import Parallel` | 应为 `from harness import Parallel`（已在顶层导出） |
| Discussion 未设 `position_schema` | `position_schema` 必填，`TaskFailedError` |
| Discussion/Condition/Loop 嵌入 Parallel | 不允许：`InvalidPipelineError` |
| Agent 同时设 `system_prompt` 和 `description` | `system_prompt` 优先，`description` 被忽略 |
| 设置了 `output_schema` 但 prompt 没要求 JSON | 在 prompt 里明确说明 JSON 格式要求 |
| Parallel 子任务访问兄弟任务的结果 | 子任务只能看到 Parallel 块**之前**的 state 快照 |
| `Dialogue` 中判断首轮用 `if not ctx.history` | 所有角色共用 `history`；改用 `if not ctx.all_from("角色名")` |
| `until=lambda ctx: ctx.round >= N` 轮次终止 | 用 `until_round`（每轮结束后检查），`until` 会在第一个角色发完后就触发 |
| `Dialogue(stream_callback=...)` 多角色流式 | 用 `role_stream_callback=lambda role, chunk: ...` |
| `progress_callback(event, round, role)` 位置参数 | 接收单个对象：`def on_progress(evt: DialogueProgressEvent): ...` |
| 使用 `Task(...)` | 改用 `LLMTask(...)`，`Task` 已废弃 |
| v1 `fn(results)` 签名报错 | v1 签名自动兼容，无需修改；但新代码推荐 `fn(state)` |

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
- 参考示例：`examples/stock_discussion/main.py`（Discussion）、`examples/stock_traders/main.py`（Agent + Dialogue）、`examples/agent_discussion/main.py`
- 项目文档：`CLAUDE.md`
- 查看运行历史：`harness runs` / `harness runs --failed`

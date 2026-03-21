---
name: generating-harness-pipelines
description: Use when user describes a workflow they want to automate and needs a Harness pipeline Python script generated. Triggers: "帮我写一个 Harness pipeline", "用 Harness 实现 XXX", "生成 Harness 脚本", user describes a multi-step AI task needing orchestration.
---

# Generating Harness Pipelines

## Your Job (3 Phases)

**Phase 1 → Clarify:** Ask targeted questions until you understand the full workflow.
**Phase 2 → Design:** Think through the best pipeline structure; use WebSearch if needed for external APIs or libraries.
**Phase 3 → Generate + Test:** Write the script, run it, fix errors until it works.

---

## Phase 1: Requirement Clarification

Before writing any code, ask the user these questions (combine into one message, only ask what's not already clear):

```
1. 目标输出是什么？(文件/终端输出/API 调用/数据库写入？)
2. 数据来源是什么？(本地文件/Web 搜索/API/用户输入？)
3. 哪些步骤需要 LLM？哪些是纯 Python/Shell 可以完成的？
4. 输出格式有要求吗？(Markdown/JSON/特定结构？)
5. 是否需要定时运行？(cron 表达式？)
6. 错误处理预期？(遇到失败是重试/跳过/报警？)
```

**When to proceed without asking:** If the user's description already answers most questions (e.g., "研究 X 然后生成 Markdown 报告"), state your assumptions explicitly and proceed to Phase 2.

---

## Phase 2: Pipeline Design

### Step 1: Map the workflow to Task types

| User's need | Best Task type | Notes |
|-------------|---------------|-------|
| LLM 分析/写作/总结 | `LLMTask` | 默认用 Claude Code CLI，bypassPermissions |
| 数据获取/处理/文件操作 | `FunctionTask` | 纯 Python，不过 LLM |
| 调用 Shell 工具 | `ShellTask` | asyncio subprocess |
| 独立的并行任务 | `Parallel([...])` | 所有子任务结束后继续 |
| 提交后轮询结果的 API | `PollingTask` | 视频生成、TTS、长时间 job |

**Decision rule:** Prefer `FunctionTask` over `LLMTask` when the step is deterministic (file I/O, math, formatting). Use `LLMTask` only when reasoning or language is required.

### Step 2: Think through data flow + output_schema

Each task receives `results: list[Result]` — all prior task outputs:

```
Task 0 output → results[0].output  (available in Task 1+)
Task 1 output → results[1].output  (available in Task 2+)
Task N output → results[N].output  (available in Task N+1+)
```

**When to use `output_schema`:** If a later task needs to access specific fields from an earlier LLMTask's output, use `output_schema=SomeModel` to get structured data instead of raw strings.

```python
from pydantic import BaseModel

class ScanResult(BaseModel):
    issues: list[str]
    summary: str

# Task 0: LLM returns structured data
LLMTask(prompt="扫描项目，列出所有问题", output_schema=ScanResult)

# Task 1: access specific fields
LLMTask(prompt=lambda results: f"修复这些问题：{results[0].output.issues}")
#                                                              ^^^^^^ typed field, not raw string
```

Use `output_schema` when:
- A subsequent task needs specific fields (`.issues`, `.score`, `.items`)
- You want to validate LLM output structure before proceeding
- The output will be processed by a FunctionTask that expects structured data

Skip `output_schema` when: the next step just embeds the whole text into another prompt.

### Step 3: Search for external APIs

Before writing any `FunctionTask` that calls an external library (akshare, httpx, pandas, etc.):
- Use WebSearch to verify the actual function signature and return type
- Do not guess API shapes from memory

### Step 4: State your design, then proceed

Write a brief pipeline diagram before coding:

```
FunctionTask [数据获取] — akshare.stock_zh_a_hist(), 输出 DataFrame
    ↓
LLMTask [分析]         — output_schema=AnalysisResult(signal, reasons)
    ↓
FunctionTask [保存]    — results[1].output.signal → {symbol}_report.md
```

If requirements are clear, proceed immediately after stating the design. Only wait for user confirmation if there's a genuine ambiguity that would change the structure.

---

## Phase 3: Generate + Test

### Code Structure (complete, runnable template)

```python
"""script_name.py — 一句话描述。

pipeline:
  FunctionTask [步骤名] — 做什么，输出什么
      ↓
  LLMTask [步骤名]      — 做什么，输出什么
      ↓
  FunctionTask [保存]   — 写入文件，返回 Path

运行:
    python examples/script_name.py <arg> [--output dir]
"""
from __future__ import annotations

import argparse
import asyncio
import re
from datetime import date
from pathlib import Path

from harness import FunctionTask, Harness, LLMTask


# ---------------------------------------------------------------------------
# (Optional) Structured output schema
# ---------------------------------------------------------------------------

# from pydantic import BaseModel
#
# class MyResult(BaseModel):
#     field1: str
#     field2: list[str]


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def make_prompt(captured_var: str) -> callable:
    """Closure: capture external variable for dynamic prompt."""
    def _prompt(results: list) -> str:
        prior = results[0].output
        return f"Based on:\n{prior}\n\nNow do X for {captured_var}"
    return _prompt


def make_save_fn(name: str, output_dir: Path, result_index: int) -> callable:
    """Closure: save results[result_index].output to a dated Markdown file."""
    def _save(results: list) -> Path:
        content = results[result_index].output
        safe = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
        path = output_dir / f"{safe}_{date.today()}.md"
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path
    return _save


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run(arg: str, output_dir: Path) -> None:
    h = Harness(
        project_path=".",
        stream_callback=lambda text: print(text, end="", flush=True),
    )
    pr = await h.pipeline(
        [
            LLMTask(prompt=make_prompt(arg)),                          # Task 0
            FunctionTask(fn=make_save_fn(arg, output_dir, 0)),         # Task 1
        ],
        name=f"pipeline-{arg[:20]}",
    )
    saved_path: Path = pr.results[1].output
    print(f"\n完成 | 文件：{saved_path} | {pr.total_duration_seconds:.1f}s | tokens: {pr.total_tokens:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main_cli() -> None:
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("arg", help="主要输入参数")
    parser.add_argument("--output", default=".", help="输出目录（默认：当前目录）")
    args = parser.parse_args()
    asyncio.run(run(args.arg, Path(args.output).resolve()))


if __name__ == "__main__":
    main_cli()
```

### Testing Protocol

**Distinguish by pipeline type before running:**

**Pipeline contains only FunctionTask / ShellTask** → run full end-to-end immediately:
```bash
# 1. Syntax check
python -c "import ast; ast.parse(open('examples/script.py').read()); print('syntax OK')"
# 2. --help
python examples/script.py --help
# 3. End-to-end run (safe, no LLM cost)
python examples/script.py "test_input" --output /tmp/harness_test
# 4. Verify output
ls -la /tmp/harness_test/ && head -20 /tmp/harness_test/*
```

**Pipeline contains LLMTask** → syntax + import check only, then inform user:
```bash
# 1. Syntax check
python -c "import ast; ast.parse(open('examples/script.py').read()); print('syntax OK')"
# 2. Import check (catches missing dependencies without invoking Claude CLI)
python -c "import examples.script_name; print('imports OK')"
# 3. --help
python examples/script.py --help
```
Then tell the user: *"语法和导入检查通过。完整运行会调用 Claude CLI（消耗 tokens），请确认后执行：`python examples/script.py <arg>`"*

**Fix and re-check until all steps pass.**

---

## Key Patterns

### Dynamic prompt (needs prior result)
```python
def make_prompt(captured: str) -> callable:
    def _prompt(results: list) -> str:
        data = results[0].output
        return f"Based on: {data}\n\nNow do X for {captured}"
    return _prompt
```

### Structured output between tasks
```python
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    signal: str          # "看涨" / "看跌" / "中性"
    reasons: list[str]
    confidence: int      # 1-5

LLMTask(prompt="分析以下数据...", output_schema=AnalysisResult)
# Next task accesses: results[N].output.signal, results[N].output.reasons
```

### File save with date
```python
def make_save_fn(name: str, output_dir: Path, result_index: int) -> callable:
    def _save(results: list) -> Path:
        content = results[result_index].output
        safe = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
        path = output_dir / f"{safe}_{date.today()}.md"
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path
    return _save
```

### WebSearch in LLMTask prompt
```python
return f"""请使用 WebSearch 工具搜索「{topic}」，至少进行 4 次查询：
1. 官方信息（官网/文档/定价）
2. 技术架构与实现原理
3. 用户评价（HackerNews/Reddit）
4. 竞品对比

每条信息标注来源 URL 和日期。不得凭记忆编造数字或事实。"""
```

### Resume on failure
```python
# If pipeline fails partway, resume from saved run_id:
# results = await h.pipeline(tasks, resume_from="<run_id_from_error>")
# Run `harness runs --failed` to find the run_id
```

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `make_save_fn` called but not defined | Always include it in the file if used |
| `lambda results: results[1].output` in Task 1 | Task 1 only sees `results[0]` |
| `prompt=lambda results: f"...{x}"` in loop | Use closure to capture `x` |
| Running end-to-end for LLMTask pipeline without warning | Inform user: will invoke Claude CLI and cost tokens |
| Not using `output_schema` when next task needs specific fields | Add `output_schema=SomeModel`; access `.field` not raw string |
| Heavy processing in LLMTask | Move to FunctionTask (deterministic = no LLM needed) |
| `Task(...)` | Use `LLMTask(...)` — `Task` is deprecated |

---

## Reference

- Harness project: `~/ai/projects/Harness`
- Working examples: `examples/research_report.py`, `examples/code_stats.py`, `examples/analyze_harness.py`
- Design doc: `design/v1-design.md`
- View run history: `harness runs` / `harness runs --failed`

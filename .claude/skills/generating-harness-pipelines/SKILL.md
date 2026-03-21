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
4. 输出格式有要求吗？(Markdown/JSON/特定模板？)
5. 是否需要定时运行？(cron 表达式？)
6. 错误处理预期？(遇到失败是重试/跳过/报警？)
```

**When to proceed without asking:** If the user's description already answers most questions (e.g., "研究 X 然后生成 Markdown 报告"), make reasonable assumptions and state them explicitly before coding.

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

### Step 2: Think through data flow

Each task receives `results: list[Result]` — all prior task outputs:

```
Task 0 output → results[0].output  (available in Task 1+)
Task 1 output → results[1].output  (available in Task 2+)
Task N output → results[N].output  (available in Task N+1+)
```

Watch for: does any step need data from multiple prior tasks? That's a signal for a dynamic prompt or FunctionTask aggregator.

### Step 3: Search if needed

If the pipeline involves external libraries or APIs you're not certain about (akshare, MiniMax, Hailuo, etc.):
- Use WebSearch to find the actual API signature before writing FunctionTask
- Check for async support (FunctionTask runs sync libs fine via asyncio)

### Step 4: State your design

Before writing code, write a brief pipeline diagram and confirm with user:

```
FunctionTask [数据获取] — akshare 获取 K 线，输出 DataFrame
    ↓
LLMTask [分析]         — 分析指标，输出看涨/看跌信号
    ↓
FunctionTask [保存]    — 写入 {symbol}_report.md
```

---

## Phase 3: Generate + Test

### Code Structure

```python
"""script_name.py — 一句话描述。

pipeline:
  FunctionTask [步骤] — 描述
      ↓
  LLMTask [步骤]      — 描述
      ↓
  FunctionTask [保存] — 输出到文件

运行:
    python examples/script_name.py <arg> [--option value]
"""
from __future__ import annotations
import argparse, asyncio
from pathlib import Path
from harness import Harness, LLMTask, FunctionTask, ShellTask, Parallel, PollingTask


def make_prompt(captured_var: str) -> callable:
    """Use closure to capture external variables for dynamic prompts."""
    def _prompt(results: list) -> str:
        prior = results[0].output
        return f"Based on:\n{prior}\n\nNow do X for {captured_var}"
    return _prompt


async def run(arg: str, output_dir: Path) -> None:
    h = Harness(
        project_path=".",
        stream_callback=lambda text: print(text, end="", flush=True),
    )
    pr = await h.pipeline(
        [
            LLMTask(prompt=make_prompt(arg)),
            FunctionTask(fn=make_save_fn(arg, output_dir, result_index=0)),
        ],
        name=f"pipeline-{arg[:20]}",
    )
    print(f"\n完成 | {pr.total_duration_seconds:.1f}s | tokens: {pr.total_tokens:,}")


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arg", help="...")
    parser.add_argument("--output", default=".", help="输出目录")
    args = parser.parse_args()
    asyncio.run(run(args.arg, Path(args.output).resolve()))


if __name__ == "__main__":
    main_cli()
```

### Testing Protocol

After generating, ALWAYS test — do not skip:

```bash
# 1. Syntax check (instant)
python -c "import ast; ast.parse(open('examples/script.py').read()); print('syntax OK')"

# 2. --help works
python examples/script.py --help

# 3. Real end-to-end run
python examples/script.py "test_input" --output /tmp/harness_test

# 4. Verify output
ls -la /tmp/harness_test/
head -20 /tmp/harness_test/*
```

Fix and re-run until Step 3 completes successfully end-to-end.

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

### File save with date
```python
import re
from datetime import date

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

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `lambda results: results[1].output` in Task 1 | Task 1 only sees `results[0]` |
| `prompt=lambda results: f"...{x}"` in loop | Use closure to capture `x` |
| Skipping test step | Always run a real end-to-end test |
| Heavy processing in LLMTask | Move to FunctionTask (deterministic = no LLM needed) |
| `Task(...)` | Use `LLMTask(...)` — `Task` is deprecated |

---

## Reference

- Harness project: `~/ai/projects/Harness`
- Working examples: `examples/research_report.py`, `examples/code_stats.py`, `examples/analyze_harness.py`
- Design doc: `design/v1-design.md`

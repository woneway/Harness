"""analyze_harness.py — 分析 → 优化 → 复盘 三阶段流水线。

pipeline：
  FunctionTask  ：收集源码快照
      ↓
  LLMTask [分析]：输出结构化分析报告（问题清单 + 优先级）
      ↓
  LLMTask [优化]：根据报告直接修改代码文件（bypassPermissions）
      ↓
  FunctionTask  ：收集优化后源码快照
      ↓
  LLMTask [复盘]：对比前后差异，评估改进效果

运行：
    python examples/analyze_harness.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from harness import Harness, FunctionTask, LLMTask

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# 辅助：收集源码快照
# ---------------------------------------------------------------------------


def collect_source(results: list) -> dict[str, str]:
    """读取 harness/ 下所有 .py 文件，返回 {相对路径: 内容}。"""
    root = PROJECT_ROOT / "harness"
    return {
        str(p.relative_to(PROJECT_ROOT)): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*.py"))
        if p.stat().st_size > 0
    }


def collect_source_after(results: list) -> dict[str, str]:
    """优化完成后再次收集，与 collect_source 逻辑相同。"""
    return collect_source(results)


# ---------------------------------------------------------------------------
# Task 1：分析
# ---------------------------------------------------------------------------


def make_analyze_prompt(results: list) -> str:
    sources: dict[str, str] = results[0].output
    code_block = "\n\n".join(
        f"### {path}\n```python\n{content}\n```"
        for path, content in sources.items()
    )
    return f"""你是一位资深 Python 架构师，请深度分析以下框架（Harness）的源代码。

{code_block}

---

请输出结构化分析报告，格式如下：

## 架构总览
一段话描述分层结构和各模块职责。

## 设计亮点（3条）
- 亮点1：...
- 亮点2：...
- 亮点3：...

## 问题清单（按优先级排序）
列出 3-5 个问题，每条包含：
- **[P0/P1/P2] 问题名**：问题描述 + 涉及文件

## 优化方案
针对 P0/P1 问题，给出具体可执行的修改方案（说明要改哪个文件、改什么）。

## 一句话总结
...

请用中文，保持专业简洁。"""


# ---------------------------------------------------------------------------
# Task 2：优化
# ---------------------------------------------------------------------------


def make_optimize_prompt(results: list) -> str:
    analysis: str = results[1].output
    sources: dict[str, str] = results[0].output
    file_list = "\n".join(f"- {p}" for p in sources)

    return f"""你是一位资深 Python 工程师。

以下是对 Harness 框架的分析报告：

{analysis}

---

项目根目录：{PROJECT_ROOT}
源文件清单：
{file_list}

请根据分析报告中的「优化方案」，直接对相关文件进行修改：
1. 优先处理 P0 问题，其次 P1
2. 每处修改保持最小改动，不要重构无关代码
3. 修改完成后，输出一份简短的「修改摘要」，格式：
   - 文件名：做了什么改动（一句话）

注意：你可以直接读写项目文件，项目根目录是 {PROJECT_ROOT}"""


# ---------------------------------------------------------------------------
# Task 3：复盘
# ---------------------------------------------------------------------------


def make_review_prompt(results: list) -> str:
    before: dict[str, str] = results[0].output
    analysis: str = results[1].output
    after: dict[str, str] = results[3].output

    # 找出有变化的文件
    changed = [
        path for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    ]

    if not changed:
        diff_block = "（未检测到文件变化）"
    else:
        diff_block = ""
        for path in changed:
            b = before.get(path, "（新文件）")
            a = after.get(path, "（已删除）")
            diff_block += f"\n### {path}\n**修改前**（前200字）：\n{b[:200]}\n\n**修改后**（前200字）：\n{a[:200]}\n"

    return f"""你是一位 code reviewer，请对本次优化进行复盘。

原始分析报告：
{analysis}

---

发生变化的文件：
{diff_block}

---

请输出复盘报告，包含：

## 优化完成情况
逐条评估分析报告中的问题是否得到解决（✅已解决 / ⚠️部分解决 / ❌未处理）。

## 改动质量评估
评价修改的代码质量（是否引入新问题、是否最小改动原则等）。

## 遗留问题
列出本次未处理、下次应跟进的问题。

## 总结
一句话概括本次优化的效果。

请用中文，保持客观专业。"""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def main() -> None:
    h = Harness(
        project_path=str(PROJECT_ROOT),
        stream_callback=lambda text: print(text, end="", flush=True),
    )

    print("=" * 60)
    print("  Harness 自动优化流水线：分析 → 优化 → 复盘")
    print("=" * 60)

    pr = await h.pipeline(
        [
            FunctionTask(fn=collect_source),           # Task 0
            LLMTask(prompt=make_analyze_prompt),        # Task 1：分析
            LLMTask(prompt=make_optimize_prompt),       # Task 2：优化
            FunctionTask(fn=collect_source_after),      # Task 3
            LLMTask(prompt=make_review_prompt),         # Task 4：复盘
        ],
        name="analyze-optimize-review",
    )

    print(f"\n\n{'=' * 60}")
    print(f"  完成")
    print(f"  耗时：{pr.total_duration_seconds:.1f}s  |  tokens：{pr.total_tokens:,}")
    print(f"  Run ID：{pr.run_id[:8]}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())

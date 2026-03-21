"""research_report.py — 给定主题，联网深度调研，生成专业报告并保存为 Markdown 文件。

pipeline（默认 deep 模式，5 阶段）：
  LLMTask [多维搜索]  ：8 个维度 × 多次 WebSearch，覆盖技术/商业/用户/竞品
      ↓
  LLMTask [大纲]      ：基于真实搜索结果，制定结构化调研大纲
      ↓
  LLMTask [正文]      ：按大纲逐章撰写，充分引用搜索数据与来源
      ↓
  LLMTask [专家审校]  ：事实核查 + 补充遗漏 + 标注不确定内容 + 优化表达
      ↓
  FunctionTask        ：保存 <主题>_<日期>.md，打印摘要

模式选项：
  --depth quick     : 3 阶段（搜索 → 正文 → 保存），快速概览
  --depth standard  : 4 阶段（搜索 → 大纲 → 正文 → 保存），均衡
  --depth deep      : 5 阶段（搜索 → 大纲 → 正文 → 审校 → 保存），默认，最高质量

运行：
    python examples/research_report.py Clawith
    python examples/research_report.py "大语言模型量化技术" --output ./reports
    python examples/research_report.py "Rust 异步运行时" --depth standard
    python examples/research_report.py "OpenAI o3" --depth quick
"""

from __future__ import annotations

import argparse
import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from harness import FunctionTask, Harness, LLMTask


# ---------------------------------------------------------------------------
# Task 0：多维深度搜索
# ---------------------------------------------------------------------------


def make_gather_prompt(topic: str, depth: str) -> str:
    """返回指示 LLM 进行系统性多维搜索的 prompt。"""
    today = datetime.now().strftime("%Y-%m-%d")

    if depth == "quick":
        search_instructions = f"""请使用 WebSearch 工具搜索「{topic}」，至少进行 3 次查询，覆盖：
1. 基本信息（定义、官网、最新版本）
2. 核心功能与主要用途
3. 用户评价与典型反馈

每条信息标注来源 URL。输出结构化摘要。"""
        return f"你是专业信息研究员。今天是 {today}。\n\n{search_instructions}"

    # standard / deep 共用详细搜索
    return f"""你是一位专业的信息研究员，擅长系统性信息收集与交叉验证。今天是 {today}。

请对「{topic}」进行**多维度深度调研**，按以下步骤执行 WebSearch：

## 搜索计划（必须完成全部 8 个维度）

**维度 1：基本定义与官方信息**
- 搜索：`{topic} official site`（或官方中文名）
- 搜索：`{topic} documentation overview`
- 目标：官网链接、产品定位、版本、定价页

**维度 2：技术架构与实现**
- 搜索：`{topic} architecture how it works`
- 目标：核心技术原理、技术栈、创新点

**维度 3：商业模式与市场**
- 搜索：`{topic} funding valuation business model`
- 目标：融资信息、商业模式、市场规模

**维度 4：最新动态（过去 6 个月）**
- 搜索：`{topic} 2025 2026 latest news`
- 目标：重要更新、新功能、重大事件

**维度 5：正面用户评价**
- 搜索：`{topic} review positive site:reddit.com OR site:news.ycombinator.com`
- 目标：用户喜欢的功能、成功案例

**维度 6：负面反馈与批评**
- 搜索：`{topic} criticism problems issues limitations`
- 目标：已知缺陷、用户抱怨、争议

**维度 7：竞品对比**
- 搜索：`{topic} vs alternatives competitors comparison`
- 目标：主要竞品列表、差异化对比

**维度 8：专家评测与深度分析**
- 搜索：`{topic} analysis deep dive expert review`
- 目标：行业专家观点、第三方评测

## 输出格式

按上述 8 个维度分节输出结构化摘要：
- 每节 3-6 条关键发现
- 每条信息附：来源 URL + 发布日期（如可获得）
- 最后附「信息质量评估」：
  - 信息时效性（最新数据的日期）
  - 来源权威性（官方/媒体/社区）
  - 发现的信息矛盾或不确定项

严格要求：所有数据来自搜索结果，不得凭记忆捏造任何数字或事实。"""


# ---------------------------------------------------------------------------
# Task 1：调研大纲（standard / deep 模式）
# ---------------------------------------------------------------------------


def make_outline_prompt(topic: str) -> callable:
    """闭包：返回依赖搜索结果的大纲 prompt 函数。"""

    def _prompt(results: list) -> str:
        gathered_info: str = results[0].output
        today = datetime.now().strftime("%Y-%m-%d")
        return f"""你是一位资深行业分析师。今天是 {today}。

已为你收集了关于「{topic}」的多维度真实信息：

---
{gathered_info}
---

请基于上述**真实信息**制定一份调研大纲，用于指导后续撰写完整报告。

要求：
1. 包含 6-9 个主要章节（覆盖：背景 → 产品/技术 → 商业/市场 → 用户反馈 → 竞争格局 → 风险 → 趋势展望 → 结论）
2. 每章列出 2-4 个关键分析角度，**结合搜索结果中的具体发现**（如：第 X 维度发现了 Y 数据）
3. 特别标注搜索中发现的**矛盾信息或不确定内容**，以便正文重点核实
4. 最后一章为「参考来源」，用于整理本报告引用的主要 URL

输出 Markdown 格式大纲，简洁专业，不写正文。"""

    return _prompt


# ---------------------------------------------------------------------------
# Task 2：撰写正文
# ---------------------------------------------------------------------------


def make_report_prompt(topic: str, has_outline: bool = True) -> callable:
    """闭包：返回撰写正文的 prompt 函数。

    Args:
        has_outline: True = standard/deep 模式（有大纲），False = quick 模式（无大纲）
    """

    def _prompt(results: list) -> str:
        gathered_info: str = results[0].output
        today = datetime.now().strftime("%Y-%m-%d")

        if has_outline:
            outline: str = results[1].output
            outline_section = f"""## 调研大纲
---
{outline}
---

"""
            instruction_prefix = "请严格按照大纲章节顺序展开，不遗漏任何章节。"
        else:
            outline_section = ""
            instruction_prefix = "请自行组织章节结构（背景 → 产品/技术 → 市场 → 用户反馈 → 竞品 → 结论）。"

        return f"""你是一位资深行业分析师，擅长基于实证数据撰写客观专业的调研报告。今天是 {today}。

请为「{topic}」撰写一份**完整的专业调研报告**。

{outline_section}## 真实信息参考
---
{gathered_info}
---

## 撰写要求

{instruction_prefix}

**准确性（最重要）：**
- 每个重要数据点（数字、日期、功能描述）必须源自上方「真实信息参考」
- 不确定或无法从搜索结果验证的内容，用角注标记：`*[待核实]*`
- 禁止凭记忆编造任何数字、日期或事件

**完整性：**
- 必须覆盖正面价值和负面批评，不偏向任何一方
- 竞品对比要具体（名称、差异点），不要泛泛而谈
- 每个主要结论需要至少一条证据支撑

**专业性：**
- 报告开头：「执行摘要」（150-250 字，含核心结论）
- 报告结尾：「关键结论」（5-7 条 bullet，每条一句话）+ 「局限性说明」
- 使用 Markdown 格式，标题层级清晰（# ## ###）
- 重要数据用粗体，来源 URL 用内联链接

**报告顶部元信息：**

---
title: {topic} 调研报告
date: {today}
depth: {"deep" if has_outline else "quick"}
---

请直接输出完整报告正文，不要添加额外说明。"""

    return _prompt


# ---------------------------------------------------------------------------
# Task 3：专家审校（仅 deep 模式）
# ---------------------------------------------------------------------------


def make_critique_prompt(topic: str) -> callable:
    """闭包：返回对草稿进行专家审校的 prompt 函数。"""

    def _prompt(results: list) -> str:
        gathered_info: str = results[0].output
        draft_report: str = results[2].output  # Task 2 的输出
        today = datetime.now().strftime("%Y-%m-%d")
        return f"""你是一位严谨的专业编辑，擅长事实核查和调研报告质量审查。今天是 {today}。

请对以下「{topic}」调研报告草稿进行**专家审校**，然后输出修订后的完整报告。

## 原始搜索数据（事实核查依据）
---
{gathered_info}
---

## 报告草稿
---
{draft_report}
---

## 审校任务

请按以下步骤审查并修订：

**1. 事实核查（对照原始搜索数据）**
- 找出所有无法在搜索数据中验证的具体数字/日期/事件
- 将无法验证的内容标注 `*[待核实：原文写X，搜索数据未见此来源]*`
- 将明显错误的内容直接修正

**2. 完整性补充**
- 搜索数据中有但草稿未提及的重要信息（尤其是负面批评、重要竞品）
- 将补充内容自然融入对应章节

**3. 平衡性检查**
- 如草稿过于正面（缺少批评）或过于负面，调整至客观中立
- 确保「用户反馈」章节同时包含正面和负面声音

**4. 表达优化**
- 去除空洞套话（"具有重要意义"、"值得关注"等）
- 让每句话都携带具体信息
- 确保执行摘要准确反映全文核心结论

## 输出

直接输出修订后的**完整报告**（不要写审校过程，不要解释修改了什么）。
保留原有 YAML 元信息，将 `depth` 字段值改为 `deep (reviewed)`。"""

    return _prompt


# ---------------------------------------------------------------------------
# Task N：保存报告到文件
# ---------------------------------------------------------------------------


def make_save_fn(topic: str, output_dir: Path, report_task_index: int) -> callable:
    """闭包：保存报告，文件名含日期。"""

    def _save(results: list) -> Path:
        report_content: str = results[report_task_index].output

        today = datetime.now().strftime("%Y%m%d")
        safe_name = re.sub(r'[\\/:*?"<>|]', "_", topic).strip()
        filename = f"{safe_name}_{today}.md"

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        output_path.write_text(report_content, encoding="utf-8")

        return output_path

    return _save


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def run(
    topic: str,
    output_dir: Path,
    depth: Literal["quick", "standard", "deep"],
) -> None:
    h = Harness(
        project_path=".",
        stream_callback=lambda text: print(text, end="", flush=True),
    )

    depth_labels = {"quick": "快速（3 阶段）", "standard": "均衡（4 阶段）", "deep": "深度（5 阶段）"}
    print("=" * 60)
    print(f"  调研主题：{topic}")
    print(f"  调研深度：{depth_labels[depth]}")
    print(f"  报告输出：{output_dir}")
    print("=" * 60)
    print()

    if depth == "quick":
        # 3 阶段：搜索 → 直接写报告 → 保存
        tasks = [
            LLMTask(prompt=make_gather_prompt(topic, depth)),           # Task 0
            LLMTask(prompt=make_report_prompt(topic, has_outline=False)), # Task 1
            FunctionTask(fn=make_save_fn(topic, output_dir, 1)),        # Task 2
        ]
        save_index = 2

    elif depth == "standard":
        # 4 阶段：搜索 → 大纲 → 正文 → 保存
        tasks = [
            LLMTask(prompt=make_gather_prompt(topic, depth)),           # Task 0
            LLMTask(prompt=make_outline_prompt(topic)),                  # Task 1
            LLMTask(prompt=make_report_prompt(topic, has_outline=True)), # Task 2
            FunctionTask(fn=make_save_fn(topic, output_dir, 2)),        # Task 3
        ]
        save_index = 3

    else:  # deep（默认）
        # 5 阶段：搜索 → 大纲 → 正文 → 专家审校 → 保存
        tasks = [
            LLMTask(prompt=make_gather_prompt(topic, depth)),           # Task 0
            LLMTask(prompt=make_outline_prompt(topic)),                  # Task 1
            LLMTask(prompt=make_report_prompt(topic, has_outline=True)), # Task 2
            LLMTask(prompt=make_critique_prompt(topic)),                 # Task 3
            FunctionTask(fn=make_save_fn(topic, output_dir, 3)),        # Task 4
        ]
        save_index = 4

    pr = await h.pipeline(tasks, name=f"research-{topic[:30]}")

    saved_path: Path = pr.results[save_index].output

    print(f"\n\n{'=' * 60}")
    print(f"  调研报告生成完成")
    print(f"  文件：{saved_path}")
    print(f"  耗时：{pr.total_duration_seconds:.1f}s  |  tokens：{pr.total_tokens:,}")
    print(f"  Run ID：{pr.run_id[:8]}")
    print(f"{'=' * 60}")


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="给定主题，联网深度调研，自动生成专业调研报告",
    )
    parser.add_argument("topic", help="调研主题，例如：Clawith、大语言模型量化技术")
    parser.add_argument(
        "--output",
        default=".",
        help="报告保存目录（默认：当前目录）",
    )
    parser.add_argument(
        "--depth",
        choices=["quick", "standard", "deep"],
        default="deep",
        help="调研深度：quick（3阶段快速）/ standard（4阶段均衡）/ deep（5阶段+审校，默认）",
    )
    args = parser.parse_args()

    asyncio.run(run(
        topic=args.topic,
        output_dir=Path(args.output).resolve(),
        depth=args.depth,
    ))


if __name__ == "__main__":
    main_cli()

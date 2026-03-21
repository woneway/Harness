"""research_report.py — 给定主题，联网搜索真实信息，生成调研报告并保存为 Markdown 文件。

pipeline：
  LLMTask [搜索]   ：使用 WebSearch 搜集主题的真实信息，整理为结构化摘要
      ↓
  LLMTask [大纲]   ：根据搜集到的真实信息，制定调研大纲
      ↓
  LLMTask [正文]   ：依据大纲和真实信息撰写完整调研报告（Markdown 格式）
      ↓
  FunctionTask     ：将报告保存为 <主题>.md 文件

运行：
    python examples/research_report.py Clawith
    python examples/research_report.py "大语言模型量化技术"
    python examples/research_report.py "Rust 异步运行时" --output ./reports
"""

from __future__ import annotations

import argparse
import asyncio
import re
from datetime import datetime
from pathlib import Path

from harness import FunctionTask, Harness, LLMTask


# ---------------------------------------------------------------------------
# Task 0：联网搜索真实信息
# ---------------------------------------------------------------------------


def make_gather_prompt(topic: str) -> str:
    """返回指示 LLM 使用 WebSearch 搜集信息的静态 prompt。"""
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""你是一位专业的信息研究员。今天是 {today}。

请使用 WebSearch 工具搜索「{topic}」的以下信息：

1. 官网 / GitHub / 官方文档 / 定价页面
2. 核心功能与技术架构（产品定位、解决什么问题）
3. 最新动态与版本发布（过去 12 个月的重要更新）
4. 用户评价与社区讨论（HackerNews、Reddit、ProductHunt、Twitter/X）
5. 主要竞品及差异化对比
6. 市场规模、融资情况、团队背景（如适用）

搜索要求：
- 至少进行 4 次独立的 WebSearch 查询，覆盖不同维度
- 优先使用英文关键词搜索，必要时补充中文搜索
- 每条信息标注来源 URL 和发布/更新时间

输出格式：
- 结构化摘要，按上述 6 个维度分节
- 每节包含 3-5 条关键发现，每条附来源 URL
- 最后附「信息可信度评估」（数据时效性、来源权威性）"""


# ---------------------------------------------------------------------------
# Task 1：根据真实信息制定调研大纲
# ---------------------------------------------------------------------------


def make_outline_prompt(topic: str) -> callable:
    """闭包：捕获 topic，返回依赖搜索结果的大纲 prompt 函数。"""

    def _prompt(results: list) -> str:
        gathered_info: str = results[0].output
        today = datetime.now().strftime("%Y-%m-%d")
        return f"""你是一位资深行业分析师。今天是 {today}。

已为你收集了关于「{topic}」的真实信息如下：

---
{gathered_info}
---

请基于上述真实信息，制定一份**调研大纲**，用于指导后续撰写完整调研报告：

要求：
1. 大纲应包含 5-8 个主要章节
2. 每个章节列出 2-4 个需要深入分析的关键问题（结合已收集信息中的具体发现）
3. 章节顺序符合「背景 → 产品/技术分析 → 市场与竞争格局 → 用户反馈 → 趋势与展望 → 结论」的逻辑
4. 最后一节为「数据来源」，整理本报告引用的主要 URL 列表

请用 Markdown 格式输出大纲，保持简洁专业，不要输出正文内容。"""

    return _prompt


# ---------------------------------------------------------------------------
# Task 2：根据大纲和真实信息撰写完整报告
# ---------------------------------------------------------------------------


def make_report_prompt(topic: str) -> callable:
    """闭包：捕获 topic，返回依赖前两个任务结果的报告撰写 prompt 函数。"""

    def _prompt(results: list) -> str:
        gathered_info: str = results[0].output
        outline: str = results[1].output
        today = datetime.now().strftime("%Y-%m-%d")
        return f"""你是一位资深行业分析师。今天是 {today}。

请依据以下大纲和真实信息，为主题「{topic}」撰写一份**完整的调研报告**：

## 调研大纲
---
{outline}
---

## 真实信息参考
---
{gathered_info}
---

撰写要求：
1. 严格按照大纲章节顺序展开，不遗漏任何章节
2. 充分引用「真实信息参考」中的具体数据、事实和来源 URL，确保内容有据可查
3. 不要凭空捏造数据——仅使用搜索结果中已验证的信息，不确定的内容标注「待核实」
4. 报告开头包含「执行摘要」（200 字以内），结尾包含「关键结论」（3-5 条）
5. 全文使用 Markdown 格式，标题层级清晰（# ## ###）
6. 报告元信息放在最顶部：

---
title: {topic} 调研报告
date: {today}
---

请直接输出完整报告，不要添加额外说明。"""

    return _prompt


# ---------------------------------------------------------------------------
# Task 3：保存报告到文件
# ---------------------------------------------------------------------------


def make_save_fn(topic: str, output_dir: Path) -> callable:
    """闭包：捕获 topic 和输出目录，返回文件保存函数。"""

    def _save(results: list) -> Path:
        report_content: str = results[2].output

        safe_name = re.sub(r'[\\/:*?"<>|]', "_", topic).strip()
        filename = f"{safe_name}.md"

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        output_path.write_text(report_content, encoding="utf-8")

        return output_path

    return _save


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="给定主题，联网搜索真实信息，用 Harness pipeline 自动生成调研报告",
    )
    parser.add_argument("topic", help="调研主题，例如：Clawith、大语言模型量化技术")
    parser.add_argument(
        "--output",
        default=".",
        help="报告保存目录（默认：当前目录）",
    )
    args = parser.parse_args()

    topic: str = args.topic
    output_dir = Path(args.output).resolve()

    h = Harness(
        project_path=".",
        stream_callback=lambda text: print(text, end="", flush=True),
    )

    print("=" * 60)
    print(f"  调研主题：{topic}")
    print(f"  报告输出：{output_dir}")
    print("=" * 60)
    print()

    pr = await h.pipeline(
        [
            LLMTask(prompt=make_gather_prompt(topic)),          # Task 0：联网搜索
            LLMTask(prompt=make_outline_prompt(topic)),         # Task 1：制定大纲
            LLMTask(prompt=make_report_prompt(topic)),          # Task 2：撰写正文
            FunctionTask(fn=make_save_fn(topic, output_dir)),   # Task 3：保存文件
        ],
        name="research-report",
    )

    saved_path: Path = pr.results[3].output

    print(f"\n\n{'=' * 60}")
    print(f"  调研报告生成完成")
    print(f"  文件：{saved_path}")
    print(f"  耗时：{pr.total_duration_seconds:.1f}s  |  tokens：{pr.total_tokens:,}")
    print(f"  Run ID：{pr.run_id[:8]}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())

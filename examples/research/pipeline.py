"""调研 Pipeline 组装：parse → discover → collect → discuss → report → save。"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from harness import (
    Condition,
    Discussion,
    FunctionTask,
    LLMTask,
    Parallel,
)
from harness.runners.base import AbstractRunner
from harness.tasks.discussion import all_agree_on

logger = logging.getLogger(__name__)

from .agents import create_agents
from .schemas.position import ProjectEvaluation
from .state import ResearchState
from .tasks.collect_github import collect_github
from .tasks.parse_input import parse_input

# SecondBrain 输出路径
SECONDBRAIN_DIR = Path.home() / "ai" / "SecondBrain"


def _stream_print(text: str) -> None:
    """LLMTask 流式输出回调。"""
    print(text, end="", flush=True)


def _step_banner(label: str) -> None:
    """打印步骤分隔栏。"""
    print(f"\n{'─' * 50}")
    print(f"  ▶ {label}")
    print(f"{'─' * 50}\n")


def _progress_handler(e) -> None:
    """Discussion 进度回调。"""
    if e.event == "streaming":
        print(e.content or "", end="", flush=True)
    elif e.event == "start":
        print(f"\n  [{e.event}] {e.agent_name}", flush=True)
    elif e.event == "complete":
        print(f"\n  [{e.event}] {e.agent_name}", flush=True)
    elif e.event == "error":
        content_preview = f": {e.content[:80]}" if e.content else ""
        print(f"\n  [{e.event}] {e.agent_name}{content_preview}", flush=True)


def _strip_markdown_fences(text: str) -> str:
    """剥离 markdown 代码块包裹（```json ... ``` 或 ``` ... ```）。"""
    import re
    return re.sub(r"^```\w*\n?|```\s*$", "", text.strip()).strip()


def _parse_competitors_json(state: ResearchState) -> list[dict]:
    """将 LLMTask 返回的 JSON 字符串解析为 list[dict]，写入 state.competitors。

    同时回填 target_url：从 competitors 中找到与 target_project 匹配的项目 URL。
    """
    raw = state.competitors
    if isinstance(raw, str):
        cleaned = _strip_markdown_fences(raw)
        try:
            parsed = json.loads(cleaned)
            state.competitors = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            logger.warning("Failed to parse competitors JSON: %s", cleaned[:200])
            state.competitors = []

    # 回填 target_url：从 competitors 中匹配 target_project
    if state.target_project and not state.target_url:
        target_lower = state.target_project.lower()
        for comp in state.competitors:
            if not isinstance(comp, dict):
                continue
            comp_name = comp.get("name", "").lower()
            comp_url = comp.get("url", "")
            if target_lower in comp_name or comp_name in target_lower:
                state.target_url = comp_url
                break

    if not state.competitors:
        logger.warning("No competitors found after parsing — downstream steps may have limited data")

    return state.competitors


def _save_report(state: ResearchState) -> str:
    """将报告写入 SecondBrain 知识库。"""
    import re

    if state.target_project:
        name = state.target_project
    else:
        # topic 模式：从 raw_input 生成简短文件名
        name = state.raw_input[:40]
    # 清理文件名：只保留字母数字和连字符
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"[\s_]+", "-", name).lower()
    filename = f"{name or 'research'}-research.md"
    output_path = SECONDBRAIN_DIR / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(state.report, encoding="utf-8")

    state.output_path = str(output_path)
    return str(output_path)


def _format_metrics(metrics: dict) -> str:
    """将 github_metrics 格式化为可读文本。"""
    if not metrics:
        return "暂无数据"
    lines = []
    for name, data in metrics.items():
        if isinstance(data, dict) and "error" not in data:
            lines.append(
                f"### {name}\n"
                f"- Stars: {data.get('stars', 'N/A')} | Forks: {data.get('forks', 'N/A')}\n"
                f"- Language: {data.get('language', 'N/A')} | License: {data.get('license', 'N/A')}\n"
                f"- Open Issues: {data.get('open_issues', 'N/A')}\n"
                f"- Description: {data.get('description', 'N/A')}\n"
                f"- Last Push: {data.get('pushed_at', 'N/A')}"
            )
        else:
            lines.append(f"### {name}\n- 数据获取失败")
    return "\n\n".join(lines)


def build_pipeline(runner: AbstractRunner) -> tuple[list, ResearchState]:
    """构建调研 pipeline，返回 (steps, state)。"""
    tech_analyst, community_assessor, architect = create_agents(runner)
    state = ResearchState()
    today = date.today().isoformat()

    def _parse_with_banner(state: ResearchState) -> str:
        _step_banner("Step 0: 解析输入")
        result = parse_input(state)
        print(f"  类型: {state.input_type} | 目标: {state.target_project or state.raw_input}")
        if state.competitors:
            print(f"  对比: {', '.join(c.get('name', '?') for c in state.competitors if isinstance(c, dict))}")
        return result

    def _collect_with_banner(state: ResearchState) -> str:
        _step_banner("Step 2a: 采集 GitHub 数据")
        result = collect_github(state)
        for name, data in state.github_metrics.items():
            if isinstance(data, dict) and "error" not in data:
                print(f"  {name}: ⭐{data.get('stars', '?')} | 🍴{data.get('forks', '?')}")
            else:
                print(f"  {name}: 获取失败")
        return result

    steps = [
        # Step 0: 解析输入
        FunctionTask(fn=_parse_with_banner),

        # Step 1: 发现竞品（仅 topic/project 模式需要）
        Condition(
            check=lambda state: state.input_type in ("topic", "project") and not state.competitors,
            if_true=[
                FunctionTask(fn=lambda state: _step_banner("Step 1: 搜索同类项目") or ""),
                LLMTask(
                    prompt=lambda state: (
                        f"我需要调研{'主题: ' + state.raw_input if state.input_type == 'topic' else '项目: ' + state.target_project}。\n\n"
                        f"请搜索并找出这个{'领域' if state.input_type == 'topic' else '项目'}的 3-5 个最相关的开源项目"
                        f"{'（包括该项目本身的 GitHub URL）' if state.input_type == 'project' else ''}。\n\n"
                        f"对每个项目，给出：\n"
                        f"1. 项目名称\n"
                        f"2. GitHub URL（owner/repo 格式）\n"
                        f"3. 一句话描述\n\n"
                        f"以 JSON 数组格式输出，每项包含 name, url, description 字段。\n"
                        f"只输出 JSON，不要其他文字。"
                    ),
                    output_key="competitors",
                    stream_callback=_stream_print,
                ),
                FunctionTask(fn=_parse_competitors_json),
            ],
        ),

        # Step 2: 并行采集数据
        Parallel([
            # 2.0: GitHub 定量指标
            FunctionTask(fn=_collect_with_banner),  # mutates state.github_metrics
            # 2.1: 定性信息（由 Claude 搜索和分析）
            LLMTask(
                prompt=lambda state: (
                    f"请对以下开源项目进行详细的定性调研：\n\n"
                    f"主项目：{state.target_project}\n"
                    f"对比项目：{', '.join(c.get('name', '') for c in state.competitors if isinstance(c, dict)) if state.competitors else '无'}\n\n"
                    f"对每个项目，请通过搜索和阅读官方文档/README，分析：\n"
                    f"1. 核心功能和设计理念\n"
                    f"2. 技术架构和关键抽象\n"
                    f"3. 文档质量和入门体验\n"
                    f"4. 社区评价和常见反馈\n"
                    f"5. 典型使用场景\n\n"
                    f"请详细输出每个项目的分析。"
                ),
                output_key="qualitative_info",
                stream_callback=_stream_print,
            ),
        ]),

        # Step 3: 三 Agent 结构化讨论
        FunctionTask(fn=lambda state: _step_banner("Step 3: 多 Agent 讨论") or ""),
        Discussion(
            agents=[tech_analyst, community_assessor, architect],
            position_schema=ProjectEvaluation,
            topic="评估对比开源项目及其同类竞品",
            background=lambda state: (
                f"## 调研目标\n{state.raw_input}\n\n"
                f"## GitHub 指标\n{_format_metrics(state.github_metrics)}\n\n"
                f"## 定性分析\n{state.qualitative_info[:3000]}"
            ),
            max_rounds=3,
            convergence=all_agree_on("recommendation"),
            progress_callback=_progress_handler,
            output_key="discussion",
        ),

        # Step 4: 生成最终报告
        FunctionTask(fn=lambda state: _step_banner("Step 4: 生成报告") or ""),
        LLMTask(
            prompt=lambda state: (
                f"基于以下调研数据和多 Agent 讨论结果，生成一份完整的 Markdown 调研报告。\n\n"
                f"## 要求\n"
                f"1. 报告必须以 YAML front matter 开头（见下方格式）\n"
                f"2. 包含 6 个标准章节：项目概览、核心功能、社区活跃度、同类项目对比、优劣势分析、适用场景推荐\n"
                f"3. 使用表格展示对比数据\n"
                f"4. 中文撰写，技术术语保留英文\n\n"
                f"## YAML Front Matter 格式\n"
                f"```yaml\n"
                f"---\n"
                f"title: \"{state.target_project} 调研 — [一句话描述]\"\n"
                f"tags: [{state.target_project}, research, open-source]\n"
                f"created: {today}\n"
                f"updated: {today}\n"
                f"type: research\n"
                f"status: active\n"
                f"---\n"
                f"```\n\n"
                f"## GitHub 指标数据\n{_format_metrics(state.github_metrics)}\n\n"
                f"## 定性分析\n{state.qualitative_info[:2000]}\n\n"
                f"## Discussion 结果\n"
                f"收敛: {state.discussion.converged if state.discussion else 'N/A'}\n"
                f"轮数: {state.discussion.rounds_completed if state.discussion else 'N/A'}\n\n"
                f"最终立场：\n"
                + (
                    "\n".join(
                        f"- {name}: {pos.project_name} 评分 {pos.score}/10 — {pos.recommendation}\n"
                        f"  优势: {', '.join(pos.strengths)}\n"
                        f"  劣势: {', '.join(pos.weaknesses)}\n"
                        f"  最佳场景: {pos.best_for}"
                        for name, pos in (state.discussion.final_positions or {}).items()
                    )
                    if state.discussion and state.discussion.final_positions
                    else "无讨论结果"
                )
            ),
            output_key="report",
            stream_callback=_stream_print,
        ),

        # Step 5: 保存到 SecondBrain
        FunctionTask(fn=_save_report, output_key="output_path"),
    ]

    return steps, state

"""调研 Pipeline 组装：parse → discover → collect → discuss → report → save。"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

from harness import (
    Condition,
    Discussion,
    DiscussionOutput,
    FunctionTask,
    LLMTask,
)
from harness.runners.base import AbstractRunner
from harness.tasks.discussion import all_agree_on

from .agents import create_agents
from .schemas.position import ProjectEvaluation
from .state import ResearchState
from .tasks.collect_github import collect_github
from .tasks.parse_input import parse_input

logger = logging.getLogger(__name__)


def _create_extraction_runner() -> AbstractRunner | None:
    """尝试创建便宜的 Haiku runner 用于 Discussion Phase 2 立场提取。

    如果 ANTHROPIC_API_KEY 未设置则返回 None，复用默认 runner。
    """
    try:
        from harness.runners.anthropic import AnthropicRunner
    except ImportError:
        logger.warning("AnthropicRunner 不可用（缺少依赖），Discussion Phase 2 将使用默认 runner")
        return None
    try:
        return AnthropicRunner(model="claude-haiku-4-5-20251001")
    except ValueError:
        logger.info("ANTHROPIC_API_KEY 未设置，Discussion Phase 2 将使用默认 runner")
        return None

# SecondBrain 输出路径
SECONDBRAIN_DIR = Path.home() / "ai" / "SecondBrain"


def _stream_print(text: str) -> None:
    """LLMTask 流式输出回调。"""
    print(text, end="", flush=True)


def _step_banner(label: str) -> str:
    """打印步骤分隔栏。"""
    print(f"\n{'─' * 50}")
    print(f"  ▶ {label}")
    print(f"{'─' * 50}\n")
    return ""


def _progress_handler(e) -> None:
    """Discussion 进度回调。"""
    if e.event == "streaming":
        # streaming 输出实时显示，带 agent 名前缀
        print(f"  [{e.agent_name}] {e.content or ''}", end="", flush=True)
    elif e.event == "tool":
        # 工具调用：已由 executor 格式化为可读描述
        print(f"\n  [{e.agent_name}] 🔧 {e.content}", flush=True)
    elif e.event == "phase":
        print(f"\n  [{e.agent_name}] 📋 {e.content}", flush=True)
    elif e.event == "start":
        print(f"\n  ⏳ [{e.agent_name}] 开始发言...", flush=True)
    elif e.event == "complete":
        print(f"\n  ✅ [{e.agent_name}] 发言完成", flush=True)
    elif e.event == "error":
        content_preview = f": {e.content[:80]}" if e.content else ""
        print(f"\n  ❌ [{e.agent_name}] {content_preview}", flush=True)


def _strip_markdown_fences(text: str) -> str:
    """剥离 markdown 代码块包裹，并处理 JSON 前的非 JSON 前缀说明文字。

    LLM 可能输出：
        "edict 这个名称对应... \n```json\n[...]\n```\n    先去 fence，再从第一个 `[` 或 `{` 开始截取。
    """
    stripped = re.sub(r"^```\w*\n?|```\s*$", "", text.strip())
    # 从第一个 [ 或 { 开始截取（处理 LLM 前缀说明文字）
    match = re.search(r"[\[{]", stripped)
    if match:
        stripped = stripped[match.start():]
    return stripped.strip()


def _recover_partial_json_array(text: str) -> list[dict]:
    """从截断的 JSON 数组中恢复完整的对象。

    当 LLM 输出被截断时（如 token 限制），使用 json.JSONDecoder.raw_decode
    逐个提取完整的 JSON 对象（支持嵌套结构）。
    """
    recovered: list[dict] = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        try:
            obj, end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict) and "name" in obj:
                recovered.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx += 1
    return recovered


def _parse_competitors_json(state: ResearchState) -> list[dict]:
    """将 LLMTask 返回的 JSON 字符串解析为 list[dict]，写入 state.competitors。

    三级容错：直接解析 → 剥离 markdown → 提取完整对象（截断恢复）。
    同时回填 target_url：从 competitors 中找到与 target_project 匹配的项目 URL。
    """
    raw = state.competitors
    if isinstance(raw, str):
        cleaned = _strip_markdown_fences(raw)
        # Level 1: 直接解析
        try:
            parsed = json.loads(cleaned)
            state.competitors = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            # Level 2: 截断恢复 — 提取所有完整的 JSON 对象
            recovered = _recover_partial_json_array(cleaned)
            if recovered:
                logger.warning(
                    "JSON 被截断，恢复了 %d 个完整对象（原文 %d 字符）",
                    len(recovered), len(cleaned),
                )
                state.competitors = recovered
            else:
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


def _format_competitors(competitors: list[dict]) -> str:
    """格式化竞品列表。"""
    if not competitors:
        return "暂无"
    return "\n".join(
        f"- {c.get('name', '?')}: {c.get('url', '')} — {c.get('description', '')}"
        for c in competitors if isinstance(c, dict)
    )


def _format_discussion_insights(discussion: DiscussionOutput | None) -> str:
    """整合结构化立场和定性分析，按 Agent 分组输出。

    每个 Agent 的输出包含：
    1. 结构化立场（评分、推荐理由、优劣势、最佳场景）
    2. 最终轮文字分析（定性补充）
    """
    if not discussion or not discussion.final_positions:
        return "无讨论数据"

    lines = []

    for agent_name, pos in discussion.final_positions.items():
        # 1. 结构化立场（BaseModel → dict）
        if hasattr(pos, "model_dump"):
            pos_dict = pos.model_dump()
        elif hasattr(pos, "dict"):
            pos_dict = pos.dict()  # noqa: PTH208
        else:
            pos_dict = dict(pos) if isinstance(pos, dict) else {}

        pos_lines = [
            f"### {agent_name}",
            f"**评分**: {pos_dict.get('score', '?')}/10 | **{pos_dict.get('recommendation', '?')}**",
            f"**推荐理由**: {pos_dict.get('project_name', '')}",
        ]
        strengths = pos_dict.get("strengths", [])
        if strengths:
            pos_lines.append(f"**优势**: {', '.join(strengths)}")
        weaknesses = pos_dict.get("weaknesses", [])
        if weaknesses:
            pos_lines.append(f"**劣势**: {', '.join(weaknesses)}")
        best_for = pos_dict.get("best_for", "")
        if best_for:
            pos_lines.append(f"**最佳场景**: {best_for}")

        lines.append("\n".join(pos_lines))

        # 2. 定性分析补充：从 turns 中找该 Agent 最后轮的文字
        agent_turns = [t for t in discussion.turns if t.agent_name == agent_name]
        if agent_turns:
            last_turn = max(agent_turns, key=lambda t: t.round)
            snippet = last_turn.response[:600]
            if len(last_turn.response) > 600:
                cut = snippet.rsplit("。", 1)
                snippet = (cut[0] + "…" if len(cut) > 1 else snippet + "…")
            lines.append(f"\n*分析摘要*: {snippet}\n")

    if not lines:
        return "无讨论数据"

    # 收敛状态
    if discussion.converged:
        lines.append(f"\n**讨论结论**: 收敛于第 {discussion.convergence_round or ''} 轮，各方达成共识。")
    else:
        lines.append(f"\n**讨论结论**: 经 {discussion.rounds_completed} 轮讨论，尚未收敛，请参考各方立场自行判断。")

    return "\n\n".join(lines)


def build_pipeline(runner: AbstractRunner) -> tuple[list, ResearchState]:
    """构建调研 pipeline，返回 (steps, state)。"""
    tech_analyst, community_assessor, architect = create_agents(runner)
    state = ResearchState()
    today = date.today().isoformat()
    extraction_runner = _create_extraction_runner()

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
                FunctionTask(fn=lambda state: _step_banner("Step 1: 搜索同类项目")),
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

        # Step 2: 采集 GitHub 定量指标
        FunctionTask(fn=_collect_with_banner),

        # Step 3: 三 Agent 结构化讨论
        # （agents 自带 Claude CLI 工具链，可自行搜索和分析项目，无需预先采集定性数据）
        FunctionTask(fn=lambda state: _step_banner("Step 3: 多 Agent 讨论")),
        Discussion(
            agents=[tech_analyst, community_assessor, architect],
            position_schema=ProjectEvaluation,
            topic=lambda state: (
                f"评估 {state.target_project or state.raw_input} 及其同类竞品"
            ),
            background=lambda state: (
                f"## 调研目标\n{state.raw_input}\n\n"
                f"## 竞品\n{_format_competitors(state.competitors)}\n\n"
                f"## GitHub 指标\n{_format_metrics(state.github_metrics)}"
            ),
            max_rounds=2,
            convergence=all_agree_on("recommendation"),
            extraction_runner=extraction_runner,
            progress_callback=_progress_handler,
            output_key="discussion",
        ),

        # Step 4: 生成最终报告
        FunctionTask(fn=lambda state: _step_banner("Step 4: 生成报告")),
        LLMTask(
            prompt=lambda state: (
                f"基于以下调研数据和多 Agent 讨论结果，生成一份完整的 Markdown 调研报告。\n\n"
                f"## 要求\n"
                f"1. 报告必须以 YAML front matter 开头（见下方格式）\n"
                f"2. 包含 6 个标准章节：项目概览、核心功能、社区活跃度、同类项目对比、优劣势分析、适用场景推荐\n"
                f"3. 使用表格展示对比数据\n"
                f"4. 中文撰写，技术术语保留英文\n"
                f"5. 充分利用 Agent 的结构化分析（评分、优劣势、最佳场景），结合 GitHub 指标数据\n\n"
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
                f"## Agent 分析（结构化立场 + 定性分析）\n{_format_discussion_insights(state.discussion)}\n"
            ),
            output_key="report",
            stream_callback=_stream_print,
        ),

        # Step 5: 保存到 SecondBrain
        FunctionTask(fn=_save_report, output_key="output_path"),
    ]

    return steps, state

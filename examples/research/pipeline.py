"""调研 Pipeline 组装：parse → discuss → record → save。

Agent 自主调研模式：不预采集数据，4 个 Agent 各自使用工具搜索、分析、辩论。
记录员基于全量讨论内容撰写最终报告。
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

from harness import (
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
from .tasks.parse_input import parse_input

logger = logging.getLogger(__name__)

# ── 输出路径 ───────────────────────────────────────────────────────────────

DIARY_DIR = Path.home() / "ai" / "SecondBrain" / "Inbox" / "调研内容"
REPORT_DIR = Path.home() / "ai" / "SecondBrain" / "Knowledge" / "Areas" / "Tech"


def _create_extraction_runner() -> AbstractRunner | None:
    """尝试创建便宜的 Haiku runner 用于 Discussion Phase 2 立场提取。"""
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


# ── UI 辅助 ────────────────────────────────────────────────────────────────

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
        print(f"  [{e.agent_name}] {e.content or ''}", end="", flush=True)
    elif e.event == "tool":
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


# ── 讨论格式化（全量，不截断） ─────────────────────────────────────────────

def _format_full_discussion(discussion: DiscussionOutput | None) -> str:
    """将全部讨论内容格式化为 Markdown，完整保留所有发言和立场。"""
    if not discussion or not discussion.turns:
        return "无讨论数据"

    sections: list[str] = []

    # 按轮次展示完整讨论过程
    for round_num in range(discussion.rounds_completed):
        round_turns = [t for t in discussion.turns if t.round == round_num]
        parts = [f"### 第 {round_num + 1} 轮"]
        for turn in round_turns:
            parts.append(f"#### {turn.agent_name}")
            parts.append(turn.response)
            # 附带该轮结构化立场摘要
            if turn.position and hasattr(turn.position, "model_dump"):
                pos = turn.position.model_dump()
                stance = (
                    f"> **立场**: {pos.get('score', '?')}/10 | "
                    f"{pos.get('recommendation', '?')} | "
                    f"最佳场景: {pos.get('best_for', '?')}"
                )
                risks = pos.get("risks", [])
                if risks:
                    stance += f"\n> **风险**: {', '.join(risks)}"
                parts.append(stance)
        sections.append("\n\n".join(parts))

    # 立场演变
    if discussion.position_history:
        evolution_lines = ["### 立场演变"]
        for agent_name, positions in discussion.position_history.items():
            trail = " → ".join(
                f"R{i}({p.score}/{p.recommendation})"
                if hasattr(p, "score") else f"R{i}(?)"
                for i, p in enumerate(positions)
            )
            evolution_lines.append(f"- **{agent_name}**: {trail}")
        sections.append("\n".join(evolution_lines))

    # 收敛状态
    if discussion.converged:
        sections.append(
            f"**收敛**: 第 {discussion.convergence_round} 轮达成共识"
        )
    else:
        sections.append(
            f"**未收敛**: 经 {discussion.rounds_completed} 轮讨论，"
            f"各方立场仍有分歧"
        )

    return "\n\n".join(sections)


# ── 讨论日记持久化 ──────────────────────────────────────────────────────────

def _save_discussion_diaries(state: ResearchState) -> str:
    """将每轮每个 Agent 的完整发言保存到 SecondBrain/Inbox/调研内容/。"""
    discussion = state.discussion
    if not discussion or not discussion.turns:
        return "无讨论内容可保存"

    project_name = _sanitize_filename(
        state.target_project or state.raw_input[:40]
    )
    today = date.today().isoformat()
    diary_dir = DIARY_DIR / f"{project_name}-{today}"
    diary_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for turn in discussion.turns:
        agent_slug = _sanitize_filename(turn.agent_name)
        filename = f"round_{turn.round}_{agent_slug}.md"
        filepath = diary_dir / filename

        # 组装日记内容
        lines = [
            f"# {turn.agent_name} — 第 {turn.round + 1} 轮",
            f"",
            f"**项目**: {state.target_project or state.raw_input}",
            f"**日期**: {today}",
            f"",
            f"---",
            f"",
            turn.response,
        ]

        # 附结构化立场
        if turn.position and hasattr(turn.position, "model_dump"):
            pos = turn.position.model_dump()
            lines.extend([
                f"",
                f"---",
                f"",
                f"## 结构化立场",
                f"- **评分**: {pos.get('score', '?')}/10",
                f"- **推荐**: {pos.get('recommendation', '?')}",
                f"- **优势**: {', '.join(pos.get('strengths', []))}",
                f"- **劣势**: {', '.join(pos.get('weaknesses', []))}",
                f"- **风险**: {', '.join(pos.get('risks', []))}",
                f"- **最佳场景**: {pos.get('best_for', '?')}",
            ])

        filepath.write_text("\n".join(lines), encoding="utf-8")
        saved.append(str(filepath))

    logger.info("讨论日记已保存到 %s（%d 个文件）", diary_dir, len(saved))
    print(f"  讨论日记已保存: {diary_dir} ({len(saved)} 个文件)")
    return str(diary_dir)


def _sanitize_filename(name: str) -> str:
    """清理字符串为安全的文件/目录名。"""
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"[\s_]+", "-", name).lower()
    return name or "unknown"


# ── 报告保存 ───────────────────────────────────────────────────────────────

def _save_report(state: ResearchState) -> str:
    """将最终报告写入 SecondBrain/Knowledge/Areas/{area}/。

    解析两段式文本：
    - 第一行：{"area": "...", "title": "..."}  JSON
    - 第二行起：Markdown 报告正文
    """
    raw = str(state.report).strip()
    lines = raw.split("\n")

    # 解析第一行的 JSON
    area = ""
    title = ""
    if lines:
        first_line = lines[0].strip()
        try:
            import json as _json

            meta = _json.loads(first_line)
            area = meta.get("area", "")
            title = meta.get("title", "")
        except Exception:
            pass

    # 降级：从 YAML front matter 中提 title
    if not title:
        for line in lines[1:10]:
            m = re.search(r'^title:\s*["\']?([^"\']+)["\']?', line)
            if m:
                title = m.group(1).strip()
                break

    # 降级：从 state.area 或默认
    area = area.strip() if area else (state.area.strip() if state.area else "Tech")
    if not area:
        area = "Tech"

    state.area = area

    # Markdown 正文：跳过第一行 JSON，从第二行开始
    if len(lines) > 1:
        report_content = "\n".join(lines[1:]).strip()
    else:
        report_content = raw  # 解析失败就用全文

    # 生成文件名
    if title:
        safe_title = _sanitize_filename(title)
        filename = f"{safe_title}.md"
    else:
        project_name = _sanitize_filename(
            state.target_project or state.raw_input[:40]
        )
        filename = f"{project_name}-research.md"

    output_path = REPORT_DIR / area / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content, encoding="utf-8")

    state.output_path = str(output_path)
    print(f"  报告已保存: {output_path} (area={area}, title={title!r})")
    return str(output_path)


# ── Pipeline 构建 ──────────────────────────────────────────────────────────

def build_pipeline(runner: AbstractRunner) -> tuple[list, ResearchState]:
    """构建调研 pipeline，返回 (steps, state)。"""
    tech_diver, eco_observer, devil_advocate, advisor = create_agents(runner)
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

    def _build_discussion_background(state: ResearchState) -> str:
        """构建 Discussion background，只给方向，不预消化数据。"""
        parts = [f"## 调研目标\n{state.raw_input}"]

        if state.input_type == "comparison" and state.competitors:
            # comparison 模式：用户已指定对比项目
            names = [state.target_project] + [
                c.get("name", "?") for c in state.competitors
                if isinstance(c, dict)
            ]
            parts.append(f"## 用户指定的对比项目\n{', '.join(names)}")
        elif state.target_url:
            parts.append(f"## 目标项目 URL\n{state.target_url}")

        parts.append(
            "## 调研要求\n"
            "- 每位 Agent 必须自行使用工具（WebSearch、gh api 等）查询所需数据\n"
            "- 不要凭印象或已有知识回答，必须用工具验证\n"
            "- 发言中详细记录你查阅了哪些资料、发现了什么具体数据\n"
            "- 用事实和数据支持你的立场\n"
            "- 如果是单个项目调研，请自行搜索同类竞品进行对比"
        )
        return "\n\n".join(parts)

    steps = [
        # Step 0: 解析输入
        FunctionTask(fn=_parse_with_banner),

        # Step 1: 多 Agent 自主调研与辩论
        FunctionTask(fn=lambda state: _step_banner("Step 1: 多 Agent 调研与辩论")),
        Discussion(
            agents=[tech_diver, eco_observer, devil_advocate, advisor],
            position_schema=ProjectEvaluation,
            topic=lambda state: (
                f"评估 {state.target_project or state.raw_input}"
                + (f" 及其竞品" if state.input_type != "comparison" else "")
            ),
            background=_build_discussion_background,
            max_rounds=2,
            convergence=all_agree_on("recommendation"),
            extraction_runner=extraction_runner,
            progress_callback=_progress_handler,
            output_key="discussion",
        ),

        # Step 2: 保存讨论日记
        FunctionTask(fn=_save_discussion_diaries),

        # Step 3: 记录员撰写最终报告
        FunctionTask(fn=lambda state: _step_banner("Step 2: 记录员撰写报告")),
        LLMTask(
            prompt=lambda state: (
                f"你是一位专业的技术调研记录员。以下是 4 位专家对"
                f"「{state.target_project or state.raw_input}」的完整讨论记录。\n\n"
                f"## 输出格式（必须严格遵守）\n"
                f"第一行输出 JSON 对象，包含 area 和 title 两个字段，不要有任何前缀或解释文字：\n"
                f'{{"area": "Tech", "title": "OpenClaw 调研报告"}}\n'
                f"第二行起，输出完整的 Markdown 调研报告正文。\n\n"
                f"## area 可选值\n"
                f"Tech（技术/开发）、Life（生活）、Finance（金融）、Health（健康）、\n"
                f"Learning（学习）、Entertainment（娱乐）、Business（商业）、Other（其他）\n\n"
                f"## 报告要求\n"
                f"1. **保留所有有价值信息**：专家查到的数据、代码证据、案例都要写进报告\n"
                f"2. **忠实呈现分歧**：如果专家意见不一致，双方论据都要呈现\n"
                f"3. **魔鬼代言人的风险必须单独成章**：不能淡化或忽略\n"
                f"4. **使用表格展示对比数据**（Stars、Forks、License 等）\n"
                f"5. **中文撰写**，技术术语保留英文\n\n"
                f"## 报告结构\n"
                f"1. YAML front matter\n"
                f"2. 项目概览\n"
                f"3. 技术分析（架构、API、代码质量）\n"
                f"4. 生态评估（社区、安全、可持续性）\n"
                f"5. 风险与争议（魔鬼代言人的发现）\n"
                f"6. 竞品对比（表格）\n"
                f"7. 选型建议（按场景区分）\n\n"
                f"## YAML Front Matter 格式\n"
                f"```yaml\n"
                f"---\n"
                f"title: \"调研报告标题\"\n"
                f"tags: [{state.target_project or 'research'}, research, open-source]\n"
                f"created: {today}\n"
                f"updated: {today}\n"
                f"type: research\n"
                f"status: active\n"
                f"---\n"
                f"```\n\n"
                f"## 完整讨论记录\n\n{_format_full_discussion(state.discussion)}\n"
            ),
            output_key="report",
            stream_callback=_stream_print,
        ),

        # Step 4: 保存最终报告
        FunctionTask(fn=_save_report, output_key="output_path"),
    ]

    return steps, state

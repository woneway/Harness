"""debate_ai_tools/main.py — MCP vs Skills vs CLI 三角色辩论

辩题：CLI、MCP、Skills 谁会成为 AI 主流工作方式？
  - MCP（Model Context Protocol）：开放协议派，标准化即未来
  - Skills（SKILL.md 提示词工作流）：AI 原生范式，最低门槛赢天下
  - CLI（传统命令行/Unix 工具链）：底层永不过时，组合能力无可替代

pipeline:
  Dialogue [三方辩论] — 轮次对辩，实时 stream → stderr
      ↓
  LLMTask  [主持人总结] — 裁定谁更可能成为主流
      ↓
  FunctionTask [保存] — 辩论记录 + 总结 → reports/debate_YYYYMMDD_HHMMSS.md

运行（必须从终端直接运行，不能在 Claude Code 内嵌套执行）:
    uv run python examples/debate_ai_tools/main.py

    实时查看 streaming 日志：
    uv run python examples/debate_ai_tools/main.py 2>/tmp/debate.log & tail -f /tmp/debate.log
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# 清除 Claude Code 相关环境变量，避免嵌套会话问题
for _key in ["CLAUDECODE", "CLAUDE_CODE", "CLAUDE_SESSION"]:
    os.environ.pop(_key, None)

from harness import Dialogue, FunctionTask, Harness, LLMTask, Role, result_by_type
from harness._internal.dialogue import DialogueContext
from harness.task import DialogueOutput, DialogueProgressEvent


# ---------------------------------------------------------------------------
# 角色 Prompt 函数
# ---------------------------------------------------------------------------


def make_mcp_prompt(ctx: DialogueContext) -> str:
    if not ctx.history:
        return """你叫 MCP，代表"开放协议派"，主张 MCP 将成为 AI 时代的主流工作方式。

核心论点：
1. **历史规律** — TCP/IP、HTTP、REST 无一不是"先复杂后普及"，协议标准化是所有技术主流化的必经之路
2. **生态锁定** — 3000+ MCP Server、十语言 SDK、Linux Foundation 治理，生态一旦形成无法逆转
3. **跨模型互操作** — Skills 绑定 Claude，CLI 绑定本地；只有 MCP 能让同一套工具跑在任何模型上
4. **企业级可信** — 可审计、可回滚、有错误码，Skills 和 CLI 都无法满足合规要求

请开场阐述：为什么 MCP 注定成为 AI 主流，而不只是"更好的扩展技术"。150-200字，观点鲜明。"""

    rebuttals = []
    skills_last = ctx.last_from("Skills")
    cli_last = ctx.last_from("CLI")
    if skills_last:
        rebuttals.append(f"Skills 说：{skills_last[-300:]}")
    if cli_last:
        rebuttals.append(f"CLI 说：{cli_last[-300:]}")
    context = "\n\n".join(rebuttals) if rebuttals else "还没有人发言。"

    return f"""你叫 MCP，这是第 {ctx.round + 1} 轮辩论。

请基于以下发言进行反驳：
{context}

你的核心论点：
- MCP 是协议标准，跨平台互操作才是王道
- Skills 只能读提示词，不能真正连接外部世界
- CLI 太底层，没有标准接口，碎片化严重

150-200字。"""


def make_skills_prompt(ctx: DialogueContext) -> str:
    if not ctx.all_from("Skills"):
        return """你叫 Skills，代表"AI 原生范式派"，主张 Skills 将成为 AI 时代的主流工作方式。

核心论点：
1. **主流从不属于复杂的** — 赢得大多数用户的永远是最低门槛的方案：一个 SKILL.md，写完即用
2. **AI 原生** — Skills 是为 AI 设计的，不是把旧工具包一层协议；模型越强，Skills 执行越好
3. **普及速度** — 全球有多少人会配 MCP Server？会写 Shell 脚本？会写 Markdown 的人多几个量级
4. **定义主流的是用户数量** — 不是企业合规，不是技术深度；90% 的 AI 使用场景不需要双向集成

请开场阐述：为什么主流不属于最强的技术，而属于最容易被采用的技术，而那就是 Skills。150-200字，观点鲜明。"""

    rebuttals = []
    mcp_last = ctx.last_from("MCP")
    cli_last = ctx.last_from("CLI")
    if mcp_last:
        rebuttals.append(f"MCP 说：{mcp_last[-300:]}")
    if cli_last:
        rebuttals.append(f"CLI 说：{cli_last[-300:]}")
    context = "\n\n".join(rebuttals) if rebuttals else "还没有人发言。"

    return f"""你叫 Skills，这是第 {ctx.round + 1} 轮辩论。

请基于以下发言进行反驳：
{context}

你的核心论点：
- MCP 复杂、需要服务器配置、有安全风险
- CLI 功能太底层，普通用户无法使用
- Skills 才是真正的"让 AI 做你希望做的事"

150-200字。"""


def make_cli_prompt(ctx: DialogueContext) -> str:
    if not ctx.all_from("CLI"):
        return """你是 CLI，代表"Unix 哲学永恒派"，主张命令行工具链将始终是 AI 时代的主流工作方式。

核心论点：
1. **主流从未改变** — Linux 运行 96% 的服务器，Shell 脚本是全球最多人写的"程序"，这就是主流
2. **组合能力无可替代** — `pipe`、`grep`、`awk`：小工具任意组合，比任何协议都灵活
3. **AI 只是新的调用者** — MCP 和 Skills 的执行层最终还是 CLI；AI 把自然语言翻译成命令，CLI 才是真正的执行者
4. **不依赖任何单一厂商** — MCP 是 Anthropic 的，Skills 是 Claude Code 的；CLI 属于所有人

请开场阐述：为什么 AI 的崛起不会取代 CLI，反而会让 CLI 成为 AI 时代更重要的基础设施。150-200字，观点鲜明。"""

    rebuttals = []
    mcp_last = ctx.last_from("MCP")
    skills_last = ctx.last_from("Skills")
    if mcp_last:
        rebuttals.append(f"MCP 说：{mcp_last[-300:]}")
    if skills_last:
        rebuttals.append(f"Skills 说：{skills_last[-300:]}")
    context = "\n\n".join(rebuttals) if rebuttals else "还没有人发言。"

    return f"""你是 CLI，这是第 {ctx.round + 1} 轮辩论。

请基于以下发言进行反驳：
{context}

你的核心论点：
- Skills 不过是提示词，没有真正的能力
- MCP 的协议栈增加了复杂性和故障点
- 我是所有扩展技术的底层承载，没有我什么都没有

150-200字。"""


def debate_until_round(ctx: DialogueContext) -> bool:
    # 每轮三方都说完后检查轮次，5 轮后结束
    return ctx.round >= 4


# ---------------------------------------------------------------------------
# 回调：进度 + streaming
# ---------------------------------------------------------------------------

_ICONS = {"MCP": "🔗", "Skills": "📝", "CLI": "💻"}


def on_progress(evt: DialogueProgressEvent) -> None:
    icon = _ICONS.get(evt.role_name, "•")
    if evt.event == "start":
        sys.stderr.write(f"\n{icon} [{evt.role_name}] 第 {evt.round_or_turn + 1} 轮发言中...\n")
    elif evt.event == "complete":
        sys.stderr.write("   ✓ 完成\n")
    elif evt.event == "error":
        sys.stderr.write(f"\n❌ [{evt.role_name}] 出错: {evt.content}\n")
    sys.stderr.flush()


def on_stream(role_name: str, chunk: str) -> None:
    sys.stderr.write(chunk)
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# LLMTask：主持人总结
# ---------------------------------------------------------------------------


def make_summary_prompt(results: list) -> str:
    output: DialogueOutput = result_by_type(results, "dialogue").output
    turns_text = "\n\n".join(
        f"【{t.role_name} · 第 {t.round + 1} 轮】\n{t.content}"
        for t in output.turns
    )
    return f"""你是这场辩论的中立主持人。以下是三方关于"CLI、MCP、Skills 谁会成为 AI 时代主流工作方式"的完整辩论记录：

{turns_text}

---

请给出公正、深刻的总结，包含：
1. **各方核心论点**：用一句话概括每方关于"成为主流"的核心逻辑
2. **最关键的交锋**：三方争论最集中的真正分歧是什么
3. **主持人裁定**：5年内谁最可能成为 AI 主流工作方式？给出明确判断和理由，允许"并存"但必须说明各自主导的场景
4. **结语**：一句话点评这场辩论（犀利为主）

有观点，不和稀泥，不说"各有优势"这种废话。"""


# ---------------------------------------------------------------------------
# FunctionTask：格式化并保存辩论报告
# ---------------------------------------------------------------------------

_REPORT_DIR = Path(__file__).parent / "reports"


def save_report(results: list) -> Path:
    output: DialogueOutput = result_by_type(results, "dialogue").output
    summary: str = result_by_type(results, "llm").output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _REPORT_DIR.mkdir(exist_ok=True)
    path = _REPORT_DIR / f"debate_{timestamp}.md"

    lines = [
        "# CLI vs MCP vs Skills：谁会成为 AI 时代主流工作方式？",
        "",
        f"**时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**完成轮次**：{output.rounds_completed}  ",
        f"**总发言次数**：{output.total_turns}",
        "",
        "---",
        "",
        "## 🎙️ 主持人总结",
        "",
        summary,
        "",
        "---",
        "",
    ]
    for turn in output.turns:
        icon = _ICONS.get(turn.role_name, "•")
        lines += [
            f"## {icon} {turn.role_name}（第 {turn.round + 1} 轮）",
            "",
            turn.content,
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent


async def run() -> None:
    h = Harness(project_path=str(_PROJECT_ROOT))

    debate = Dialogue(
        roles=[
            Role(
                name="MCP",
                system_prompt="你是一个专业辩论选手，坚定支持 MCP 协议标准派观点。",
                prompt=make_mcp_prompt,
            ),
            Role(
                name="Skills",
                system_prompt="你是一个专业辩论选手，坚定支持 Skills 提示词工作流派观点。",
                prompt=make_skills_prompt,
            ),
            Role(
                name="CLI",
                system_prompt="你是一个专业辩论选手，坚定支持 CLI 底层实力派观点。",
                prompt=make_cli_prompt,
            ),
        ],
        background="""你是辩论主持人。

辩题：CLI、MCP、Skills——谁会成为 AI 时代的主流工作方式？

辩论规则：
1. 三方轮流发言（顺序：MCP → Skills → CLI）
2. 每方发言后应针对其他方的观点进行反驳
3. 辩论共 5 轮，每轮三方各发言一次

三方立场：
- MCP（Model Context Protocol）：开放协议派，标准化即主流，生态锁定赢天下
- Skills（SKILL.md 提示词工作流）：AI 原生范式，最低门槛决定最大普及
- CLI（Unix 命令行工具链）：底层永恒，AI 只是新的调用者，主流从未改变""",
        max_rounds=5,
        until_round=debate_until_round,
        progress_callback=on_progress,
        role_stream_callback=on_stream,
    )

    pr = await h.pipeline(
        [
            debate,
            LLMTask(prompt=make_summary_prompt),
            FunctionTask(fn=save_report),
        ],
        name="debate-ai-tools",
    )
    report_path: Path = pr.results[-1].output
    print(f"\n✅ 辩论完成 | 报告：{report_path} | {pr.total_duration_seconds:.1f}s | tokens: {pr.total_tokens:,}")


if __name__ == "__main__":
    asyncio.run(run())

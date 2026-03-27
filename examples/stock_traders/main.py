"""多游资盘中讨论下午选股 — Agent 结构化角色 + MCP 工具实时数据。

每个 Agent 通过 ClaudeCliRunner 启动 Claude CLI 子进程，
自主调用 financedata MCP 等工具查询实时行情，无需硬编码数据。

前提：Claude CLI 已配置 financedata MCP server。

运行：
    uv run python examples/stock_traders/main.py
"""

import asyncio
import sys

from harness import (
    Agent,
    Dialogue,
    FunctionTask,
    Harness,
    LLMTask,
    State,
)
from harness.runners.base import AbstractRunner
from harness.runners.claude_cli import ClaudeCliRunner


# ── Runner ────────────────────────────────────────────────────────────────


def make_runner() -> AbstractRunner:
    return ClaudeCliRunner()


# ── 角色定义 ──────────────────────────────────────────────────────────────

TOOL_INSTRUCTION = (
    "你有金融数据工具可用，请主动调用工具查询今日 A 股实时行情数据，"
    "包括但不限于：涨跌停家数、连板高度、炸板率、板块涨幅排名、"
    "个股分时量价等。用真实数据支撑你的分析，不要编造数据。"
)


def create_agents(runner: AbstractRunner) -> list[Agent]:
    """创建四位游资角色。"""
    return [
        Agent(
            name="涨停敢死队",
            description="打板/排板风格的激进短线游资。",
            goal="盘中识别涨停板机会，尾盘排板或半路板介入。",
            backstory=(
                "从营业部散户起家，靠打板战法实现资金翻倍。"
                "擅长集合竞价抢筹、炸板回封判断，对换手率和封单量极度敏感。"
            ),
            constraints=[
                "只做当日能封板的票，不做弱势股",
                "炸板超过 2 次坚决不碰",
                "单票仓位不超过 30%",
                TOOL_INSTRUCTION,
                "回答不超过 200 字",
            ],
            runner=runner,
        ),
        Agent(
            name="龙头猎手",
            description="龙头战法选手，追求辨识度和市场合力。",
            goal="在每轮行情中识别真龙头，区分龙头和跟风。",
            backstory=(
                "研究过近十年所有连板龙头的走势，总结出「辨识度+合力」模型。"
                "认为真龙头具备题材新鲜度、资金一致性、跟风股梯队三个特征。"
            ),
            constraints=[
                "只做板块龙头，不碰跟风补涨",
                "龙头分歧日可低吸，一致日不追高",
                "如果看不清主线，宁可空仓",
                TOOL_INSTRUCTION,
                "回答不超过 200 字",
            ],
            runner=runner,
        ),
        Agent(
            name="情绪周期派",
            description="基于市场情绪温度判断仓位和节奏。",
            goal="判断当前情绪周期阶段，给出仓位建议。",
            backstory=(
                "深研「情绪周期四阶段」模型：冰点→回暖→高潮→退潮。"
                "通过涨停家数、跌停家数、连板高度、炸板率四个指标量化情绪温度。"
                "在冰点期重仓出击，退潮期严格空仓。"
            ),
            constraints=[
                "退潮期建议仓位不超过 20%",
                "冰点转折信号出现前不重仓",
                "每次回答必须先判断当前情绪阶段",
                TOOL_INSTRUCTION,
                "回答不超过 200 字",
            ],
            runner=runner,
        ),
        Agent(
            name="量价玄学家",
            description="纯量价派，只看成交量和分时图。",
            goal="通过量价关系和分时走势判断主力意图。",
            backstory=(
                "不看基本面、不看消息面，只相信量价不会说谎。"
                "擅长分时图判断：早盘急拉放量是出货还是抢筹，"
                "尾盘缩量回落是洗盘还是弱势。"
                "对均量线、量比、内外盘比有独到理解。"
            ),
            constraints=[
                "分析必须引用具体量价数据",
                "不讨论基本面和消息面",
                "缩量一字板视为强势锁仓，放量滞涨视为出货",
                TOOL_INSTRUCTION,
                "回答不超过 200 字",
            ],
            runner=runner,
        ),
    ]


# ── State ─────────────────────────────────────────────────────────────────


class TradingState(State):
    individual_views: str = ""
    discussion_result: str = ""
    afternoon_picks: str = ""


# ── 阶段 1: 独立分析 ─────────────────────────────────────────────────────


async def phase1_independent_analysis(agents: list[Agent]) -> str:
    """四位游资并发独立分析盘面，各自调用工具查询实时数据。"""
    print("\n" + "=" * 60)
    print("阶段 1: 四位游资独立盘面分析（并发，各自查询实时数据）")
    print("=" * 60)

    prompt = (
        "现在请你分析今日 A 股盘面。"
        "请先使用工具查询今日实时行情数据（涨跌停数据、连板高度、"
        "板块涨幅排名、领涨个股等），然后基于你的交易风格给出分析和下午操作判断。"
    )

    tasks = [agent.run(prompt) for agent in agents]
    results = await asyncio.gather(*tasks)

    views: list[str] = []
    for agent, result in zip(agents, results):
        print(f"\n【{agent.name}】")
        print(result)
        views.append(f"[{agent.name}]: {result}")

    return "\n\n".join(views)


# ── 阶段 2-3: Dialogue 讨论 + 汇总 ───────────────────────────────────────


async def phase2_3_discussion_and_summary(
    agents: list[Agent],
    individual_views: str,
    runner: AbstractRunner,
) -> tuple[str, str]:
    """圆桌讨论 + LLMTask 汇总。"""
    print("\n" + "=" * 60)
    print("阶段 2: 四位游资圆桌讨论下午选股")
    print("=" * 60)

    h = Harness(project_path=".", runner=runner)

    state = TradingState()
    state._set_output("individual_views", individual_views)

    pr = await h.pipeline(
        [
            # 阶段 2: 圆桌讨论
            Dialogue(
                background=(
                    "四位游资风格各异，现在围坐讨论下午的操作计划。\n"
                    "各自已完成独立盘面分析（含实时数据查询），观点如下：\n"
                    f"{individual_views}\n\n"
                    "讨论中如需验证数据，可随时调用工具查询。"
                ),
                roles=[
                    agents[0].as_role(
                        lambda ctx: (
                            "基于你查到的实时数据和其他人的观点，阐述下午打板机会。"
                            "如需验证其他人提到的个股，请调用工具查询。"
                            + (f"\n回应上一轮讨论：{ctx.last_from('龙头猎手') or ''}" if ctx.round > 0 else "")
                        )
                    ),
                    agents[1].as_role(
                        lambda ctx: (
                            f"回应涨停敢死队的观点：{ctx.last_from('涨停敢死队') or ''}，"
                            "从龙头辨识角度给出选股方向。可调用工具查具体个股数据验证。"
                        )
                    ),
                    agents[2].as_role(
                        lambda ctx: (
                            "综合前面讨论，判断当前情绪周期阶段，给出仓位建议。"
                            f"回应：{ctx.last_from('龙头猎手') or ''}"
                        )
                    ),
                    agents[3].as_role(
                        lambda ctx: (
                            "从量价角度验证前面提到的票，请调用工具查询具体量价数据。"
                            f"回应情绪周期派的仓位建议：{ctx.last_from('情绪周期派') or ''}"
                        )
                    ),
                ],
                max_rounds=2,
                progress_callback=lambda e: print(
                    f"  [{e.event}] {e.role_name}"
                    + (
                        f": {e.content[:80]}..."
                        if e.content and len(e.content) > 80
                        else f": {e.content}" if e.content else ""
                    )
                ),
            ),
            # 提取讨论结果到 state
            FunctionTask(
                fn=lambda state: str(state._results[-1].output),
                output_key="discussion_result",
            ),
            # 阶段 3: 汇总选股
            LLMTask(
                prompt=lambda state: (
                    f"以下是四位游资的讨论纪要（基于实时行情数据）：\n{state.discussion_result}\n\n"
                    "请汇总讨论结果，输出：\n"
                    "1. 各位的核心观点（每人一句话）\n"
                    "2. 达成共识的方向\n"
                    "3. 下午具体选股建议（2-3 只，含代码、买入逻辑和仓位）\n"
                    "4. 风险提示"
                ),
                system_prompt="你是一位资深投资经理，负责汇总团队讨论并输出可执行的交易计划。",
                output_key="afternoon_picks",
            ),
        ],
        state=state,
    )

    print("\n" + "=" * 60)
    print("阶段 3: 汇总选股建议")
    print("=" * 60)
    discussion = pr.results[1].output
    picks = pr.results[2].output
    print(picks)

    return discussion, picks


# ── 主函数 ────────────────────────────────────────────────────────────────


async def main() -> None:
    runner = make_runner()
    print(f"使用 runner: {type(runner).__name__}")

    agents = create_agents(runner)
    print(f"\n创建了 {len(agents)} 位游资角色：")
    for a in agents:
        print(f"  - {a.name}: {a.description}")

    # 阶段 1: 独立分析（各自调用 MCP 工具查询实时数据）
    individual_views = await phase1_independent_analysis(agents)

    # 阶段 2-3: 讨论 + 汇总
    discussion, picks = await phase2_3_discussion_and_summary(
        agents, individual_views, runner,
    )

    print("\n" + "=" * 60)
    print("完成！四位游资盘中讨论选股流程结束。")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

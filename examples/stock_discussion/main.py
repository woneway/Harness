"""Discussion 结构化讨论验证示例 — 多 Agent 选股讨论。

验证核心能力：
1. Agent 结构化立场（position_schema）
2. 立场演变追踪（position_history）
3. 收敛检测（convergence）
4. 框架自动构建提示词（含其他 Agent 立场）

运行：
    uv run python examples/stock_discussion/main.py
"""

import asyncio

from pydantic import BaseModel

from harness import (
    Agent,
    Discussion,
    DiscussionOutput,
    FunctionTask,
    Harness,
    LLMTask,
    State,
)
from harness.tasks.discussion import all_agree_on, positions_stable
from harness.runners.claude_cli import ClaudeCliRunner, PermissionMode


# ── Position Schema ────────────────────────────────────────────────────────


class TradingPosition(BaseModel):
    top_pick: str           # 首选股票
    direction: str          # "buy" / "sell" / "hold"
    confidence: float       # 0-1
    key_reason: str         # 一句话理由


# ── Progress Handler ──────────────────────────────────────────────────────


def _progress_handler(e) -> None:
    """Discussion 进度回调：显示 start/complete/error + 流式文本。"""
    if e.event == "streaming":
        # 流式文本：不换行，实时显示 Claude 输出
        print(e.content or "", end="", flush=True)
    elif e.event == "start":
        print(f"\n  [{e.event}] {e.agent_name}", flush=True)
    elif e.event == "complete":
        # complete 时 content 是完整回复，这里只显示状态（流式已输出过文本）
        print(f"\n  [{e.event}] {e.agent_name}", flush=True)
    elif e.event == "error":
        content_preview = f": {e.content[:80]}" if e.content else ""
        print(f"\n  [{e.event}] {e.agent_name}{content_preview}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────


async def main() -> None:
    runner = ClaudeCliRunner(permission_mode=PermissionMode.BYPASS)
    print("使用 ClaudeCliRunner (bypassPermissions)")

    # 定义 Agent
    analyst = Agent(
        name="技术分析师",
        description="看K线、MACD、均线系统的技术派。",
        goal="判断短期走势方向和关键点位",
        constraints=["只基于技术指标判断", "不考虑消息面"],
        runner=runner,
    )
    trader = Agent(
        name="短线交易员",
        description="盘中选股，关注资金流向和龙虎榜。",
        goal="选出最可能涨的票",
        constraints=["只做短线", "关注量价关系"],
        runner=runner,
    )
    risk_mgr = Agent(
        name="风控经理",
        description="评估风险，控制仓位。",
        goal="确保每笔交易风险可控",
        constraints=["回撤不超过3%", "单票仓位不超过30%"],
        runner=runner,
    )

    # State
    class TradingState(State):
        market: str = ""
        discussion: DiscussionOutput | None = None
        plan: str = ""

    # Pipeline
    h = Harness(project_path=".", runner=runner)

    print("\n" + "=" * 60)
    print("Discussion 结构化讨论：三人选股")
    print("=" * 60)

    pr = await h.pipeline([
        # Step 0: 模拟行情数据
        FunctionTask(
            fn=lambda state: (
                "今日 A 股概况：\n"
                "- 上证 +0.8%，深证 +1.2%，创业板 +1.5%\n"
                "- 涨停 45 家，跌停 8 家\n"
                "- 北向资金净买入 60 亿\n"
                "- 热门板块：AI算力（+3.2%）、光伏（+2.1%）、券商（+1.8%）\n"
                "- 龙虎榜：中际旭创（机构买入 5 亿）、通威股份（游资对倒）\n"
                "- 资金流向：AI板块主力净流入 80 亿，光伏 30 亿"
            ),
            output_key="market",
        ),

        # Step 1: 三人结构化讨论
        Discussion(
            agents=[analyst, trader, risk_mgr],
            position_schema=TradingPosition,
            topic="下午盘选股：选一只最值得买入的票",
            background=lambda state: f"今日行情数据：\n{state.market}",
            max_rounds=3,
            convergence=all_agree_on("top_pick"),
            progress_callback=_progress_handler,
            output_key="discussion",
        ),

        # Step 2: 基于讨论结果制定操作计划
        LLMTask(
            prompt=lambda state: (
                f"基于以下三人讨论结果，制定具体操作计划（买入价、止损价、目标价、仓位）。\n\n"
                f"是否达成共识：{state.discussion.converged}\n"
                f"讨论轮数：{state.discussion.rounds_completed}\n\n"
                f"最终立场：\n"
                + "\n".join(
                    f"- {name}: {pos.top_pick} ({pos.direction}, "
                    f"信心 {pos.confidence:.0%}) — {pos.key_reason}"
                    for name, pos in state.discussion.final_positions.items()
                )
            ),
            output_key="plan",
        ),
    ], state=TradingState())

    # 输出结果
    output = pr.results[1].output
    assert isinstance(output, DiscussionOutput)

    print("\n" + "=" * 60)
    print("讨论结果")
    print("=" * 60)
    print(f"轮数: {output.rounds_completed}，发言: {output.total_turns} 次")
    print(f"收敛: {output.converged}", end="")
    if output.convergence_round is not None:
        print(f"（第 {output.convergence_round} 轮）")
    else:
        print()

    print("\n最终立场：")
    for name, pos in output.final_positions.items():
        print(f"  {name}: {pos.top_pick} ({pos.direction}, "
              f"信心 {pos.confidence:.0%}) — {pos.key_reason}")

    print("\n立场演变：")
    for name, positions in output.position_history.items():
        picks = [f"{p.top_pick}({p.direction})" for p in positions]
        print(f"  {name}: {' → '.join(picks)}")

    print("\n" + "=" * 60)
    print("操作计划")
    print("=" * 60)
    print(pr.results[2].output)


if __name__ == "__main__":
    asyncio.run(main())

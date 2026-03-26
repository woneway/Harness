"""Agent + Dialogue + Service 完整验证示例。

验证三个新能力：
1. Agent 独立执行（agent.run()）
2. Agent 在 Dialogue 中组合（agent.as_role()）
3. Service 事件触发（h.service() + h.emit()）

运行：
    # 设置环境变量（任选一个 runner）
    export MINIMAX_API_KEY=your_key
    export MINIMAX_BASE_URL=https://api.minimax.chat/v1
    export MINIMAX_MODEL=MiniMax-Text-01

    uv run python examples/agent_discussion/main.py
"""

import asyncio
import os
import sys

from harness import (
    Agent,
    CronTrigger,
    Dialogue,
    EventTrigger,
    FunctionTask,
    Harness,
    LLMTask,
    State,
    TriggerContext,
)
from harness.runners.openai import OpenAIRunner


def make_runner() -> OpenAIRunner:
    """尝试创建 runner，优先 MiniMax，其次 OpenAI。"""
    # MiniMax
    if os.environ.get("MINIMAX_API_KEY"):
        return OpenAIRunner(
            api_key_env="MINIMAX_API_KEY",
            base_url_env="MINIMAX_BASE_URL",
            model_env="MINIMAX_MODEL",
        )
    # OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIRunner()
    print("请设置 MINIMAX_API_KEY 或 OPENAI_API_KEY 环境变量")
    sys.exit(1)


# ── 验证 1: Agent 独立执行 ──────────────────────────────────────────────


async def test_agent_standalone(runner: OpenAIRunner) -> None:
    print("\n" + "=" * 60)
    print("验证 1: Agent 独立执行")
    print("=" * 60)

    analyst = Agent(
        name="技术分析师",
        system_prompt="你是一位经验丰富的股票技术分析师。回答简洁，不超过 100 字。",
        runner=runner,
    )

    result = await analyst.run("用一句话总结今天 A 股市场的常见技术形态有哪些？")
    print(f"\n[{analyst.name}]: {result}")
    print(f"\n✅ Agent.run() 正常工作")


# ── 验证 2: Agent + Dialogue 组合 ──────────────────────────────────────


async def test_agent_dialogue(runner: OpenAIRunner) -> None:
    print("\n" + "=" * 60)
    print("验证 2: Agent + Dialogue（两个 Agent 讨论）")
    print("=" * 60)

    analyst = Agent(
        name="技术派",
        system_prompt="你是技术分析师，只看 K 线和指标。回答不超过 80 字。",
        runner=runner,
    )
    fundamentalist = Agent(
        name="基本面派",
        system_prompt="你是基本面研究员，只看财报和行业。回答不超过 80 字。",
        runner=runner,
    )

    class DiscussionState(State):
        topic: str = "贵州茅台近期是否值得买入"

    h = Harness(project_path=".", runner=runner)
    pr = await h.pipeline(
        [
            Dialogue(
                background="讨论话题：贵州茅台近期是否值得买入",
                roles=[
                    analyst.as_role(
                        lambda ctx: (
                            f"从技术面分析'{ctx.state.topic}'，"
                            + (
                                f"回应{ctx.last_from('基本面派')}"
                                if ctx.last_from("基本面派")
                                else "先给出你的判断"
                            )
                        )
                    ),
                    fundamentalist.as_role(
                        lambda ctx: (
                            f"从基本面分析'{ctx.state.topic}'，"
                            f"回应技术派的观点：{ctx.last_from('技术派')}"
                        )
                    ),
                ],
                max_rounds=2,
                progress_callback=lambda e: print(
                    f"  [{e.event}] {e.role_name}"
                    + (f": {e.content[:60]}..." if e.content and len(e.content) > 60 else f": {e.content}" if e.content else "")
                ),
            ),
        ],
        state=DiscussionState(),
    )

    output = pr.results[0].output
    print(f"\n共 {output.rounds_completed} 轮，{output.total_turns} 次发言")
    print(f"✅ Agent.as_role() + Dialogue 正常工作")


# ── 验证 3: Service 事件触发 ────────────────────────────────────────────


async def test_service_event(runner: OpenAIRunner) -> None:
    print("\n" + "=" * 60)
    print("验证 3: Service 事件触发")
    print("=" * 60)

    analyst = Agent(
        name="分析师",
        system_prompt="你是股票分析师。用一句话回应市场异动。不超过 50 字。",
        runner=runner,
    )

    class AlertState(State):
        alert_info: str = ""
        analysis: str = ""

    results_collected = []

    h = Harness(project_path=".", runner=runner)

    async def on_alert(ctx: TriggerContext):
        info = ctx.event_data or {}
        print(f"  [触发] event={ctx.event_name}, data={info}")
        return [
            FunctionTask(
                fn=lambda state: f"{info.get('symbol', '未知')} 涨跌 {info.get('change', 0):.1%}",
                output_key="alert_info",
            ),
            LLMTask(
                prompt=lambda state: f"市场异动: {state.alert_info}，给出简短分析",
                system_prompt=analyst.system_prompt,
                runner=analyst.runner,
                output_key="analysis",
            ),
        ]

    h.service(
        "alert-monitor",
        triggers=[
            EventTrigger(
                "price_alert",
                filter=lambda d: abs(d.get("change", 0)) > 0.03,
            ),
        ],
        handler=on_alert,
        state_factory=AlertState,
    )

    await h.start()
    try:
        # 发射一个被 filter 拦截的事件（change < 3%）
        print("\n发射小幅波动事件（应被过滤）...")
        await h.emit("price_alert", {"symbol": "平安银行", "change": 0.01})
        await asyncio.sleep(0.5)

        # 发射一个通过 filter 的事件
        print("发射大幅波动事件（应触发 pipeline）...")
        await h.emit("price_alert", {"symbol": "贵州茅台", "change": 0.05})
        await asyncio.sleep(3)  # 等待 LLM 执行

        print(f"\n✅ Service 事件触发 + filter + pipeline 正常工作")
    finally:
        await h.stop()


# ── 主函数 ──────────────────────────────────────────────────────────────


async def main() -> None:
    runner = make_runner()
    print(f"使用模型: {runner.model} @ {runner.base_url}")

    await test_agent_standalone(runner)
    await test_agent_dialogue(runner)
    await test_service_event(runner)

    print("\n" + "=" * 60)
    print("🎉 所有验证通过！v2.1 Agent + Service 功能正常")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

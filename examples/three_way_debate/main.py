"""three_way_debate/main.py — 三方辩论：Harness 是否应支持 streaming Dialogue？

三个角色：
  advocate  — 推动方，主张加 streaming
  skeptic   — 质疑方，反对过早引入复杂度
  judge     — 裁判，综合双方，在认为可以裁决时给出最终判断

until 条件：judge 发言包含"我的决定是"，提前终止。

运行：
    uv run python examples/three_way_debate/main.py

    # 不调用 Claude CLI，用 mock runner 看结构：
    uv run python examples/three_way_debate/main.py --mock
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harness import Dialogue, Harness, Role
from harness._internal.dialogue import DialogueContext

# ---------------------------------------------------------------------------
# 辩题背景
# ---------------------------------------------------------------------------

BACKGROUND = """
我们正在讨论是否为 Harness 的 Dialogue 功能添加 streaming 支持。
目前 Dialogue 等待每个角色完整输出后才进入下一轮；streaming 方案会逐 token 推送。

请保持发言简洁（3-5 句），直接回应对方的最新论点。
""".strip()


# ---------------------------------------------------------------------------
# Prompt 构造函数
# ---------------------------------------------------------------------------

def advocate_prompt(ctx: DialogueContext) -> str:
    if ctx.round == 0:
        return "请阐述为什么 Harness Dialogue 应该支持 streaming 输出，重点讲用户体验收益。"

    skeptic_last = ctx.last_from("skeptic") or ""
    judge_last = ctx.last_from("judge") or ""
    return (
        f"怀疑方的最新反对意见是：\n{skeptic_last}\n\n"
        f"裁判的最新看法是：\n{judge_last}\n\n"
        "请针对性地回应怀疑方，并补充你方最有力的论点。"
    )


def skeptic_prompt(ctx: DialogueContext) -> str:
    advocate_last = ctx.last_from("advocate") or ""
    judge_last = ctx.last_from("judge") or ""

    if ctx.round == 0:
        return (
            f"倡导方刚才说：\n{advocate_last}\n\n"
            "请提出你认为最关键的反对理由，聚焦在实现复杂度和维护成本上。"
        )

    return (
        f"倡导方的最新回应是：\n{advocate_last}\n\n"
        f"裁判的最新看法是：\n{judge_last}\n\n"
        "请指出倡导方论据中最薄弱的环节，并坚守你的立场。"
    )


def judge_prompt(ctx: DialogueContext) -> str:
    advocate_last = ctx.last_from("advocate") or ""
    skeptic_last = ctx.last_from("skeptic") or ""
    all_advocate = ctx.all_from("advocate")
    all_skeptic = ctx.all_from("skeptic")

    rounds_so_far = ctx.round + 1
    history_summary = (
        f"已进行 {rounds_so_far} 轮。"
        f"倡导方共发言 {len(all_advocate)} 次，怀疑方共发言 {len(all_skeptic)} 次。"
    )

    if ctx.round < 1:
        return (
            f"{history_summary}\n\n"
            f"倡导方最新论点：\n{advocate_last}\n\n"
            f"怀疑方最新论点：\n{skeptic_last}\n\n"
            "作为裁判，请综合评估双方论点，指出各自的合理之处，暂不作最终判断。"
        )

    return (
        f"{history_summary}\n\n"
        f"倡导方最新论点：\n{advocate_last}\n\n"
        f"怀疑方最新论点：\n{skeptic_last}\n\n"
        "辩论已经充分。请综合所有轮次的论点，给出你的最终裁决。"
        "如果你已经可以作出判断，请在回答中明确包含「我的决定是」。"
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run(mock: bool = False) -> None:
    project_path = str(Path(__file__).parent.parent.parent)

    if mock:
        from harness.runners.base import AbstractRunner, RunnerResult

        # 调用顺序：每轮 advocate(0) → skeptic(1) → judge(2)
        _call_seq = [0]

        responses_by_role = {
            "advocate": [
                "streaming 能大幅提升用户感知响应速度，对长对话尤其重要。",
                "实现成本可以通过分层抽象控制，runner 层已经支持 stream_callback。",
                "我坚持认为用户体验收益远大于实现成本。",
            ],
            "skeptic": [
                "streaming 会让 Dialogue 的 session 管理复杂度翻倍，不值得。",
                "advocate 的分层方案听起来好，但实际上会造成 API 不一致。",
                "在没有真实用户需求数据前，过早优化是浪费。",
            ],
            "judge": [
                "双方各有道理：体验收益真实，但复杂度风险也不容忽视。",
                "我的决定是：暂不支持 streaming Dialogue，待 v2 收集真实需求后再评估。",
            ],
        }
        role_order = ["advocate", "skeptic", "judge"]

        class MockRunner(AbstractRunner):
            async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
                idx = _call_seq[0] % 3
                role = role_order[idx]
                round_num = _call_seq[0] // 3
                _call_seq[0] += 1
                pool = responses_by_role[role]
                text = pool[min(round_num, len(pool) - 1)]
                return RunnerResult(text=text, tokens_used=10, session_id=f"mock-{role}")

        harness = Harness(project_path=project_path, runner=MockRunner())

        # 绕过 SQLAlchemy（mock 模式不需要持久化）
        from unittest.mock import AsyncMock, MagicMock
        storage = MagicMock()
        storage.save_run = AsyncMock()
        storage.save_task_log = AsyncMock()
        storage.update_run = AsyncMock()
        storage.get_task_logs = AsyncMock(return_value=[])
        harness._storage = storage
        harness._initialized = True
    else:
        harness = Harness(project_path=project_path)

    dialogue = Dialogue(
        background=BACKGROUND,
        max_rounds=3,
        until=lambda ctx: "我的决定是" in (ctx.last_from("judge") or ""),
        roles=[
            Role(
                name="advocate",
                system_prompt="你是一位热情的技术倡导者，主张为 Harness Dialogue 添加 streaming 支持。发言简洁有力，每次聚焦一个核心论点。",
                prompt=advocate_prompt,
            ),
            Role(
                name="skeptic",
                system_prompt="你是一位务实的工程师，对过早引入复杂功能持谨慎态度。发言简洁，用具体的工程问题反驳对方。",
                prompt=skeptic_prompt,
            ),
            Role(
                name="judge",
                system_prompt="你是公正的技术架构裁判，综合权衡双方论点。在认为辩论已经充分时，给出明确的最终判断，并在判断中包含「我的决定是」。",
                prompt=judge_prompt,
            ),
        ],
    )

    print("=" * 60)
    print("三方辩论：Harness 是否应支持 streaming Dialogue？")
    print("=" * 60)

    pr = await harness.pipeline([dialogue])

    result = pr.results[0]
    output = result.output

    print(f"\n共进行 {output.rounds_completed} 轮，{len(output.turns)} 次发言\n")
    print("-" * 60)

    for turn in output.turns:
        role_labels = {"advocate": "🟢 倡导方", "skeptic": "🔴 质疑方", "judge": "⚖️  裁判"}
        label = role_labels.get(turn.role_name, turn.role_name)
        print(f"\n[Round {turn.round + 1}] {label}")
        print(turn.content)

    print("\n" + "=" * 60)
    print(f"最终裁决来自：{output.final_speaker}")
    print(output.final_content)
    print("=" * 60)


if __name__ == "__main__":
    mock_mode = "--mock" in sys.argv
    asyncio.run(run(mock=mock_mode))

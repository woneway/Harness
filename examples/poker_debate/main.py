"""poker_debate/main.py — 五方辩论：翻前 AA 遇对手 All In，该 Call 吗？

五个角色：
  gto        — GTO机器，纯数学，equity为王
  veteran    — 江湖老炮，经验主义，相信读人
  mtt_pro    — 锦标赛鬼才，ICM专家，情境分析
  fish       — 超级鱼，业余直觉，AA必Call
  exploiter  — 心理战术师，专注对手range和tells

until 条件：老炮（veteran）说出"老夫的判决是"时终止辩论。

运行：
    uv run python examples/poker_debate/main.py
    uv run python examples/poker_debate/main.py --mock
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
德州扑克场景：翻前（Pre-flop），你手握 AA（一对老A），对手突然 All In。
筹码深度、位置、锦标赛/现金桌类型均未指定——各角色可根据自己的理论提出假设。

请用扑克术语自由辩论，每次发言3-6句，直接回应其他角色的最新观点。
""".strip()

# ---------------------------------------------------------------------------
# Prompt 构造函数
# ---------------------------------------------------------------------------

def gto_prompt(ctx: DialogueContext) -> str:
    if ctx.round == 0:
        return (
            "请从GTO和期望值角度，论证翻前AA遇到all in为何必须call。"
            "引用具体的equity数据和数学推导。"
        )
    others = _latest_others(ctx, "gto")
    return (
        f"其他玩家的最新观点如下：\n{others}\n\n"
        "请用数学反驳最不理性的那个论点，并补充GTO框架下的关键数据。"
    )


def veteran_prompt(ctx: DialogueContext) -> str:
    if ctx.round == 0:
        return (
            "请以30年牌龄的江湖老炮身份，谈谈你见过的'翻前fold AA'的真实案例，"
            "以及读人直觉在这个决策中的重要性。"
        )
    others = _latest_others(ctx, "veteran")
    if ctx.round >= 2:
        return (
            f"其他人说：\n{others}\n\n"
            "辩论已经够充分了。作为见过最多的老炮，请给出你的最终经验总结。"
            "如果你已经听够了，请明确说出「老夫的判决是」，然后给出你的裁决。"
        )
    return (
        f"其他玩家的最新观点：\n{others}\n\n"
        "从你的江湖经验出发，指出最脱离实战的那个观点，并讲一个你亲历的相关手牌。"
    )


def mtt_prompt(ctx: DialogueContext) -> str:
    if ctx.round == 0:
        return (
            "请从锦标赛ICM角度论证：为什么同样是翻前AA遇all in，"
            "在不同的锦标赛阶段（泡沫期、FT bubble、HU）可能有不同的最优决策？"
        )
    others = _latest_others(ctx, "mtt_pro")
    return (
        f"其他角色的最新观点：\n{others}\n\n"
        "请指出哪些论点忽略了ICM的影响，并给出具体的筹码深度场景说明差异。"
    )


def fish_prompt(ctx: DialogueContext) -> str:
    if ctx.round == 0:
        return (
            "你是一个业余扑克玩家，对理论不感兴趣。"
            "请用最直觉的方式表达你的观点：拿到AA遇到all in该怎么办？"
        )
    others = _latest_others(ctx, "fish")
    return (
        f"那些专家刚才说：\n{others}\n\n"
        "你听不太懂那些复杂理论，请用大白话说说你的感受，"
        "顺便问一个让专家们哑口无言的天真问题。"
    )


def exploiter_prompt(ctx: DialogueContext) -> str:
    if ctx.round == 0:
        return (
            "请从exploit（针对性剥削）角度论证：翻前AA遇all in的决策，"
            "核心不在于AA本身有多强，而在于对手的pushing range有多宽。"
            "举例说明什么情况下call是错的，什么情况下fold才是exploit最大化。"
        )
    others = _latest_others(ctx, "exploiter")
    return (
        f"其他人的最新论点：\n{others}\n\n"
        "请指出哪个论点犯了'忽略对手range'的错误，并补充具体的range分析。"
        "特别回应一下鱼（fish）的观点。"
    )


# ---------------------------------------------------------------------------
# 辅助：获取除自己外所有角色的最新发言
# ---------------------------------------------------------------------------

ROLE_LABELS = {
    "gto":       "GTO机器",
    "veteran":   "江湖老炮",
    "mtt_pro":   "锦标赛鬼才",
    "fish":      "超级鱼",
    "exploiter": "心理战术师",
}


def _latest_others(ctx: DialogueContext, self_name: str) -> str:
    lines = []
    seen: set[str] = set()
    for turn in reversed(ctx.history):
        if turn.role_name == self_name:
            continue
        if turn.role_name in seen:
            continue
        seen.add(turn.role_name)
        label = ROLE_LABELS.get(turn.role_name, turn.role_name)
        lines.append(f"【{label}】{turn.content}")
        if len(seen) == len(ROLE_LABELS) - 1:
            break
    return "\n\n".join(reversed(lines)) if lines else "（暂无其他发言）"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run(mock: bool = False) -> None:
    project_path = str(Path(__file__).parent.parent.parent)

    if mock:
        from harness.runners.base import AbstractRunner, RunnerResult
        from unittest.mock import AsyncMock, MagicMock

        role_order = ["gto", "mtt_pro", "fish", "exploiter", "veteran"]
        _seq = [0]

        mock_responses: dict[str, list[str]] = {
            "gto": [
                "AA对抗任意两张牌的equity超过80%，对抗KK也有81%。翻前call是纯数学结论，没有讨论空间。",
                "ICM和reads都是二阶考量，EV为负才需要fold，AA的EV永远为正。鱼说得对，只是原因错了。",
                "老炮的'读人'在大样本下统计上无效，现代求解器早已证明这一点。",
            ],
            "veteran": [
                "我见过一个老头，专打$5/$10，二十年只用AA/KK推，你GTO算法能读出这个吗？fold了赚100BB。",
                "GTO是针对未知对手的策略，实战里对手不是随机的。我的直觉是积累了几百万手牌的压缩。",
                "老夫的判决是：100BB以下默认call，但认识对手的情况下永远先问自己'他用什么推'。AA不是免死金牌。",
            ],
            "mtt_pro": [
                "泡沫期10BB短筹，AA当然call。但200BB深筹FT bubble，ICM折扣可能让EV为正变成锦标赛-EV。",
                "GTO的chip-EV框架在锦标赛里是错的。ICMizer跑过数据：某些bubble场景AA对KK可以fold。",
                "鱼问的问题其实很好——'最好的牌'只在现金桌成立，锦标赛的筹码价值是非线性的。",
            ],
            "fish": [
                "AA是最好的牌！不call还等什么？！折了AA我会睡不着觉的！",
                "你们说的ICM是什么？EV又是什么？我只知道AA赢的概率最高，为什么要fold最好的牌？！",
                "那如果对手每次都用AA推呢？那不就是两个人都应该call了吗……等等这说不通……",
            ],
            "exploiter": [
                "关键不是你的牌，是对手的range。对手用100%range推，call。对手只用AA推，EV接近于0，随机策略。",
                "GTO说的是针对未知对手的下限策略，但我们在实战里从不面对未知对手。reads乘以range才是exploit。",
                "鱼问了个好问题：如果双方都是AA，pot对半分，EV=0。这恰恰说明range比手牌本身更重要。",
            ],
        }

        class MockRunner(AbstractRunner):
            async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
                idx = _seq[0] % len(role_order)
                role = role_order[idx]
                round_num = _seq[0] // len(role_order)
                _seq[0] += 1
                pool = mock_responses[role]
                text = pool[min(round_num, len(pool) - 1)]
                return RunnerResult(text=text, tokens_used=10, session_id=f"mock-{role}")

        harness = Harness(project_path=project_path, runner=MockRunner())
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
        until=lambda ctx: "老夫的判决是" in (ctx.last_from("veteran") or ""),
        roles=[
            Role(
                name="gto",
                system_prompt=(
                    "你是一台GTO扑克求解器的化身，只相信数学和期望值。"
                    "你对'读人'和'直觉'嗤之以鼻，认为所有非EV论点都是情绪化的。"
                    "发言简洁犀利，喜欢引用具体数字，偶尔流露出对其他玩家的优越感。"
                ),
                prompt=gto_prompt,
            ),
            Role(
                name="mtt_pro",
                system_prompt=(
                    "你是一名锦标赛职业选手，深谙ICM（独立筹码模型）。"
                    "你认为现金桌和锦标赛是两个完全不同的游戏，"
                    "大多数讨论翻前策略的人都忽略了筹码深度和ICM压力。"
                    "发言学术而严谨，但也能听出其他人论点的漏洞。"
                ),
                prompt=mtt_prompt,
            ),
            Role(
                name="fish",
                system_prompt=(
                    "你是一个刚学会德州扑克的业余玩家，对理论一无所知。"
                    "你的逻辑很简单：好牌就该all in，坏牌就fold。"
                    "你经常说出让专家无法反驳的天真问题，虽然是无意识的。"
                    "说话带着真实的困惑和热情，不要假装懂你不懂的东西。"
                ),
                prompt=fish_prompt,
            ),
            Role(
                name="exploiter",
                system_prompt=(
                    "你是一名专注exploit（针对性剥削）的玩家，"
                    "你认为GTO是懒人策略，真正的高手应该根据对手的具体倾向调整策略。"
                    "你特别关注对手的pushing range，认为'AA有多强'是个假命题，"
                    "正确的问题是'对手用什么range推'。发言逻辑清晰，擅长举极端案例。"
                ),
                prompt=exploiter_prompt,
            ),
            Role(
                name="veteran",
                system_prompt=(
                    "你是一个打了30年德州扑克的江湖老炮，见过各种神仙打架。"
                    "你相信读人比任何理论都重要，经常用亲历的手牌故事说话。"
                    "说话有点大牌，但确实有两把刷子。辩论最后你会给出「老夫的判决是」。"
                ),
                prompt=veteran_prompt,
            ),
        ],
    )

    print("=" * 60)
    print("五方辩论：翻前 AA 遇对手 All In，该 Call 吗？")
    print("=" * 60)

    pr = await harness.pipeline([dialogue])

    output = pr.results[0].output

    print(f"\n共 {output.rounds_completed} 轮，{len(output.turns)} 次发言\n")

    role_icons = {
        "gto":       "🤖 GTO机器",
        "veteran":   "🧓 江湖老炮",
        "mtt_pro":   "🏆 锦标赛鬼才",
        "fish":      "🐟 超级鱼",
        "exploiter": "🎭 心理战术师",
    }

    current_round = -1
    for turn in output.turns:
        if turn.round != current_round:
            current_round = turn.round
            print(f"\n{'─' * 60}")
            print(f"  第 {turn.round + 1} 轮")
            print(f"{'─' * 60}")
        label = role_icons.get(turn.role_name, turn.role_name)
        print(f"\n{label}")
        print(turn.content)

    print("\n" + "=" * 60)
    final_label = role_icons.get(output.final_speaker, output.final_speaker)
    print(f"最终判决 — {final_label}")
    print("=" * 60)
    print(output.final_content)


if __name__ == "__main__":
    mock_mode = "--mock" in sys.argv
    asyncio.run(run(mock=mock_mode))

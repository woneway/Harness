"""poker_debate/main.py — 五方辩论：翻前 AA 遇对手 All In，该 Call 吗？

五个角色：
  gto        — GTO机器，纯数学，equity为王
  veteran    — 江湖老炮，经验主义，相信读人
  mtt_pro    — 锦标赛鬼才，ICM专家，情境分析
  fish       — 超级鱼，业余直觉，AA必Call
  exploiter  — 心理战术师，专注对手range和tells

模式：回合模式（next_speaker 动态决定谁发言）
  - 开场（第0-4回合）：5人依次亮出各自立场
  - 辩论（第5回合起）：按张力弧度循环点名，每人直接回应上一发言者
  - 终止：老炮说出「老夫的判决是」时结束

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
from harness.task import DialogueTurn

# ---------------------------------------------------------------------------
# 辩题背景
# ---------------------------------------------------------------------------

BACKGROUND = """
德州扑克场景：翻前（Pre-flop），你手握 AA（一对老A），对手突然 All In。
筹码深度、位置、锦标赛/现金桌类型均未指定——各角色可根据自己的理论提出假设。

请直接回应其他角色的最新观点，每次发言4-6句，有话直说，不要客气。
""".strip()

ROLE_LABELS = {
    "gto":       "GTO机器",
    "veteran":   "江湖老炮",
    "mtt_pro":   "锦标赛鬼才",
    "fish":      "超级鱼",
    "exploiter": "心理战术师",
}

# 开场顺序
OPENING_ORDER = ["gto", "mtt_pro", "fish", "exploiter", "veteran"]

# 辩论弧度：制造天然张力的循环顺序
# veteran↔gto（直觉 vs 数学），exploiter↔mtt_pro（range vs ICM），fish 穿插制造混乱
DEBATE_CYCLE = ["veteran", "gto", "exploiter", "mtt_pro", "fish"]


def next_speaker(history: list[DialogueTurn]) -> str:
    n = len(history)
    if n < len(OPENING_ORDER):
        return OPENING_ORDER[n]
    return DEBATE_CYCLE[(n - len(OPENING_ORDER)) % len(DEBATE_CYCLE)]


# ---------------------------------------------------------------------------
# 辅助：构建发言上下文文本
# ---------------------------------------------------------------------------

def _last_speaker_context(ctx: DialogueContext, self_name: str) -> str:
    """上一位发言者的内容（如果不是自己）。"""
    if not ctx.history:
        return ""
    last = ctx.history[-1]
    if last.role_name == self_name:
        return ""
    label = ROLE_LABELS.get(last.role_name, last.role_name)
    return f"【{label}】刚才说：\n{last.content}"


def _recent_context(ctx: DialogueContext, self_name: str, n: int = 3) -> str:
    """最近 n 条他人发言（排除自己）。"""
    lines = []
    for turn in reversed(ctx.history):
        if turn.role_name == self_name:
            continue
        label = ROLE_LABELS.get(turn.role_name, turn.role_name)
        lines.append(f"【{label}】{turn.content}")
        if len(lines) >= n:
            break
    return "\n\n".join(reversed(lines)) if lines else "（暂无其他发言）"


# ---------------------------------------------------------------------------
# 各角色 Prompt 函数
# ---------------------------------------------------------------------------

def gto_prompt(ctx: DialogueContext) -> str:
    # 开场
    if not ctx.all_from("gto"):
        return (
            "请从GTO和期望值角度，论证翻前AA遇到all in为何必须call。"
            "引用具体的equity数据：AA vs 随机两张牌、AA vs KK、AA vs AKs。"
            "最后点名挑战一下你认为最不理性的论点会来自哪个方向。"
        )
    last = _last_speaker_context(ctx, "gto")
    if last:
        return (
            f"{last}\n\n"
            "请直接反驳这个观点。用具体的EV数字说明对方哪里错了，"
            "如果对方说的有任何合理成分，也可以承认——但要用数学框架重新表述。"
        )
    recent = _recent_context(ctx, "gto")
    return (
        f"近期发言：\n{recent}\n\n"
        "挑出最违背EV原则的那个论点，用求解器数据彻底驳倒它。"
    )


def veteran_prompt(ctx: DialogueContext) -> str:
    # 开场
    if not ctx.all_from("veteran"):
        return (
            "请以30年牌龄的江湖老炮身份亮出你的立场。"
            "讲一个你亲历的'翻前fold AA'并且正确的真实手牌故事，"
            "然后说明读人直觉在这个决策中的核心地位。"
            "最后点一下你最不服气的那种论点。"
        )
    last = _last_speaker_context(ctx, "veteran")
    total_turns = len(ctx.history)
    # 15回合后开始催促收尾
    if total_turns >= 15:
        recent = _recent_context(ctx, "veteran")
        return (
            f"近期发言：\n{recent}\n\n"
            "辩论已经够充分了。作为见过最多的老炮，给出你的最终裁决。"
            "必须明确说出「老夫的判决是」，然后一句话总结你的立场。"
        )
    if last:
        return (
            f"{last}\n\n"
            "从你的江湖经验出发，直接反驳这个观点。"
            "讲一个相关的真实手牌来支撑你的判断，语气可以强硬一点。"
        )
    recent = _recent_context(ctx, "veteran")
    return (
        f"近期发言：\n{recent}\n\n"
        "指出最脱离实战的那个论点，用你的江湖经验痛击它。"
    )


def mtt_pro_prompt(ctx: DialogueContext) -> str:
    # 开场
    if not ctx.all_from("mtt_pro"):
        return (
            "请从锦标赛ICM角度亮出你的立场。"
            "给出一个具体场景：泡沫期200BB深筹，AA对KK，ICM折扣如何让这手牌的决策变复杂？"
            "然后说明为什么大多数人讨论这个问题时都在用错误的框架。"
        )
    last = _last_speaker_context(ctx, "mtt_pro")
    if last:
        return (
            f"{last}\n\n"
            "请直接回应这个观点。如果对方忽略了ICM，用具体的筹码深度场景说明影响有多大；"
            "如果对方已经考虑了ICM，指出他们的模型哪里还不完整。"
        )
    recent = _recent_context(ctx, "mtt_pro")
    return (
        f"近期发言：\n{recent}\n\n"
        "指出哪个论点在锦标赛场景下是致命错误，给出ICMizer的数据支撑。"
    )


def fish_prompt(ctx: DialogueContext) -> str:
    # 开场
    if not ctx.all_from("fish"):
        return (
            "你是个业余玩家，刚学会德州扑克。"
            "请用最朴素的直觉表达你的立场：拿到AA遇到all in该怎么办？"
            "然后问一个你真的搞不懂的问题——关于那些专家说的术语。"
        )
    last = _last_speaker_context(ctx, "fish")
    if last:
        return (
            f"{last}\n\n"
            "你听不太懂这些复杂理论，但你觉得哪里怪怪的。"
            "用大白话说出你的困惑，再问一个听起来天真但其实挺犀利的问题。"
            "不要假装懂你不懂的东西。"
        )
    recent = _recent_context(ctx, "fish")
    return (
        f"专家们刚才说：\n{recent}\n\n"
        "用大白话说说你理解了什么、没理解什么，然后问一个天真的问题。"
    )


def exploiter_prompt(ctx: DialogueContext) -> str:
    # 开场
    if not ctx.all_from("exploiter"):
        return (
            "请从exploit（针对性剥削）角度亮出你的立场。"
            "论证翻前AA遇all in的决策核心是对手的pushing range，而不是AA本身有多强。"
            "举一个极端例子：什么情况下fold AA才是exploit最大化？"
        )
    last = _last_speaker_context(ctx, "exploiter")
    if last:
        return (
            f"{last}\n\n"
            "直接回应这个观点。指出对方是否犯了'忽略对手range'的错误，"
            "或者对方的论点在range分析上有什么漏洞。"
            "如果是鱼的问题，认真回答——鱼有时候问出的问题比专家还深刻。"
        )
    recent = _recent_context(ctx, "exploiter")
    return (
        f"近期发言：\n{recent}\n\n"
        "指出最严重的'忽略range'错误，用具体的range数据反驳。"
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run(mock: bool = False) -> None:
    project_path = str(Path(__file__).parent.parent.parent)

    if mock:
        from harness.runners.base import AbstractRunner, RunnerResult
        from unittest.mock import AsyncMock, MagicMock

        # 回合模式：按 next_speaker 逻辑顺序预设回复
        # 开场5回合 + 辩论回合（按 DEBATE_CYCLE 循环）
        mock_script: list[tuple[str, str]] = [
            # 开场：各亮立场
            ("gto",       "AA对任意两张牌equity超80%，对KK是81%，对AKs是87%。翻前call是纯数学结论，不存在讨论空间。任何说'要看情况'的论点，本质上是在用情绪替代EV计算。"),
            ("mtt_pro",   "GTO机器忽略了一个关键变量：筹码价值的非线性。泡沫期200BB深筹，ICMizer跑出来某些场景AA对KK的锦标赛EV是负的。chip-EV框架在锦标赛里根本不成立。"),
            ("fish",      "AA是最好的牌！不call还等什么？！不过……ICM是什么意思？EV我知道，就是'期望值'，对吧？那ICM是'期望'加什么？"),
            ("exploiter", "两位专家都犯了同一个错误：讨论AA有多强，而不是讨论对手用什么range推。对手只用KK+推？call。对手100%range推？call。对手只用AA推？EV≈0，随机策略。牌本身是次要变量。"),
            ("veteran",   "我见过一个老头在$5/$10连续20年只用AA/KK推，牌桌上所有人都知道。有一次他推了，我手握KK，fold了。他翻出AA。数学说call是对的——但那天fold救了我200BB。读人不是玄学，是信息处理。"),
            # 辩论弧度：veteran↔gto 交锋
            ("veteran",   "GTO机器，你说大样本下读人无效。但扑克是有限次数的游戏，你我不会打够100万手。在样本量有限的实战里，一次正确的read可以改变整场的胜负。你的'大样本'论是在偷换概念。"),
            ("gto",       "老炮，你的'200BB fold KK'案例恰恰证明了我的观点：你赢了那一手，但如果在同样场景下打100次，fold KK的EV是负的。你把一次幸运的结果当成了策略正确性的证明，这是典型的结果导向谬误。"),
            ("exploiter", "两位争的其实是同一件事的两面。GTO说的是'对未知对手的最优下限策略'，老炮说的是'对已知对手的最优上限策略'。问题是：你真的'认识'你的对手吗？还是你以为你认识？"),
            ("mtt_pro",   "Exploiter，你说的'认识对手'在锦标赛里还多了一层复杂性：即使你100%确定对手在bluff，如果call出局会让你泡沫期出局，锦标赛EV可能仍然是负的。情境决定一切。"),
            ("fish",      "等等……如果对手只用AA推，那不就是两个AA同时出现了吗？德扑里一副牌只有两张A……哦不对，是四张。那如果两个人都有AA，谁赢？"),
            # 继续循环，veteran开始收尾
            ("veteran",   "鱼问了个好问题——如果双方都有AA，是chop（平分底池）。这恰好说明一点：AA的价值来自对手没有AA。Exploiter说的range分析本质上就是在估算对手有AA的概率。读人和range分析，是同一件事的不同表达。"),
            ("gto",       "老炮这次说的有道理——但'读人'的问题是误差率太高。求解器可以精确计算range，人脑的read充满噪声。在你'感觉'对手有KK的100次里，有多少次你是错的？我敢打赌超过40%。"),
            ("exploiter", "GTO，你说的40%误差率——这个数字从哪来的？如果一个专业玩家的read准确率能到65%，exploit策略的EV就已经超过GTO了。问题不是'读人有没有用'，是'你的读人够不够准'。"),
            ("mtt_pro",   "各位，有一点你们都还没提：现金桌和锦标赛的read价值是不同的。锦标赛里，一次错误的read可以结束你的比赛；现金桌里，你可以重新buy-in。这让风险承受和read的价值完全不同。"),
            ("fish",      "所以……结论是什么？是该call还是不call啊？你们说了这么多，我更搞不懂了。AA遇到all in，我还是觉得应该call，因为AA是最好的牌。但我现在也想知道对手是什么人了……"),
            # veteran收尾（第15+回合触发强制收尾提示）
            ("veteran",   "老夫的判决是：翻前AA遇all in，默认call，但不是因为'AA最强'。是因为在没有充分信息的情况下，call的EV确实为正。然而，当你对对手有清晰判断时——无论是通过reads还是range分析——那个判断才是真正的决策依据。数学给你底线，读人给你上限。"),
        ]
        _seq = [0]

        class MockRunner(AbstractRunner):
            async def execute(self, prompt, *, system_prompt, session_id, **kwargs) -> RunnerResult:
                idx = _seq[0]
                _seq[0] += 1
                if idx < len(mock_script):
                    role, text = mock_script[idx]
                else:
                    text = "（发言结束）"
                return RunnerResult(text=text, tokens_used=10, session_id=f"mock-{idx}")

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
        max_rounds=20,          # 回合模式下作为 max_turns 的计算基数
        max_turns=20,           # 最多20回合
        next_speaker=next_speaker,
        until=lambda ctx: "老夫的判决是" in (ctx.last_from("veteran") or ""),
        roles=[
            Role(
                name="gto",
                system_prompt=(
                    "你是一台GTO扑克求解器的化身，只相信数学和期望值。"
                    "你对'读人'和'直觉'嗤之以鼻，认为所有非EV论点都是情绪化的。"
                    "发言简洁犀利，喜欢引用具体数字，偶尔流露出对其他玩家的优越感。"
                    "当被直接点名反驳时，必须正面回应，不能回避。"
                ),
                prompt=gto_prompt,
            ),
            Role(
                name="mtt_pro",
                system_prompt=(
                    "你是一名锦标赛职业选手，深谙ICM（独立筹码模型）。"
                    "你认为现金桌和锦标赛是两个完全不同的游戏，"
                    "大多数讨论翻前策略的人都忽略了筹码深度和ICM压力。"
                    "发言学术而严谨，但当被质疑时会强力回击。"
                ),
                prompt=mtt_pro_prompt,
            ),
            Role(
                name="fish",
                system_prompt=(
                    "你是一个刚学会德州扑克的业余玩家，对理论一无所知。"
                    "你的逻辑很简单：好牌就该all in，坏牌就fold。"
                    "你经常无意中问出让专家无法反驳的天真问题。"
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
                    "说话有点大牌，但确实有两把刷子。"
                    "辩论充分后，你会给出「老夫的判决是」，总结你的最终裁决。"
                ),
                prompt=veteran_prompt,
            ),
        ],
    )

    print("=" * 60)
    print("五方辩论：翻前 AA 遇对手 All In，该 Call 吗？")
    print("（回合模式：动态点名，直接交锋）")
    print("=" * 60)

    pr = await harness.pipeline([dialogue])
    output = pr.results[0].output

    print(f"\n共 {output.rounds_completed} 回合，{len(output.turns)} 次发言\n")

    role_icons = {
        "gto":       "🤖 GTO机器",
        "veteran":   "🧓 江湖老炮",
        "mtt_pro":   "🏆 锦标赛鬼才",
        "fish":      "🐟 超级鱼",
        "exploiter": "🎭 心理战术师",
    }

    for i, turn in enumerate(output.turns):
        phase = "开场" if i < len(OPENING_ORDER) else "辩论"
        label = role_icons.get(turn.role_name, turn.role_name)
        print(f"\n{'─' * 60}")
        print(f"  回合 {i + 1}【{phase}】{label}")
        print(f"{'─' * 60}")
        print(turn.content)

    print("\n" + "=" * 60)
    final_label = role_icons.get(output.final_speaker, output.final_speaker)
    print(f"最终判决 — {final_label}")
    print("=" * 60)
    print(output.final_content)


if __name__ == "__main__":
    mock_mode = "--mock" in sys.argv
    asyncio.run(run(mock=mock_mode))

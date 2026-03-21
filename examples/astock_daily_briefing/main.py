"""astock_daily_briefing.py — 每天早上8点，抓取A股热点，发送 Telegram 晨报。

pipeline:
  FunctionTask [市场热度]  — akshare 获取昨日涨停板，输出热门行业+强势股摘要
      ↓
  LLMTask [分析热点]       — 结合涨停数据 + WebSearch，提炼3件大事，格式化Telegram消息
      ↓
  FunctionTask [发送]      — httpx POST to Telegram Bot API

环境变量（必须）：
  TELEGRAM_BOT_TOKEN  — Telegram Bot Token（从 @BotFather 获取）
  TELEGRAM_CHAT_ID    — 目标 Chat ID（可用 @userinfobot 查询）

安装依赖：
  uv add akshare

运行：
  python examples/astock_daily_briefing.py --run-once   # 立即运行一次（测试）
  python examples/astock_daily_briefing.py              # 启动定时任务，每天08:00运行
"""
from __future__ import annotations

import asyncio
import os
from datetime import date, timedelta

import httpx

from harness import FunctionTask, Harness, LLMTask


# ---------------------------------------------------------------------------
# Task 0: 获取昨日涨停板数据
# ---------------------------------------------------------------------------

def fetch_market_heat(results: list) -> str:
    """用 akshare 拉取最近交易日的涨停板数据，返回文本摘要。"""
    import akshare as ak

    # 找到最近的交易日（跳过周末）
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:  # 5=Saturday, 6=Sunday
        d -= timedelta(days=1)

    date_str = d.strftime("%Y%m%d")
    df = ak.stock_zt_pool_em(date=date_str)

    if df.empty:
        return f"日期：{d}，暂无涨停板数据（可能是非交易日）。"

    # 按行业统计涨停数
    industry_counts = (
        df.groupby("所属行业").size().sort_values(ascending=False).head(6)
    )
    # 封板资金最高的强势股
    top_stocks = df.nlargest(5, "封板资金")[["名称", "所属行业", "连板数"]]

    lines = [
        f"📅 交易日：{d.strftime('%Y-%m-%d')}",
        f"📈 涨停总数：{len(df)} 只",
        "",
        "🔥 热门行业（涨停数量）：",
    ]
    for industry, count in industry_counts.items():
        lines.append(f"  {industry}: {count} 只")

    lines += ["", "💪 强势股（封板资金 TOP5）："]
    for _, row in top_stocks.iterrows():
        lines.append(
            f"  {row['名称']} | {row['所属行业']} | 连板 {row['连板数']} 天"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 1: LLM 分析热点，生成 Telegram 消息
# ---------------------------------------------------------------------------

def make_briefing_prompt(results: list) -> str:
    market_data: str = results[0].output
    today = date.today().strftime("%Y-%m-%d")
    return f"""今天是 {today} 早上，请为我生成一份 A股晨报。

## 昨日涨停板数据
{market_data}

## 任务
1. 使用 WebSearch 搜索以下关键词，获取今日最新财经资讯：
   - 「A股 {today} 热点」
   - 「A股 {today} 今日行情 主线」
   - 「沪深 {today} 财经新闻」
2. 结合涨停板数据和搜索结果，提炼今日A股最值得关注的3件事
3. 按以下格式输出（Telegram Markdown）：

📊 *A股早报 {today}*

1️⃣ *[事件标题，10字以内]*
[2~3句：是什么事、为什么热、对市场的影响]

2️⃣ *[事件标题]*
[2~3句说明]

3️⃣ *[事件标题]*
[2~3句说明]

---
_数据：东方财富涨停板 + 财经新闻_

只输出消息正文，不要添加任何额外说明或代码块。"""


# ---------------------------------------------------------------------------
# Task 2: 发送 Telegram 消息
# ---------------------------------------------------------------------------

def send_telegram(results: list) -> bool:
    """把 LLMTask 生成的消息发送到 Telegram。"""
    message: str = results[1].output
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    resp = httpx.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
        timeout=30,
    )
    resp.raise_for_status()
    print(f"\n✅ Telegram 消息发送成功（{resp.status_code}）")
    return True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_tasks() -> list:
    return [
        FunctionTask(fn=fetch_market_heat),          # Task 0
        LLMTask(prompt=make_briefing_prompt),         # Task 1
        FunctionTask(fn=send_telegram),               # Task 2
    ]


async def run_once() -> None:
    """立即运行一次（用于测试）。"""
    _check_env()
    h = Harness(
        project_path=".",
        stream_callback=lambda text: print(text, end="", flush=True),
    )
    pr = await h.pipeline(build_tasks(), name="astock-briefing-manual")
    print(f"\n完成 | {pr.total_duration_seconds:.1f}s | tokens: {pr.total_tokens:,}")


async def run_scheduled() -> None:
    """启动定时任务，每天 08:00 运行。"""
    _check_env()
    h = Harness(project_path=".")
    h.schedule(tasks=build_tasks(), cron="0 8 * * *", name="astock-briefing")
    print("⏰ 定时任务已启动，每天 08:00 发送 A股晨报。按 Ctrl+C 停止。")
    await h.start()


def _check_env() -> None:
    missing = [v for v in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"缺少环境变量：{', '.join(missing)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main_cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="A股每日热点晨报 → Telegram")
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="立即运行一次（用于测试，不启动定时任务）",
    )
    args = parser.parse_args()

    if args.run_once:
        asyncio.run(run_once())
    else:
        asyncio.run(run_scheduled())


if __name__ == "__main__":
    main_cli()

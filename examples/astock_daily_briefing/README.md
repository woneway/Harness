# astock_daily_briefing — A股每日热点晨报 → Telegram

每天早上 8 点，自动抓取 A 股涨停板数据，结合财经新闻，提炼 3 件大事，推送到 Telegram。

## Pipeline 结构

```
FunctionTask [获取昨日涨停板]  — akshare.stock_zt_pool_em(), 输出热门行业+强势股摘要
    ↓
LLMTask [分析热点]             — 结合涨停数据 + WebSearch 财经新闻，提炼3件大事，格式化Telegram消息
    ↓
FunctionTask [发送Telegram]    — httpx POST to Telegram Bot API
```

## 环境要求

**依赖：**
```bash
uv add akshare
```

**环境变量：**
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"   # 从 @BotFather 获取
export TELEGRAM_CHAT_ID="your_chat_id"       # 用 @userinfobot 查询
```

## 运行

```bash
# 立即运行一次（测试）
uv run python examples/astock_daily_briefing/main.py --run-once

# 启动定时任务（每天 08:00 自动运行，持续后台运行）
uv run python examples/astock_daily_briefing/main.py
```

## 输出示例

```
📊 *A股早报 2026-03-21*

1️⃣ *AI算力板块全线涨停*
受海外AI大模型新突破消息刺激，国内算力基础设施概念强势爆发，寒武纪、海光信息等10余只个股涨停。市场预期国内AI投资加速，板块短期情绪高涨。

2️⃣ *医药创新政策利好落地*
国家药监局发布创新药快速审批新规，多个在研管线有望加速上市。医药板块整体跟涨，CXO龙头率先反弹。

3️⃣ *地产债务重组取得进展*
某头部房企公告债务展期方案获债权人通过，带动地产板块情绪修复，保利、万科等国央企地产股小幅上涨。

---
_数据：东方财富涨停板 + 财经新闻_
```

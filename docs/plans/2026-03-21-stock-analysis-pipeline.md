# Stock Analysis Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `examples/stock_analysis.py` — a Harness pipeline that fetches A-share data, calculates technical indicators, runs three LLM analysis stages, then outputs a buy/sell/hold verdict to terminal + Markdown file.

**Architecture:** 7-step sequential pipeline: FunctionTask (data fetch) → FunctionTask (indicators) → LLMTask (technical) → LLMTask (fundamental) → LLMTask (news sentiment) → LLMTask (verdict) → FunctionTask (save/print). Each step receives all previous results via `results: list[Result]`.

**Tech Stack:** Python 3.11+, akshare, pandas-ta, pandas, harness (LLMTask / FunctionTask), argparse

---

## Prerequisites

Before starting, install dependencies:

```bash
pip install akshare pandas-ta pandas
```

Verify akshare works:

```python
import akshare as ak
df = ak.stock_zh_a_hist(symbol="600036", period="daily", adjust="qfq")
print(df.tail(3))
```

Expected: DataFrame with columns `日期`, `开盘`, `收盘`, `最高`, `最低`, `成交量`.

---

### Task 1: Scaffold the file with imports and argparse

**Files:**
- Create: `examples/stock_analysis.py`

**Step 1: Create the file with docstring, imports, and CLI entrypoint**

```python
"""stock_analysis.py — 对单只 A 股（沪深）进行综合分析，给出买/卖/持有信号。

pipeline：
  FunctionTask [数据获取]  ：akshare 获取近 120 天日K线 + 基本面数据
      ↓
  FunctionTask [技术指标]  ：pandas-ta 计算 MA/MACD/RSI/布林带
      ↓
  LLMTask [技术分析]       ：分析指标，输出技术信号
      ↓
  LLMTask [基本面分析]     ：分析 PE/PB/ROE，给出估值信号
      ↓
  LLMTask [新闻情绪]       ：WebSearch 搜索近期新闻，输出情绪信号
      ↓
  LLMTask [综合裁决]       ：整合三维信号，给出买入/持有/卖出 + 理由
      ↓
  FunctionTask [输出]      ：打印终端摘要 + 保存 <代码>_analysis.md

运行：
    python examples/stock_analysis.py 600036
    python examples/stock_analysis.py 000858 --output ./reports
    python examples/stock_analysis.py 600519 --output .
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd
import pandas_ta as ta

from harness import FunctionTask, Harness, LLMTask


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="A 股综合分析：技术面 + 基本面 + 新闻情绪 → 买/卖/持有信号",
    )
    parser.add_argument("symbol", help="股票代码，如 600036（招商银行）")
    parser.add_argument(
        "--output",
        default=".",
        help="报告保存目录（默认：当前目录）",
    )
    args = parser.parse_args()
    asyncio.run(run(symbol=args.symbol, output_dir=Path(args.output).resolve()))


if __name__ == "__main__":
    main_cli()
```

**Step 2: Run to verify no import errors**

```bash
python examples/stock_analysis.py --help
```

Expected output includes: `usage: stock_analysis.py [-h] [--output OUTPUT] symbol`

**Step 3: Commit**

```bash
git add examples/stock_analysis.py
git commit -m "feat: scaffold stock_analysis.py with imports and argparse"
```

---

### Task 2: Implement Task 0 — fetch stock data

**Files:**
- Modify: `examples/stock_analysis.py`

**Step 1: Add the dataclass and fetch function before `main_cli`**

```python
from dataclasses import dataclass

@dataclass
class StockData:
    symbol: str
    name: str
    kline: pd.DataFrame          # columns: date, open, high, low, close, volume
    fundamentals: dict           # pe_ttm, pb, roe, revenue_growth


def fetch_stock_data(symbol: str) -> callable:
    """闭包：捕获 symbol，返回 FunctionTask 兼容的函数。"""

    def _fetch(results: list) -> StockData:
        end = datetime.now()
        start = end - timedelta(days=180)   # 多取 180 天确保 120 交易日

        # 日K线（前复权）
        kline_raw = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
        kline = kline_raw.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
        })[["date", "open", "high", "low", "close", "volume"]].tail(120)

        # 股票名称
        info = ak.stock_individual_info_em(symbol=symbol)
        name_row = info[info["item"] == "股票简称"]
        name = name_row["value"].iloc[0] if not name_row.empty else symbol

        # 基本面：个股信息（PE、PB）
        fundamentals: dict = {}
        for item, key in [("市盈率(动)", "pe_ttm"), ("市净率", "pb")]:
            row = info[info["item"] == item]
            fundamentals[key] = float(row["value"].iloc[0]) if not row.empty else None

        # ROE 和营收增速从财务数据获取（取最新一期）
        try:
            profit = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按年度")
            if not profit.empty:
                fundamentals["roe"] = float(str(profit.iloc[0].get("加权净资产收益率", "") or "").replace("%", "") or 0)
                rev_cols = [c for c in profit.columns if "营业总收入" in c and "增长率" in c]
                fundamentals["revenue_growth"] = float(str(profit.iloc[0].get(rev_cols[0], "") or "").replace("%", "") or 0) if rev_cols else None
        except Exception:
            fundamentals.setdefault("roe", None)
            fundamentals.setdefault("revenue_growth", None)

        return StockData(symbol=symbol, name=name, kline=kline, fundamentals=fundamentals)

    return _fetch
```

**Step 2: Add quick smoke-test (manual, not automated)**

```bash
python -c "
import asyncio, sys
sys.path.insert(0, '.')
from examples.stock_analysis import fetch_stock_data
fn = fetch_stock_data('600036')
data = fn([])
print(data.name, len(data.kline), 'rows')
print(data.fundamentals)
"
```

Expected: prints stock name, ~120, and dict with pe_ttm/pb values.

**Step 3: Commit**

```bash
git add examples/stock_analysis.py
git commit -m "feat: add Task 0 - fetch akshare K-line and fundamentals"
```

---

### Task 3: Implement Task 1 — calculate technical indicators

**Files:**
- Modify: `examples/stock_analysis.py`

**Step 1: Add dataclass and indicator function**

```python
@dataclass
class Indicators:
    symbol: str
    name: str
    kline: pd.DataFrame          # original + indicator columns
    latest: dict                 # latest row as dict for easy LLM consumption
    fundamentals: dict


def calculate_indicators(results: list) -> Indicators:
    stock: StockData = results[0].output
    df = stock.kline.copy()

    # MA
    df["ma5"]  = ta.sma(df["close"], length=5)
    df["ma20"] = ta.sma(df["close"], length=20)
    df["ma60"] = ta.sma(df["close"], length=60)

    # MACD (fast=12, slow=26, signal=9)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)          # adds MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9

    # RSI(14)
    df["rsi14"] = ta.rsi(df["close"], length=14)

    # Bollinger Bands (20, 2)
    bbands = ta.bbands(df["close"], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)        # adds BBL_20_2.0, BBM_20_2.0, BBU_20_2.0

    latest = df.iloc[-1].to_dict()

    return Indicators(
        symbol=stock.symbol,
        name=stock.name,
        kline=df,
        latest=latest,
        fundamentals=stock.fundamentals,
    )
```

**Step 2: Smoke-test indicators**

```bash
python -c "
import asyncio, sys, types
sys.path.insert(0, '.')
from examples.stock_analysis import fetch_stock_data, calculate_indicators
data = fetch_stock_data('600036')([])
result = types.SimpleNamespace(output=data)
ind = calculate_indicators([result])
latest = ind.latest
print('close:', latest['close'])
print('ma20:', round(latest.get('ma20', 0), 2))
print('rsi14:', round(latest.get('rsi14', 0), 2))
"
```

Expected: numeric values printed without error.

**Step 3: Commit**

```bash
git add examples/stock_analysis.py
git commit -m "feat: add Task 1 - calculate MA/MACD/RSI/Bollinger indicators"
```

---

### Task 4: Implement Task 2 — LLMTask technical analysis prompt

**Files:**
- Modify: `examples/stock_analysis.py`

**Step 1: Add prompt builder function**

```python
def make_technical_prompt(results: list) -> str:
    ind: Indicators = results[1].output
    l = ind.latest                          # latest indicator values
    today = datetime.now().strftime("%Y-%m-%d")

    def fmt(v) -> str:
        return f"{v:.2f}" if v is not None and not (isinstance(v, float) and __import__('math').isnan(v)) else "N/A"

    return f"""你是专业的 A 股技术分析师。今天是 {today}。

请对 {ind.name}（{ind.symbol}）进行技术面分析，基于以下最新指标数据：

## 价格与均线
- 当前收盘价：{fmt(l.get('close'))}
- MA5：{fmt(l.get('ma5'))}
- MA20：{fmt(l.get('ma20'))}
- MA60：{fmt(l.get('ma60'))}

## MACD（12,26,9）
- DIF（MACD线）：{fmt(l.get('MACD_12_26_9'))}
- DEA（信号线）：{fmt(l.get('MACDs_12_26_9'))}
- 柱状值（MACD Histogram）：{fmt(l.get('MACDh_12_26_9'))}

## RSI(14)
- RSI：{fmt(l.get('rsi14'))}

## 布林带（20,2）
- 上轨：{fmt(l.get('BBU_20_2.0'))}
- 中轨：{fmt(l.get('BBM_20_2.0'))}
- 下轨：{fmt(l.get('BBL_20_2.0'))}

请分析：
1. 均线形态（多头/空头排列，价格与均线关系）
2. MACD 信号（金叉/死叉，柱状图趋势）
3. RSI 状态（超买/超卖/中性，参考 30/70 阈值）
4. 布林带位置（上轨突破/下轨支撑/中轨压制）

最终输出：
- **技术信号：** 看涨 / 看跌 / 中性（选一）
- **信号强度：** 强 / 中 / 弱
- **关键理由：** 3 条简短的 bullet points"""
```

**Step 2: Commit**

```bash
git add examples/stock_analysis.py
git commit -m "feat: add Task 2 - technical analysis LLM prompt"
```

---

### Task 5: Implement Task 3 — LLMTask fundamental analysis prompt

**Files:**
- Modify: `examples/stock_analysis.py`

**Step 1: Add prompt builder**

```python
def make_fundamental_prompt(results: list) -> str:
    ind: Indicators = results[1].output
    f = ind.fundamentals
    today = datetime.now().strftime("%Y-%m-%d")

    def fmt_pct(v) -> str:
        return f"{v:.1f}%" if v is not None else "N/A"

    def fmt_x(v) -> str:
        return f"{v:.1f}x" if v is not None else "N/A"

    return f"""你是专业的 A 股基本面分析师。今天是 {today}。

请对 {ind.name}（{ind.symbol}）进行基本面分析，基于以下数据：

## 估值指标
- 市盈率 PE(TTM)：{fmt_x(f.get('pe_ttm'))}
- 市净率 PB：{fmt_x(f.get('pb'))}

## 盈利能力
- 加权净资产收益率 ROE：{fmt_pct(f.get('roe'))}
- 营业总收入增长率：{fmt_pct(f.get('revenue_growth'))}

请分析：
1. 估值水平（与 A 股同行业平均 PE/PB 比较，判断高估/低估/合理）
2. 盈利质量（ROE 是否健康，增长是否持续）
3. 综合基本面评价

最终输出：
- **估值信号：** 低估 / 合理 / 高估（选一）
- **基本面质量：** 优秀 / 良好 / 一般 / 较差
- **关键理由：** 3 条简短的 bullet points"""
```

**Step 2: Commit**

```bash
git add examples/stock_analysis.py
git commit -m "feat: add Task 3 - fundamental analysis LLM prompt"
```

---

### Task 6: Implement Task 4 — LLMTask news sentiment prompt

**Files:**
- Modify: `examples/stock_analysis.py`

**Step 1: Add prompt builder**

```python
def make_sentiment_prompt(results: list) -> str:
    ind: Indicators = results[1].output
    today = datetime.now().strftime("%Y-%m-%d")

    return f"""你是专业的 A 股市场情绪分析师。今天是 {today}。

请使用 WebSearch 工具搜索 {ind.name}（{ind.symbol}）近期（最近 30 天）的新闻和市场情绪：

搜索要求：
1. 搜索"{ind.name} {today[:7]}"的近期新闻
2. 搜索"{ind.name} 利好 OR 利空"
3. 搜索"{ind.symbol} 机构评级 OR 研报"
4. 可选：搜索行业相关重大政策/事件

分析维度：
1. 重大公司事件（业绩预告/公告/人事变动/股权激励等）
2. 行业政策与监管动态
3. 机构评级变化（升级/降级）
4. 市场资金动向（主力买卖/北向资金等，如有数据）

最终输出：
- **情绪信号：** 积极 / 中性 / 消极（选一）
- **关键事件：** 3 条近期最重要的新闻摘要（每条附来源和日期）
- **风险提示：** 1-2 条需关注的潜在风险"""
```

**Step 2: Commit**

```bash
git add examples/stock_analysis.py
git commit -m "feat: add Task 4 - news sentiment LLM prompt with WebSearch"
```

---

### Task 7: Implement Task 5 — LLMTask verdict prompt

**Files:**
- Modify: `examples/stock_analysis.py`

**Step 1: Add prompt builder**

```python
def make_verdict_prompt(results: list) -> str:
    ind: Indicators = results[1].output
    technical: str = results[2].output
    fundamental: str = results[3].output
    sentiment: str = results[4].output
    today = datetime.now().strftime("%Y-%m-%d")

    return f"""你是资深 A 股投资顾问。今天是 {today}。

请综合以下三维分析，对 {ind.name}（{ind.symbol}）给出投资建议：

## 技术面分析结论
{technical}

## 基本面分析结论
{fundamental}

## 新闻情绪分析结论
{sentiment}

请综合上述三维信号，给出最终投资建议：

**输出格式（严格按此格式）：**

---
结论：[买入 / 持有 / 卖出]
信心评分：[1-5]/5（1=极低信心，5=极高信心）

核心理由：
1. [技术面：一句话]
2. [基本面：一句话]
3. [情绪面：一句话]

风险提示：
- [风险1]
- [风险2]

免责声明：本分析仅供参考，不构成投资建议，投资有风险，入市需谨慎。
---

注意：如果三维信号严重分歧，信心评分应≤2，并在理由中说明分歧。"""
```

**Step 2: Commit**

```bash
git add examples/stock_analysis.py
git commit -m "feat: add Task 5 - verdict LLM prompt synthesizing all signals"
```

---

### Task 8: Implement Task 6 — FunctionTask save and print

**Files:**
- Modify: `examples/stock_analysis.py`

**Step 1: Add the save/print function**

```python
def make_output_fn(output_dir: Path) -> callable:
    """闭包：捕获 output_dir，返回 FunctionTask 兼容的函数。"""

    def _output(results: list) -> Path:
        ind: Indicators = results[1].output
        technical: str = results[2].output
        fundamental: str = results[3].output
        sentiment: str = results[4].output
        verdict: str = results[5].output
        today = datetime.now().strftime("%Y-%m-%d")

        # ── 终端摘要 ────────────────────────────────────────────
        print(f"\n{'=' * 50}")
        print(f"  股票：{ind.name} ({ind.symbol})")
        print(f"  分析日期：{today}")
        print()
        print(verdict)
        print(f"{'=' * 50}\n")

        # ── 保存完整报告 ─────────────────────────────────────────
        report = f"""---
title: {ind.name}（{ind.symbol}）股票分析报告
date: {today}
---

# {ind.name}（{ind.symbol}）综合分析报告

> 分析日期：{today}

---

## 一、技术面分析

{technical}

---

## 二、基本面分析

{fundamental}

---

## 三、新闻情绪分析

{sentiment}

---

## 四、综合裁决

{verdict}

---

*本报告由 Harness AI Pipeline 自动生成，仅供参考，不构成投资建议。*
"""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{ind.symbol}_analysis.md"
        output_path.write_text(report, encoding="utf-8")
        return output_path

    return _output
```

**Step 2: Commit**

```bash
git add examples/stock_analysis.py
git commit -m "feat: add Task 6 - save Markdown report and print terminal summary"
```

---

### Task 9: Wire the complete pipeline in `run()`

**Files:**
- Modify: `examples/stock_analysis.py`

**Step 1: Add the `run()` async function before `main_cli`**

```python
async def run(symbol: str, output_dir: Path) -> None:
    h = Harness(
        project_path=".",
        stream_callback=lambda text: print(text, end="", flush=True),
    )

    print("=" * 50)
    print(f"  开始分析：{symbol}")
    print(f"  报告输出：{output_dir}")
    print("=" * 50)
    print()

    pr = await h.pipeline(
        [
            FunctionTask(fn=fetch_stock_data(symbol)),          # Task 0：获取数据
            FunctionTask(fn=calculate_indicators),              # Task 1：计算指标
            LLMTask(prompt=make_technical_prompt),              # Task 2：技术分析
            LLMTask(prompt=make_fundamental_prompt),            # Task 3：基本面分析
            LLMTask(prompt=make_sentiment_prompt),              # Task 4：新闻情绪
            LLMTask(prompt=make_verdict_prompt),                # Task 5：综合裁决
            FunctionTask(fn=make_output_fn(output_dir)),        # Task 6：保存报告
        ],
        name=f"stock-analysis-{symbol}",
    )

    saved_path: Path = pr.results[6].output

    print(f"\n{'=' * 50}")
    print(f"  分析完成")
    print(f"  报告：{saved_path}")
    print(f"  耗时：{pr.total_duration_seconds:.1f}s  |  tokens：{pr.total_tokens:,}")
    print(f"  Run ID：{pr.run_id[:8]}")
    print(f"{'=' * 50}")
```

**Step 2: Run end-to-end test**

```bash
python examples/stock_analysis.py 600036 --output ./reports
```

Expected:
- Streaming LLM output visible in terminal
- Terminal summary printed with 结论/信心/理由/风险
- File `reports/600036_analysis.md` created

**Step 3: Commit**

```bash
git add examples/stock_analysis.py
git commit -m "feat: wire complete 7-step stock analysis pipeline"
```

---

### Task 10: Final verification and cleanup

**Step 1: Test with a second stock**

```bash
python examples/stock_analysis.py 000858 --output ./reports
```

Expected: `reports/000858_analysis.md` created.

**Step 2: Test error case — invalid symbol**

```bash
python examples/stock_analysis.py 999999 --output ./reports
```

Expected: error message from akshare, not a silent hang.

**Step 3: Check report file structure**

```bash
cat reports/600036_analysis.md | head -30
```

Expected: YAML frontmatter + four sections (技术/基本面/情绪/裁决).

**Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete stock_analysis.py - A-share buy/sell/hold pipeline"
```

---

## File Order Summary

All code goes into a single file: `examples/stock_analysis.py`

**Function definition order (top to bottom):**

```
docstring + imports
StockData dataclass
fetch_stock_data(symbol) → callable
Indicators dataclass
calculate_indicators(results) → Indicators
make_technical_prompt(results) → str
make_fundamental_prompt(results) → str
make_sentiment_prompt(results) → str
make_verdict_prompt(results) → str
make_output_fn(output_dir) → callable
run(symbol, output_dir) → coroutine
main_cli() → None
if __name__ == "__main__": main_cli()
```

## Common Pitfalls

| Issue | Fix |
|-------|-----|
| `NaN` in indicator values | First 60 rows lack MA60 data — always use `tail(120)` and handle NaN with `fmt()` helper |
| akshare column names change | Use `.rename(columns={...})` immediately after fetch |
| pandas-ta column naming | MACD columns are `MACD_12_26_9`, `MACDh_12_26_9`, `MACDs_12_26_9` — verify with `df.columns` |
| LLM sees `results[1]` not `results[0]` | indicators is always index 1; raw stock data is index 0 |
| `math.isnan()` on non-float | Wrap in `isinstance(v, float)` check inside `fmt()` |

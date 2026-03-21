# research_report — 给定主题，联网深度调研，生成专业 Markdown 报告

## Pipeline 结构

**deep 模式（默认，5 阶段）：**
```
LLMTask [多维搜索]   — 8 个维度 × 多次 WebSearch
    ↓
LLMTask [大纲]       — 基于搜索结果制定结构化大纲
    ↓
LLMTask [正文]       — 按大纲逐章撰写，引用搜索数据
    ↓
LLMTask [专家审校]   — 事实核查 + 补充遗漏 + 平衡性检查
    ↓
FunctionTask [保存]  — 写入 {主题}_{日期}.md
```

## 运行

```bash
# 深度调研（默认，最高质量）
uv run python examples/research_report/main.py "Claude Code"

# 指定输出目录
uv run python examples/research_report/main.py "大语言模型量化技术" --output ./reports

# 快速模式（3 阶段，~10分钟）
uv run python examples/research_report/main.py "OpenAI o3" --depth quick

# 均衡模式（4 阶段）
uv run python examples/research_report/main.py "Rust 异步运行时" --depth standard
```

## 深度选项

| 模式 | 阶段 | 适用场景 |
|------|------|----------|
| `quick` | 3 | 快速了解 |
| `standard` | 4 | 日常调研 |
| `deep` | 5 | 高质量报告（默认）|

> 注意：deep 模式调用 Claude CLI 4 次，耗时 20~40 分钟。

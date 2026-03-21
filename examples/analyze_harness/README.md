# analyze_harness — 代码分析 → 优化 → 复盘 三阶段流水线

对 Harness 自身代码进行深度分析，自动提出优化建议并执行，最后复盘改动效果。

## Pipeline 结构

```
FunctionTask [收集源码]   — 读取 harness/ 下所有 .py 文件
    ↓
LLMTask [分析]            — 输出问题清单（P0/P1/P2）+ 优化方案
    ↓
LLMTask [优化]            — 直接修改项目文件（bypassPermissions）
    ↓
FunctionTask [收集源码后] — 再次读取，用于对比
    ↓
LLMTask [复盘]            — 对比前后差异，评估改进完成情况
```

## 运行

```bash
uv run python examples/analyze_harness/main.py
```

> 注意：会调用 Claude CLI 3 次，耗时较长（5~15分钟）。

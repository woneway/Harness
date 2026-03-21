# code_stats — 统计 Harness 自身代码量

扫描 harness/ 源文件，统计行数，用 LLM 给出简短分析。用于验证框架基础功能（FunctionTask + ShellTask + LLMTask）。

## Pipeline 结构

```
FunctionTask [扫描文件]  — 列出 harness/ 下所有 .py 文件
    ↓
ShellTask [wc -l]        — 统计每个文件行数
    ↓
FunctionTask [生成报告]  — 解析 wc 输出，汇总数据
    ↓
LLMTask [分析]           — 评估代码规模，给出改进建议
```

## 运行

```bash
uv run python examples/code_stats/main.py
```

## 输出示例

```
📊 Harness 代码统计报告
  .py 文件数：18
  总行数：2,847
  最大文件：executor.py (412 行)
  耗时：12.34s
  Run ID：3a7f1b2c
```

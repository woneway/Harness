# iterationforge_style — IterationForge 风格的定时扫描+修复示例

演示 Harness 高级特性：`output_schema` 结构化输出、`Memory` 历史注入、`TaskConfig` 全局配置、`cron` 定时调度、`Task` 向后兼容别名。

## Pipeline 结构

```
FunctionTask [扫描] — mock：返回 ScanResult(issues, summary, severity)
    ↓
FunctionTask [修复] — mock：基于扫描结果，返回 FixResult(fixed_count, skipped)
```

> 实际使用时将 mock 替换为 LLMTask 即可。

## 运行

```bash
# 运行一次（含向后兼容演示 + 定时任务注册演示）
uv run python examples/iterationforge_style/main.py
```

## 演示内容

1. **run_once** — 执行扫描+修复流水线，输出结构化结果
2. **demo_deprecation_warning** — 使用 `Task(...)` 触发 DeprecationWarning
3. **demo_scheduled** — 注册 `cron="0 2 * * *"` 定时任务（不实际等待）

## 输出示例

```
🔍 启动代码扫描+修复流水线...
扫描结果: 发现 3 个问题，其中 1 个高优先级
修复结果: 成功修复 2/3 个问题
跳过: ['重复代码块（需要人工审查）']
✅ Run ID: f3e2d1c0

--- 向后兼容演示 ---
⚠️  捕获到 DeprecationWarning: Task is deprecated, use LLMTask instead
Task 类型: LLMTask（LLMTask 的别名）

--- 定时调度演示 ---
✅ 定时任务已注册（cron: 0 2 * * *）
```

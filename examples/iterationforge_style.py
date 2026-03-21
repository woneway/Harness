"""iterationforge_style.py — IterationForge 风格的定时扫描+修复示例。

展示 Harness 与 IterationForge 的集成方式：
- 定时运行（cron）
- LLMTask 链式调用
- Memory 历史注入
- output_schema 结构化输出
- Task 别名（向后兼容演示）

运行方式：
    python examples/iterationforge_style.py
"""

from __future__ import annotations

import asyncio
import warnings
from pydantic import BaseModel

from harness import Harness, LLMTask, FunctionTask, Memory, Task, TaskConfig


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class ScanResult(BaseModel):
    """代码扫描结果。"""

    issues: list[str]
    summary: str
    severity: str  # "low", "medium", "high"
    memory_update: str | None = None


class FixResult(BaseModel):
    """修复结果。"""

    fixed_count: int
    summary: str
    skipped: list[str]


# ---------------------------------------------------------------------------
# 模拟扫描函数
# ---------------------------------------------------------------------------


def mock_scan(results: list) -> ScanResult:
    """模拟代码扫描（实际使用 LLMTask）。"""
    return ScanResult(
        issues=["未处理的异常", "类型注解缺失", "重复代码块"],
        summary="发现 3 个问题，其中 1 个高优先级",
        severity="medium",
        memory_update="2026-03-21 扫描发现：类型注解覆盖率 60%，需要提升",
    )


def mock_fix(results: list) -> FixResult:
    """模拟修复（实际使用 LLMTask）。"""
    scan: ScanResult = results[0].output
    return FixResult(
        fixed_count=2,
        summary=f"成功修复 2/{len(scan.issues)} 个问题",
        skipped=["重复代码块（需要人工审查）"],
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def run_once() -> None:
    """执行一次扫描+修复流水线。"""
    h = Harness(
        project_path=".",
        memory=Memory(history_runs=3, max_tokens=1500),
        default_config=TaskConfig(timeout=300, max_retries=2),
    )

    tasks = [
        # 步骤 0：扫描项目（实际场景中使用 LLMTask）
        FunctionTask(fn=mock_scan, output_schema=ScanResult),

        # 步骤 1：根据扫描结果修复高优先级问题
        FunctionTask(fn=mock_fix, output_schema=FixResult),

        # 步骤 2：运行测试验证（Shell 命令）
        # ShellTask(cmd="python -m pytest tests/ -x -q"),
    ]

    print("🔍 启动代码扫描+修复流水线...")
    result = await h.pipeline(tasks, name="iterationforge-scan")

    scan_r = result.results[0]
    fix_r = result.results[1]
    print(f"\n扫描结果: {scan_r.output.summary}")
    print(f"修复结果: {fix_r.output.summary}")
    print(f"跳过: {fix_r.output.skipped}")
    print(f"\n✅ Run ID: {result.run_id}")


async def demo_deprecation_warning() -> None:
    """演示 Task 别名的 DeprecationWarning。"""
    print("\n--- 向后兼容演示 ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        task = Task(prompt="这是旧 API 的用法")
        if w and issubclass(w[0].category, DeprecationWarning):
            print(f"⚠️  捕获到 DeprecationWarning: {w[0].message}")
        print(f"Task 类型: {type(task).__name__}（LLMTask 的别名）")


async def demo_scheduled() -> None:
    """演示定时调度（仅打印，不实际等待）。"""
    print("\n--- 定时调度演示 ---")
    h = Harness(
        project_path=".",
        memory=Memory(history_runs=5),
    )

    # 注册定时任务（每天凌晨 2 点执行）
    h.schedule(
        tasks=[
            FunctionTask(fn=mock_scan, output_schema=ScanResult),
            FunctionTask(fn=mock_fix, output_schema=FixResult),
        ],
        cron="0 2 * * *",
        name="nightly-scan",
    )

    print("✅ 定时任务已注册（cron: 0 2 * * *）")
    print("   在生产环境中调用 await h.start() 启动调度器")


async def main() -> None:
    await run_once()
    await demo_deprecation_warning()
    await demo_scheduled()


if __name__ == "__main__":
    asyncio.run(main())

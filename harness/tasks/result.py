"""Result / PipelineResult — 执行结果数据类。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel


@dataclass(frozen=True)
class Result:
    """单个 Task 的执行结果。"""

    task_index: str  # 顺序步骤："0","1"; Parallel 子任务："2.0","2.1"
    task_type: Literal["llm", "function", "shell", "polling", "dialogue"]
    output: BaseModel | str | Any
    raw_text: str | None
    tokens_used: int
    duration_seconds: float
    success: bool
    error: str | None


@dataclass(frozen=True)
class PipelineResult:
    """整条 pipeline 的执行结果。"""

    run_id: str
    name: str | None
    results: list[Result]
    total_tokens: int
    total_duration_seconds: float


def result_by_type(results: list[Result], task_type: str, n: int = 0) -> Result:
    """从 pipeline results 按 task_type 取第 n 个结果（默认第 0 个）。

    在 FunctionTask.fn 中替代脆弱的整数下标访问，当 pipeline 顺序变化时报错更明确。

    Args:
        results:   FunctionTask.fn 接收的 list[Result]。
        task_type: "llm" | "function" | "shell" | "polling" | "dialogue"
        n:         第 n 个匹配（从 0 开始），默认 0。

    Raises:
        ValueError: 没有找到匹配的 task_type 或 n 超出范围。
    """
    matches = [r for r in results if r.task_type == task_type]
    if not matches:
        raise ValueError(
            f"No result with task_type={task_type!r}. "
            f"Available types: {[r.task_type for r in results]}"
        )
    if n >= len(matches):
        raise ValueError(
            f"result_by_type: n={n} out of range for task_type={task_type!r} "
            f"(found {len(matches)} match(es))"
        )
    return matches[n]

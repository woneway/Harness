"""compat.py — v1/v2 callable 双模式检测与调用。"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, Literal

from harness.state import State
from harness.tasks.result import Result


def detect_callable_mode(fn: Callable) -> Literal["state", "results"]:
    """检查 callable 第一个参数，判断 v1 还是 v2 模式。

    v2 模式条件（满足任一）：
    - 第一个参数类型注解为 State 或其子类
    - 第一个参数名为 "state"

    其他情况（包括 lambda）默认 v1 模式（传 state._results）。
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return "results"

    params = list(sig.parameters.values())
    if not params:
        return "results"

    first = params[0]

    # 尝试通过 get_type_hints 解析注解（处理 `from __future__ import annotations`）
    ann = first.annotation
    try:
        hints = typing.get_type_hints(fn)
        ann = hints.get(first.name, first.annotation)
    except Exception:
        pass

    if ann is not inspect.Parameter.empty:
        try:
            if isinstance(ann, type) and issubclass(ann, State):
                return "state"
        except TypeError:
            pass
        # 如果 get_type_hints 返回了字符串（未解析的注解），检查是否包含 "State"
        if isinstance(ann, str) and "State" in ann:
            return "state"

    # 检查参数名
    if first.name == "state":
        return "state"

    return "results"


def call_with_compat(fn: Callable, state: State) -> Any:
    """根据检测结果调用 callable，传 state 或 state._results。"""
    mode = detect_callable_mode(fn)
    if mode == "state":
        return fn(state)
    return fn(state._results)

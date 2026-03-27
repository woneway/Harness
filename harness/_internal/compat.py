"""compat.py — v1/v2 callable 双模式检测与调用。"""

from __future__ import annotations

import inspect
import logging
import typing
from typing import Any, Callable, Literal

from harness.state import State
from harness.tasks.result import Result

logger = logging.getLogger(__name__)


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


def call_with_compat(
    fn: Callable, state: State, *, in_state_pipeline: bool = False,
) -> Any:
    """根据检测结果调用 callable，传 state 或 state._results。

    Args:
        fn: 要调用的 callable。
        state: 当前 State 对象。
        in_state_pipeline: 是否在 State 模式 pipeline 中。
            为 True 且检测为 v1 模式时会输出 warning。
    """
    mode = detect_callable_mode(fn)
    if mode == "state":
        return fn(state)
    if in_state_pipeline:
        logger.warning(
            "callable 参数名不是 'state'，将以 v1 模式调用（传入 state._results 而非 state）。"
            "如在 state pipeline 中使用，请将参数名改为 'state' 或添加 State 类型注解。"
            " callable: %r",
            fn,
        )
    return fn(state._results)

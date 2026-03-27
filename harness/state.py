"""State — v2 pipeline 状态容器，替代 results: list[Result]。"""

from __future__ import annotations

import json
import logging
import types
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict

from harness.tasks.result import Result

logger = logging.getLogger(__name__)


class State(BaseModel):
    """Pipeline 执行状态。

    用户可继承 State 添加自定义字段（Pydantic model），
    pipeline 中的 Task 通过 output_key 将结果写入 state 属性。

    内部维护 v1 兼容的 _results 列表，v1 callable 仍可通过
    state._results 访问前序结果。

    Example::

        class MyState(State):
            analysis: str = ""
            score: float = 0.0

        state = MyState()
        # pipeline 执行后：
        # state.analysis = "结果文本"
        # state._results == [Result(...), ...]
    """

    model_config = ConfigDict(extra="allow")

    _results: list[Result] = []

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Pydantic v2 private attributes need manual init
        self._results = []

    def _append_result(self, result: Result) -> None:
        """追加 Result 到内部列表（框架调用）。"""
        self._results.append(result)

    def _set_output(self, key: str, value: Any) -> None:
        """设置 output_key 对应的属性值（框架调用）。

        使用 Pydantic 的 __setattr__ 以确保 model_dump() 能包含该字段。
        当 value 是 str 且目标字段是 list/dict 类型时，尝试 json.loads 自动转换。
        """
        # 自动类型转换：str → list/dict
        value = self._coerce_output_value(key, value)

        # 对于 extra="allow" 的 model，直接赋值会被 Pydantic 追踪
        self.__dict__[key] = value
        # 确保 Pydantic 也追踪到这个 extra 字段
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__ is not None:
            self.__pydantic_extra__[key] = value
        elif hasattr(self, "model_fields") and key not in self.model_fields:
            # 初始化 __pydantic_extra__ 如果还没有
            if self.__pydantic_extra__ is None:
                object.__setattr__(self, "__pydantic_extra__", {})
            self.__pydantic_extra__[key] = value  # type: ignore[index]

    def _coerce_output_value(self, key: str, value: Any) -> Any:
        """尝试将 str 值自动转换为目标字段声明的 list/dict 类型。

        仅对已声明的 model_fields 生效，extra 字段不做转换。
        转换失败时 log warning 并返回原始值。
        """
        if not isinstance(value, str) or key not in self.model_fields:
            return value

        field_info = self.model_fields[key]
        ann = field_info.annotation
        if ann is None:
            return value

        # 解析 origin 类型（如 list[dict] → list, dict[str, Any] → dict）
        origin = get_origin(ann)

        # 展开 Optional[X] / X | None → 取出非 None 的内层类型
        if origin is Union or isinstance(ann, types.UnionType):
            args = get_args(ann)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                ann = non_none[0]
                origin = get_origin(ann)

        target = origin if origin is not None else ann

        if target in (list, dict):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (list, dict)):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "output_key '%s': 值是 str 但目标字段类型为 %s，"
                    "json.loads 转换失败，将直接写入原始字符串。",
                    key, ann,
                )
        return value

    def _snapshot(self) -> dict[str, Any]:
        """返回当前 state 的可序列化快照（用于 Parallel 只读副本和日志）。"""
        data = self.model_dump()
        data["_results_count"] = len(self._results)
        return data

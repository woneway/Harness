"""State — v2 pipeline 状态容器，替代 results: list[Result]。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from harness.tasks.result import Result


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
        """
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

    def _snapshot(self) -> dict[str, Any]:
        """返回当前 state 的可序列化快照（用于 Parallel 只读副本和日志）。"""
        data = self.model_dump()
        data["_results_count"] = len(self._results)
        return data

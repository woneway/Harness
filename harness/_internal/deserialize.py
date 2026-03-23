"""deserialize.py — 续跑时 output 反序列化辅助模块。"""

from __future__ import annotations

import importlib
import json
import logging

from pydantic import BaseModel

from harness._internal.exceptions import ResumeSchemaError

logger = logging.getLogger(__name__)


def _resolve_schema_class(class_path: str) -> type[BaseModel]:
    """根据完整限定名动态导入 Pydantic 模型类。

    Args:
        class_path: 例如 "myapp.models.MyModel"

    Returns:
        Pydantic BaseModel 子类。

    Raises:
        ResumeSchemaError: 无法导入或目标不是 BaseModel 子类。
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except Exception as e:
        logger.debug("Cannot resolve schema class %r: %s", class_path, e)
        raise ResumeSchemaError(class_path) from e

    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise ResumeSchemaError(class_path)

    return cls


def deserialize_output(
    raw_output: str | None,
    schema_class_path: str | None,
) -> BaseModel | dict | str | None:
    """反序列化存储的 output。

    规则：
    1. raw_output 为 None → None
    2. schema_class_path 非空 → _resolve_schema_class()（失败则抛 ResumeSchemaError）
       → model_validate_json(raw_output)
    3. schema_class_path 为空 → json.loads(raw_output)，失败返回原始字符串

    Args:
        raw_output: 数据库中存储的 JSON 字符串（可为 None）。
        schema_class_path: Pydantic 模型的完整限定名（可为 None）。

    Returns:
        反序列化后的对象：BaseModel 实例、dict、str，或 None。

    Raises:
        ResumeSchemaError: schema_class_path 非空但类无法导入时。
    """
    if raw_output is None:
        return None

    if schema_class_path:
        cls = _resolve_schema_class(schema_class_path)  # 失败则抛 ResumeSchemaError
        return cls.model_validate_json(raw_output)

    # 没有 schema，尝试 JSON 解析（兼容旧数据），失败返回原字符串
    try:
        return json.loads(raw_output)
    except Exception:
        return raw_output

"""tests/unit/test_deserialize.py — deserialize_output 单元测试。"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from harness._internal.deserialize import _resolve_schema_class, deserialize_output


# ---------------------------------------------------------------------------
# 测试用 Pydantic 模型（必须在顶层，可被 importlib 找到）
# ---------------------------------------------------------------------------


class _SampleModel(BaseModel):
    name: str
    value: int


# ---------------------------------------------------------------------------
# _resolve_schema_class 测试
# ---------------------------------------------------------------------------


class TestResolveSchemaClass:
    def test_resolve_schema_class_valid(self) -> None:
        """可以成功导入 _SampleModel。"""
        cls = _resolve_schema_class(
            "tests.unit.test_deserialize._SampleModel"
        )
        assert cls is _SampleModel

    def test_resolve_schema_class_invalid_module(self) -> None:
        """模块不存在时返回 None，不抛异常。"""
        result = _resolve_schema_class("nonexistent.module.SomeClass")
        assert result is None

    def test_resolve_schema_class_invalid_attribute(self) -> None:
        """模块存在但属性不存在时返回 None。"""
        result = _resolve_schema_class("harness.task.NonExistentClass")
        assert result is None

    def test_resolve_schema_class_not_basemodel(self) -> None:
        """目标类不是 BaseModel 子类时返回 None。"""
        # str 不是 BaseModel 子类
        result = _resolve_schema_class("builtins.str")
        assert result is None

    def test_resolve_schema_class_empty_string(self) -> None:
        """空字符串无法分割，返回 None。"""
        result = _resolve_schema_class("")
        assert result is None

    def test_resolve_schema_class_no_dot(self) -> None:
        """没有点号无法分割模块与类名，返回 None。"""
        result = _resolve_schema_class("NoDotHere")
        assert result is None


# ---------------------------------------------------------------------------
# deserialize_output 测试
# ---------------------------------------------------------------------------


class TestDeserializeOutput:
    def test_deserialize_none_output(self) -> None:
        """raw_output 为 None 时返回 None。"""
        result = deserialize_output(None, None)
        assert result is None

    def test_deserialize_none_output_with_schema(self) -> None:
        """raw_output 为 None，即使有 schema_class_path 也返回 None。"""
        result = deserialize_output(
            None, "tests.unit.test_deserialize._SampleModel"
        )
        assert result is None

    def test_deserialize_with_valid_schema(self) -> None:
        """schema_class_path 有效时反序列化为 Pydantic 模型实例。"""
        raw = json.dumps({"name": "hello", "value": 42})
        result = deserialize_output(
            raw, "tests.unit.test_deserialize._SampleModel"
        )
        assert isinstance(result, _SampleModel)
        assert result.name == "hello"
        assert result.value == 42

    def test_deserialize_with_unavailable_schema(self) -> None:
        """schema_class_path 指向不存在的类时，fallback 为 dict。"""
        raw = json.dumps({"name": "hello", "value": 42})
        result = deserialize_output(raw, "nonexistent.module.SomeModel")
        assert isinstance(result, dict)
        assert result["name"] == "hello"
        assert result["value"] == 42

    def test_deserialize_with_unavailable_schema_invalid_json(self) -> None:
        """schema_class_path 不可用且 JSON 解析失败时返回原始字符串。"""
        raw = "not-json-string"
        result = deserialize_output(raw, "nonexistent.module.SomeModel")
        assert result == "not-json-string"

    def test_deserialize_with_no_schema_valid_json(self) -> None:
        """schema_class_path 为 None，raw_output 是合法 JSON 时返回解析结果。"""
        raw = json.dumps({"key": "val"})
        result = deserialize_output(raw, None)
        assert result == {"key": "val"}

    def test_deserialize_with_no_schema_json_list(self) -> None:
        """schema_class_path 为 None，raw_output 是 JSON 列表时返回列表。"""
        raw = json.dumps([1, 2, 3])
        result = deserialize_output(raw, None)
        assert result == [1, 2, 3]

    def test_deserialize_with_no_schema_plain_string(self) -> None:
        """schema_class_path 为 None，raw_output 不是 JSON 时返回原字符串。"""
        raw = "plain text result"
        result = deserialize_output(raw, None)
        assert result == "plain text result"

    def test_deserialize_with_no_schema_empty_string(self) -> None:
        """schema_class_path 为 None，raw_output 为空字符串时返回空字符串。"""
        result = deserialize_output("", None)
        # 空字符串不是合法 JSON，返回原始空字符串
        assert result == ""

    def test_deserialize_field_access_on_model(self) -> None:
        """反序列化后可以通过属性名访问字段（不崩溃）。"""
        raw = json.dumps({"name": "test", "value": 99})
        result = deserialize_output(
            raw, "tests.unit.test_deserialize._SampleModel"
        )
        # 核心：不应该崩溃，且字段可正常访问
        assert result.name == "test"  # type: ignore[union-attr]
        assert result.value == 99  # type: ignore[union-attr]

    def test_deserialize_schema_invalid_data(self) -> None:
        """schema 有效但数据不符合 schema 时，model_validate_json 抛异常，
        此时应该崩溃（调用方应保证数据完整性）。"""
        raw = json.dumps({"name": "hello"})  # 缺少 value 字段
        with pytest.raises(Exception):
            deserialize_output(
                raw, "tests.unit.test_deserialize._SampleModel"
            )

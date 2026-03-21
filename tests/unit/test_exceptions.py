"""tests/unit/test_exceptions.py — 异常字段验证。"""

import pytest

from harness._internal.exceptions import (
    ClaudeNotFoundError,
    InvalidPipelineError,
    OutputSchemaError,
    TaskFailedError,
)


class TestTaskFailedError:
    def test_fields_stored(self) -> None:
        err = TaskFailedError("run-1", "2.0", "llm", "timeout", partial_results=["a"])
        assert err.run_id == "run-1"
        assert err.task_index == "2.0"
        assert err.task_type == "llm"
        assert err.error == "timeout"
        assert err.partial_results == ["a"]

    def test_default_partial_results(self) -> None:
        err = TaskFailedError("run-2", "0", "function", "boom")
        assert err.partial_results == []

    def test_message_contains_info(self) -> None:
        err = TaskFailedError("run-3", "1", "shell", "exit code 1")
        assert "1" in str(err)
        assert "shell" in str(err)
        assert "run-3" in str(err)

    def test_is_exception(self) -> None:
        with pytest.raises(TaskFailedError):
            raise TaskFailedError("r", "0", "llm", "err")


class TestClaudeNotFoundError:
    def test_message(self) -> None:
        err = ClaudeNotFoundError()
        assert "Claude" in str(err)

    def test_is_exception(self) -> None:
        with pytest.raises(ClaudeNotFoundError):
            raise ClaudeNotFoundError()


class TestInvalidPipelineError:
    def test_is_exception(self) -> None:
        with pytest.raises(InvalidPipelineError):
            raise InvalidPipelineError("Nested Parallel not allowed")


class TestOutputSchemaError:
    def test_fields_stored(self) -> None:
        from pydantic import BaseModel

        class MyModel(BaseModel):
            x: int

        err = OutputSchemaError("0", MyModel, str)
        assert err.task_index == "0"
        assert err.expected_type is MyModel
        assert err.actual_type is str

    def test_message_contains_types(self) -> None:
        err = OutputSchemaError("1", int, str)
        assert "int" in str(err)
        assert "str" in str(err)

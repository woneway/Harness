"""tests/unit/test_task_index.py — TaskIndex 单元测试。"""

from __future__ import annotations

import pytest

from harness._internal.task_index import TaskIndex


# ---------------------------------------------------------------------------
# 工厂方法 + __str__（序列化）
# ---------------------------------------------------------------------------


class TestTaskIndexStr:
    def test_sequential(self) -> None:
        assert str(TaskIndex.sequential(0)) == "0"
        assert str(TaskIndex.sequential(3)) == "3"

    def test_parallel_child(self) -> None:
        assert str(TaskIndex.parallel_child(2, 0)) == "2.0"
        assert str(TaskIndex.parallel_child(2, 1)) == "2.1"
        assert str(TaskIndex.parallel_child(0, 5)) == "0.5"

    def test_dialogue_turn(self) -> None:
        assert str(TaskIndex.dialogue_turn(0, 3)) == "0.t3"
        assert str(TaskIndex.dialogue_turn(1, 0)) == "1.t0"

    def test_dialogue_round(self) -> None:
        assert str(TaskIndex.dialogue_round(0, 0, 1)) == "0.r0.1"
        assert str(TaskIndex.dialogue_round(2, 3, 0)) == "2.r3.0"


# ---------------------------------------------------------------------------
# parse（反序列化）
# ---------------------------------------------------------------------------


class TestTaskIndexParse:
    def test_parse_sequential(self) -> None:
        idx = TaskIndex.parse("0")
        assert idx.kind == "seq"
        assert idx.outer == 0
        assert not idx.is_child

    def test_parse_sequential_large(self) -> None:
        idx = TaskIndex.parse("10")
        assert idx.outer == 10
        assert idx.kind == "seq"

    def test_parse_parallel_child(self) -> None:
        idx = TaskIndex.parse("2.0")
        assert idx.kind == "par"
        assert idx.outer == 2
        assert idx.sub == 0
        assert idx.is_child

    def test_parse_parallel_child_nonzero(self) -> None:
        idx = TaskIndex.parse("2.1")
        assert idx.sub == 1

    def test_parse_dialogue_turn(self) -> None:
        idx = TaskIndex.parse("0.t3")
        assert idx.kind == "dlg_turn"
        assert idx.outer == 0
        assert idx.sub == 3
        assert idx.is_child

    def test_parse_dialogue_round(self) -> None:
        idx = TaskIndex.parse("0.r0.1")
        assert idx.kind == "dlg_round"
        assert idx.outer == 0
        assert idx.round_ == 0
        assert idx.sub == 1
        assert idx.is_child

    def test_parse_dialogue_round_large(self) -> None:
        idx = TaskIndex.parse("2.r3.0")
        assert idx.outer == 2
        assert idx.round_ == 3
        assert idx.sub == 0

    def test_parse_invalid_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            TaskIndex.parse("abc")  # outer is not an int

    def test_parse_invalid_parallel_sub_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            TaskIndex.parse("2.x")  # sub is not int and not r/t prefix


# ---------------------------------------------------------------------------
# 往返一致性（序列化 → 解析 → 再序列化 == 原始字符串）
# ---------------------------------------------------------------------------


class TestTaskIndexRoundtrip:
    @pytest.mark.parametrize("s", [
        "0", "1", "10",
        "2.0", "2.1", "0.5",
        "0.t0", "0.t3", "1.t10",
        "0.r0.0", "0.r0.1", "2.r3.0",
    ])
    def test_roundtrip(self, s: str) -> None:
        assert str(TaskIndex.parse(s)) == s


# ---------------------------------------------------------------------------
# 查询属性
# ---------------------------------------------------------------------------


class TestTaskIndexProperties:
    def test_is_child_seq(self) -> None:
        assert not TaskIndex.sequential(0).is_child

    def test_is_child_par(self) -> None:
        assert TaskIndex.parallel_child(0, 0).is_child

    def test_is_child_dlg_turn(self) -> None:
        assert TaskIndex.dialogue_turn(0, 0).is_child

    def test_is_child_dlg_round(self) -> None:
        assert TaskIndex.dialogue_round(0, 0, 0).is_child

    def test_outer_key(self) -> None:
        assert TaskIndex.parallel_child(2, 1).outer_key == "2"
        assert TaskIndex.dialogue_turn(3, 5).outer_key == "3"
        assert TaskIndex.sequential(7).outer_key == "7"

    def test_par_child_int_valid(self) -> None:
        assert TaskIndex.parallel_child(2, 1).par_child_int() == 1

    def test_par_child_int_invalid_seq(self) -> None:
        with pytest.raises(ValueError, match="not a parallel child"):
            TaskIndex.sequential(0).par_child_int()

    def test_par_child_int_invalid_dlg(self) -> None:
        with pytest.raises(ValueError, match="not a parallel child"):
            TaskIndex.dialogue_turn(0, 0).par_child_int()


# ---------------------------------------------------------------------------
# frozen dataclass：不可变性
# ---------------------------------------------------------------------------


class TestTaskIndexImmutability:
    def test_cannot_mutate(self) -> None:
        idx = TaskIndex.sequential(0)
        with pytest.raises((AttributeError, TypeError)):
            idx.outer = 1  # type: ignore[misc]

"""tests/unit/test_event_bus.py — EventBus 测试。"""

from __future__ import annotations

import asyncio

import pytest

from harness._internal.event_bus import EventBus


class TestEventBus:
    def test_creation(self) -> None:
        bus = EventBus()
        assert bus._emitter is not None

    @pytest.mark.asyncio
    async def test_on_and_emit(self) -> None:
        bus = EventBus()
        received = []

        async def handler(data):
            received.append(data)

        bus.on("test", handler)
        bus.emit("test", {"key": "value"})

        # pyee AsyncIOEventEmitter 在 event loop 中异步执行
        await asyncio.sleep(0.05)
        assert len(received) == 1
        assert received[0] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_multiple_listeners(self) -> None:
        bus = EventBus()
        r1 = []
        r2 = []

        async def h1(data):
            r1.append(data)

        async def h2(data):
            r2.append(data)

        bus.on("evt", h1)
        bus.on("evt", h2)
        bus.emit("evt", "hello")

        await asyncio.sleep(0.05)
        assert r1 == ["hello"]
        assert r2 == ["hello"]

    @pytest.mark.asyncio
    async def test_emit_no_listeners(self) -> None:
        bus = EventBus()
        # 不应抛异常
        bus.emit("nonexistent", {"data": 1})
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_emit_none_data(self) -> None:
        bus = EventBus()
        received = []

        async def handler(data):
            received.append(data)

        bus.on("test", handler)
        bus.emit("test", None)

        await asyncio.sleep(0.05)
        assert received == [None]

    @pytest.mark.asyncio
    async def test_remove_all_listeners(self) -> None:
        bus = EventBus()
        received = []

        async def handler(data):
            received.append(data)

        bus.on("test", handler)
        bus.remove_all_listeners()
        bus.emit("test", "should_not_arrive")

        await asyncio.sleep(0.05)
        assert received == []

    @pytest.mark.asyncio
    async def test_different_events_isolated(self) -> None:
        bus = EventBus()
        r_a = []
        r_b = []

        async def h_a(data):
            r_a.append(data)

        async def h_b(data):
            r_b.append(data)

        bus.on("a", h_a)
        bus.on("b", h_b)

        bus.emit("a", "only_a")
        await asyncio.sleep(0.05)

        assert r_a == ["only_a"]
        assert r_b == []

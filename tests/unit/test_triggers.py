"""tests/unit/test_triggers.py — 触发器数据类测试。"""

from __future__ import annotations

from datetime import datetime

from harness.triggers import CronTrigger, EventTrigger, Trigger, TriggerContext


class TestCronTrigger:
    def test_basic_creation(self) -> None:
        t = CronTrigger(cron="0 2 * * *")
        assert t.cron == "0 2 * * *"
        assert t.name is None

    def test_with_name(self) -> None:
        t = CronTrigger(cron="15 15 * * 1-5", name="after-close")
        assert t.name == "after-close"


class TestEventTrigger:
    def test_basic_creation(self) -> None:
        t = EventTrigger(event="price_alert")
        assert t.event == "price_alert"
        assert t.filter is None
        assert t.name is None

    def test_with_filter(self) -> None:
        f = lambda data: data["change"] > 0.03
        t = EventTrigger(event="price_alert", filter=f)
        assert t.filter is f
        assert t.filter({"change": 0.05}) is True
        assert t.filter({"change": 0.01}) is False

    def test_with_name(self) -> None:
        t = EventTrigger(event="alert", name="big-move")
        assert t.name == "big-move"


class TestTriggerUnion:
    def test_cron_is_trigger(self) -> None:
        t: Trigger = CronTrigger(cron="0 * * * *")
        assert isinstance(t, CronTrigger)

    def test_event_is_trigger(self) -> None:
        t: Trigger = EventTrigger(event="test")
        assert isinstance(t, EventTrigger)


class TestTriggerContext:
    def test_cron_context(self) -> None:
        now = datetime.now()
        ctx = TriggerContext(
            service_name="svc1",
            trigger_type="cron",
            event_name=None,
            event_data=None,
            triggered_at=now,
        )
        assert ctx.service_name == "svc1"
        assert ctx.trigger_type == "cron"
        assert ctx.event_name is None
        assert ctx.event_data is None
        assert ctx.triggered_at == now

    def test_event_context(self) -> None:
        now = datetime.now()
        data = {"symbol": "AAPL", "change": 0.05}
        ctx = TriggerContext(
            service_name="stock-svc",
            trigger_type="event",
            event_name="price_alert",
            event_data=data,
            triggered_at=now,
        )
        assert ctx.trigger_type == "event"
        assert ctx.event_name == "price_alert"
        assert ctx.event_data == data

# 设计反馈：APScheduler v4 Alpha API 差异

**日期：2026-03-21**
**来源：实现 APSchedulerBackend 时发现**

---

## 问题

PLAN.md 中写明「APScheduler v4 集成」，但 PyPI 上 APScheduler v4 仍处于 alpha 阶段（最新为 4.0.0a6），且其 API 与 v3 有重大差异：

### 具体差异

1. **`add_schedule()` 是协程**：v3 的 `add_job()` 是同步调用，v4 的 `add_schedule()` 必须 `await`
2. **Scheduler 生命周期**：v4 使用 async context manager（`async with AsyncScheduler() as scheduler`），而非 `scheduler.start()` / `scheduler.shutdown()`
3. **CronTrigger**：位置变更为 `apscheduler.triggers.cron.CronTrigger`，用法与 v3 基本相同

### 影响

`AbstractScheduler.add_job()` 设计为同步接口（`def add_job(...) -> None`），与 v4 的 async `add_schedule()` 不兼容。

---

## 解决方案（已实现）

**延迟注册模式**：`add_job()` 保持同步接口（缓存到 `_pending_jobs`），在 `start()` 中批量异步注册。

```python
def add_job(self, fn: Callable, cron: str) -> None:
    self._pending_jobs.append((fn, cron))  # 同步缓存

async def start(self) -> None:
    self._scheduler = AsyncScheduler()
    await self._scheduler.__aenter__()
    for fn, cron in self._pending_jobs:
        await self._scheduler.add_schedule(fn, CronTrigger.from_crontab(cron))  # 异步注册
```

---

## 后续建议

- 等 APScheduler v4 正式发布后重新评估 API 稳定性
- 若 v4 正式版 API 变化，只需修改 `APSchedulerBackend`，`AbstractScheduler` 接口不变
- pyproject.toml 中的 `apscheduler>=4.0.0a1` 在正式版发布后改为 `apscheduler>=4.0`

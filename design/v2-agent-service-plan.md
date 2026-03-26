# Harness v2.1 — Agent + Service 实现方案

> 只做两件事：Agent 一等公民 + Service 长驻模式。
> 基于 v2.0 底座（State/Condition/Loop/output_key，370 tests）。

---

## 设计原则

1. **Agent 是构建块，不是 PipelineStep** — 可独立使用、可降级为 Role、可组合进 pipeline
2. **Service = 触发器 + handler** — handler 返回 pipeline steps，复用现有执行引擎
3. **不引入新概念** — 不加 SharedState、不加 AgentContext、不加 Debate
4. **可选依赖** — pyee 只在使用 EventTrigger 时才需要

---

## Part 1: Agent

### 核心设计

Agent 是**角色定义对象**，不是 PipelineStep。三种使用方式：

```python
from harness import Agent, Dialogue, LLMTask

# 1. 独立使用
analyst = Agent(
    name="analyst",
    system_prompt="你是技术分析师，严谨、数据驱动。",
    runner=my_runner,
)
text = await analyst.run("分析今日 K 线走势")

# 2. 降级为 Role，参与 Dialogue
dialogue = Dialogue(
    roles=[
        analyst.as_role(lambda ctx: f"分析: {ctx.last_from('trader')}"),
        trader.as_role(lambda ctx: f"回应: {ctx.last_from('analyst')}"),
    ],
    max_rounds=3,
)

# 3. 在 pipeline 中通过 FunctionTask 使用
from harness import FunctionTask, State

class MyState(State):
    analysis: str = ""

async def analyze(state: MyState) -> str:
    return await analyst.run(f"分析数据: {state.market}")

pipeline_result = await h.pipeline([
    FunctionTask(fn=analyze, output_key="analysis"),
], state=MyState())
```

### 接口定义

```python
# harness/agent.py

@dataclass
class Agent:
    """持久化角色，对齐 CrewAI/AutoGen/ADK class-based 范式。

    与 Role 的关系：
    - Role = name + system_prompt + prompt callable（轻量，仅 Dialogue 用）
    - Agent = Role 的超集，可独立执行、有 runner、有 tools 预留
    """
    name: str
    system_prompt: str
    runner: AbstractRunner | None = None   # None 时 run() 必须指定

    # --- 预留字段（本次不实现执行逻辑，仅存储） ---
    tools: list[Any] = field(default_factory=list)

    async def run(
        self,
        prompt: str | Callable[[State], str],
        *,
        state: State | None = None,
        output_schema: type[BaseModel] | None = None,
    ) -> str:
        """单次执行：system_prompt + prompt → runner → 文本结果。

        Args:
            prompt: 用户 prompt（str 或接受 State 的 callable）。
            state: 可选 State，传给 prompt callable。
            output_schema: 结构化输出 schema（透传给 runner）。

        Returns:
            LLM 输出文本。

        Raises:
            ValueError: runner 未设置。
            TaskFailedError: runner 执行失败。
        """

    def as_role(self, prompt: Callable[["DialogueContext"], str]) -> Role:
        """降级为 Dialogue Role。

        Agent 的 system_prompt 和 runner 透传给 Role，
        用户只需提供 prompt callable。
        """
        return Role(
            name=self.name,
            system_prompt=self.system_prompt,
            prompt=prompt,
            runner=self.runner,
        )
```

### Agent.run() 实现细节

```python
async def run(self, prompt, *, state=None, output_schema=None):
    if self.runner is None:
        raise ValueError(
            f"Agent '{self.name}' has no runner. "
            "Set agent.runner or pass runner= to constructor."
        )

    # 解析 prompt
    if callable(prompt):
        if state is None:
            raise ValueError("prompt is callable but state is None")
        prompt_text = prompt(state)
    else:
        prompt_text = prompt

    # 构建 runner kwargs
    kwargs = {}
    if output_schema is not None:
        kwargs["output_schema_json"] = output_schema.model_json_schema()

    result = await self.runner.execute(
        prompt_text,
        system_prompt=self.system_prompt,
        session_id=None,
        **kwargs,
    )
    return result.text
```

**设计决策**：
- `run()` 是轻量调用，不经过 pipeline 执行引擎（无 storage/retry/session 管理）
- 需要完整特性时，用户通过 `FunctionTask(fn=agent.run, ...)` 进入 pipeline
- `tools` 字段预留，本次不实现 tool execution（等 Claude Agent SDK 稳定后做）
- 没有 `chat()` 多轮方法——session 管理的复杂度超出 Agent 的职责

### 新建文件

| 文件 | 内容 | 预估行数 |
|------|------|---------|
| `harness/agent.py` | Agent dataclass + run() + as_role() | ~80 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `harness/__init__.py` | 导出 Agent |
| `harness/tasks/types.py` | 无需修改（Agent 不是 PipelineStep） |

### 测试（~20 个）

| 文件 | 覆盖场景 |
|------|---------|
| `tests/unit/test_agent.py` | Agent 创建（必填/可选字段）、run() 成功、run() prompt callable、run() 无 runner 报错、run() callable 无 state 报错、output_schema 透传、as_role() 返回正确 Role、as_role() 在 Dialogue 中使用、tools 预留字段 |
| `tests/integration/test_agent_pipeline.py` | Agent 在 FunctionTask 中使用、Agent + Dialogue + pipeline 组合 |

---

## Part 2: Service 模式

### 核心设计

Service = **触发器列表 + handler 函数**。触发时调用 handler 获取 pipeline steps，然后走现有 `h.pipeline()` 执行。

```python
from harness import Harness, CronTrigger, EventTrigger, TriggerContext, State

h = Harness(project_path=".")

# 定义 handler：接收触发上下文，返回要执行的 pipeline steps
async def on_market_close(ctx: TriggerContext) -> list[PipelineStep]:
    return [
        FunctionTask(fn=fetch_market_data, output_key="market"),
        Dialogue(
            roles=[analyst.as_role(...), trader.as_role(...)],
            max_rounds=5,
        ),
        LLMTask("汇总讨论结论", output_key="report"),
    ]

# 注册服务
h.service(
    "stock-discussion",
    triggers=[
        CronTrigger("15 15 * * 1-5"),               # 收盘后
        EventTrigger("price_alert"),                  # 事件触发
    ],
    handler=on_market_close,
)

# 启动（阻塞，Ctrl+C 优雅退出）
await h.start()

# 外部系统推送事件
await h.emit("price_alert", {"symbol": "AAPL", "change": 0.05})
```

### 接口定义

#### 触发器

```python
# harness/triggers.py

@dataclass
class CronTrigger:
    """定时触发器，复用 APScheduler。"""
    cron: str                    # "15 15 * * 1-5"
    name: str | None = None     # 可选名称，用于日志

@dataclass
class EventTrigger:
    """事件触发器，基于 pyee。"""
    event: str                   # 事件名（如 "price_alert"）
    filter: Callable[[Any], bool] | None = None   # 可选过滤，返回 True 才触发
    name: str | None = None

Trigger = CronTrigger | EventTrigger

@dataclass
class TriggerContext:
    """handler 接收的触发上下文。"""
    service_name: str            # 所属 service
    trigger_type: str            # "cron" | "event"
    event_name: str | None       # EventTrigger 的 event 名
    event_data: Any              # 事件数据（cron 时为 None）
    triggered_at: datetime       # 触发时间
```

#### Service 注册

```python
# harness/harness.py 新增方法

ServiceHandler = Callable[
    [TriggerContext],
    Awaitable[list[PipelineStep] | PipelineStep],
]

class Harness:
    # ... 现有方法 ...

    def service(
        self,
        name: str,
        triggers: list[Trigger],
        handler: ServiceHandler,
        *,
        state_factory: Callable[[], State] | None = None,
        pipeline_name: str | None = None,
    ) -> None:
        """注册长驻服务。

        Args:
            name: 服务名称（唯一）。
            triggers: 触发器列表。
            handler: 触发时调用，接收 TriggerContext，
                     返回 PipelineStep 或 list[PipelineStep]。
            state_factory: 每次触发时创建新 State 的工厂函数。
                          None 时使用默认 State()。
            pipeline_name: pipeline 记录名，None 时用 f"{name}-{timestamp}"。
        """

    async def emit(self, event: str, data: Any = None) -> None:
        """发射事件，触发已注册的 EventTrigger。

        Args:
            event: 事件名。
            data: 事件数据，传给 TriggerContext.event_data。
        """

    # start() / stop() 扩展以支持 service
```

### 内部实现

#### EventBus — pyee 薄封装

```python
# harness/_internal/event_bus.py

class EventBus:
    """事件总线，封装 pyee AsyncIOEventEmitter。

    pyee 是可选依赖，仅在注册 EventTrigger 时导入。
    """

    def __init__(self) -> None:
        try:
            from pyee.asyncio import AsyncIOEventEmitter
        except ImportError:
            raise ImportError(
                "pyee is required for EventTrigger. "
                "Install it with: pip install 'harness[service]'"
            )
        self._emitter = AsyncIOEventEmitter()

    def on(self, event: str, handler: Callable) -> None:
        self._emitter.on(event, handler)

    def emit(self, event: str, data: Any = None) -> None:
        self._emitter.emit(event, data)

    def remove_all_listeners(self) -> None:
        self._emitter.remove_all_listeners()
```

#### ServiceRunner — 服务执行核心

```python
# harness/_internal/service.py

@dataclass
class ServiceDef:
    """服务定义（内部数据结构）。"""
    name: str
    triggers: list[Trigger]
    handler: ServiceHandler
    state_factory: Callable[[], State] | None
    pipeline_name: str | None

class ServiceRunner:
    """管理所有已注册 service 的执行。

    职责：
    1. 为 CronTrigger 注册 APScheduler 任务
    2. 为 EventTrigger 注册 EventBus 监听
    3. 触发时调用 handler → h.pipeline()
    4. 错误隔离：单次触发失败不影响服务继续运行
    """

    def __init__(self, harness: "Harness") -> None:
        self._harness = harness
        self._services: dict[str, ServiceDef] = {}
        self._event_bus: EventBus | None = None   # 懒初始化
        self._running = False

    def register(self, svc: ServiceDef) -> None:
        """注册服务定义。"""
        if svc.name in self._services:
            raise ValueError(f"Service '{svc.name}' already registered")
        self._services[svc.name] = svc

    async def start(self, scheduler: AbstractScheduler) -> None:
        """将所有 trigger 绑定到 scheduler / event_bus。"""
        has_events = False

        for svc in self._services.values():
            for trigger in svc.triggers:
                if isinstance(trigger, CronTrigger):
                    scheduler.add_job(
                        self._make_cron_handler(svc, trigger),
                        trigger.cron,
                    )
                elif isinstance(trigger, EventTrigger):
                    has_events = True

        # 懒初始化 EventBus（仅在有 EventTrigger 时）
        if has_events:
            self._event_bus = EventBus()
            for svc in self._services.values():
                for trigger in svc.triggers:
                    if isinstance(trigger, EventTrigger):
                        self._event_bus.on(
                            trigger.event,
                            self._make_event_handler(svc, trigger),
                        )

        self._running = True

    async def stop(self) -> None:
        """清理 EventBus 监听。"""
        if self._event_bus is not None:
            self._event_bus.remove_all_listeners()
        self._running = False

    async def emit(self, event: str, data: Any = None) -> None:
        """转发事件到 EventBus。"""
        if self._event_bus is None:
            return  # 没有 EventTrigger，静默忽略
        self._event_bus.emit(event, data)

    def _make_cron_handler(self, svc: ServiceDef, trigger: CronTrigger) -> Callable:
        """创建 cron 触发的 async handler。"""
        async def _handler() -> None:
            ctx = TriggerContext(
                service_name=svc.name,
                trigger_type="cron",
                event_name=None,
                event_data=None,
                triggered_at=datetime.now(),
            )
            await self._execute(svc, ctx)
        return _handler

    def _make_event_handler(self, svc: ServiceDef, trigger: EventTrigger) -> Callable:
        """创建 event 触发的 handler。"""
        async def _handler(data: Any = None) -> None:
            # 可选过滤
            if trigger.filter is not None and not trigger.filter(data):
                return
            ctx = TriggerContext(
                service_name=svc.name,
                trigger_type="event",
                event_name=trigger.event,
                event_data=data,
                triggered_at=datetime.now(),
            )
            await self._execute(svc, ctx)
        return _handler

    async def _execute(self, svc: ServiceDef, ctx: TriggerContext) -> None:
        """触发后执行 handler → pipeline。错误隔离。"""
        try:
            steps = await svc.handler(ctx)
            if isinstance(steps, BaseTask):
                steps = [steps]  # 单个 step 包装为 list

            state = svc.state_factory() if svc.state_factory else State()

            name = svc.pipeline_name or f"{svc.name}-{ctx.triggered_at:%Y%m%d-%H%M%S}"

            await self._harness.pipeline(steps, name=name, state=state)

        except Exception:
            logger.exception(
                "Service '%s' trigger execution failed", svc.name
            )
            # 不 raise — 服务继续运行
```

### Harness 修改

```python
# harness/harness.py 变更

class Harness:
    def __init__(self, ...):
        # ... 现有字段 ...
        self._service_runner: ServiceRunner | None = None

    def service(self, name, triggers, handler, *, state_factory=None, pipeline_name=None):
        if self._service_runner is None:
            from harness._internal.service import ServiceRunner
            self._service_runner = ServiceRunner(self)

        from harness._internal.service import ServiceDef
        self._service_runner.register(ServiceDef(
            name=name,
            triggers=triggers,
            handler=handler,
            state_factory=state_factory,
            pipeline_name=pipeline_name,
        ))

    async def emit(self, event: str, data: Any = None) -> None:
        if self._service_runner is not None:
            await self._service_runner.emit(event, data)

    async def start(self) -> None:
        await self._ensure_initialized()

        # 初始化 scheduler（service 的 CronTrigger 也需要）
        if self._scheduler is None and self._service_runner is not None:
            from harness.scheduler.apscheduler import APSchedulerBackend
            self._scheduler = APSchedulerBackend()

        # 注册 service triggers
        if self._service_runner is not None:
            await self._service_runner.start(self._scheduler)

        if self._scheduler is not None:
            await self._scheduler.start()

    async def stop(self) -> None:
        if self._service_runner is not None:
            await self._service_runner.stop()
        if self._scheduler is not None:
            await self._scheduler.stop()
```

### 新建文件

| 文件 | 内容 | 预估行数 |
|------|------|---------|
| `harness/triggers.py` | CronTrigger, EventTrigger, TriggerContext, Trigger 类型别名 | ~50 |
| `harness/_internal/event_bus.py` | EventBus（pyee 封装） | ~40 |
| `harness/_internal/service.py` | ServiceDef, ServiceRunner | ~120 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `harness/harness.py` | 新增 `service()`, `emit()`；扩展 `start()`, `stop()` |
| `harness/__init__.py` | 导出 CronTrigger, EventTrigger, TriggerContext |
| `pyproject.toml` | 新增 `[project.optional-dependencies] service = ["pyee>=12.0"]` |

### 测试（~35 个）

| 文件 | 覆盖场景 |
|------|---------|
| `tests/unit/test_triggers.py` | CronTrigger/EventTrigger 创建、TriggerContext 字段、Trigger union 类型 |
| `tests/unit/test_event_bus.py` | on/emit/remove、多监听者、pyee 未安装时的 ImportError |
| `tests/unit/test_service_runner.py` | register/start/stop、CronTrigger 绑定 scheduler、EventTrigger 绑定 bus、event filter 过滤、handler 返回单个 step 包装、handler 异常不中断服务、重复注册报错、emit 无 bus 时静默 |
| `tests/integration/test_service.py` | CronTrigger + EventTrigger 混合注册、emit 触发完整 pipeline 执行、state_factory 每次创建新 State、graceful stop |

---

## 总览

| 部分 | 新建文件 | 修改文件 | 新增测试 |
|------|---------|---------|---------|
| Agent | 1 | 1 | ~20 |
| Service | 3 | 3 | ~35 |
| **合计** | **4** | **4** | **~55** |

预估代码量：~290 行新代码 + ~55 测试

### 导出清单

```python
from harness import (
    # 新增
    Agent,
    CronTrigger,
    EventTrigger,
    TriggerContext,
    # 现有（不变）
    Harness, State, Condition, Loop,
    LLMTask, FunctionTask, ShellTask, PollingTask,
    Parallel, Dialogue, Role,
    ...
)
```

### 实现顺序

```
Phase 5a: Agent                    (1 文件, ~20 tests)
  └─ 无依赖，可独立实现

Phase 5b: Service (triggers + bus)  (2 文件, ~15 tests)
  └─ 无依赖，可独立实现

Phase 5c: Service (runner + 集成)   (1 文件 + 修改, ~20 tests)
  └─ 依赖 5b
```

5a 和 5b 可并行实现。

---

## 排除项

| 不做 | 理由 |
|------|------|
| SharedState | State 本身足够，wrapper 无新能力 |
| AgentContext | 直接传参更简单，无需专用数据类 |
| Debate | Dialogue + output_schema + FunctionTask 可组合实现 |
| Agent.chat() | Session 管理复杂度不属于 Agent 职责 |
| Agent 作为 PipelineStep | 保持 PipelineStep 类型简洁，Agent 通过 FunctionTask 进入 pipeline |
| Adapter Registry | 过早抽象，FunctionTask 已满足集成需求 |
| NL Planner | 风险高，LLM 链式调用质量不可控 |
| tool execution | 等 Claude Agent SDK 稳定后再做 |
| Agent.memory | 需要独立设计长期记忆存储，推迟 |

---

## 验证示例

### 最小 Agent 示例

```python
from harness import Harness, Agent
from harness.runners.openai import OpenAIRunner

analyst = Agent(
    name="analyst",
    system_prompt="你是股票技术分析师。",
    runner=OpenAIRunner(model="gpt-4o"),
)

# 独立使用
result = await analyst.run("分析贵州茅台近期走势")
print(result)
```

### Agent + Dialogue 示例

```python
from harness import Harness, Agent, Dialogue, State, FunctionTask, LLMTask

analyst = Agent(name="analyst", system_prompt="技术分析师", runner=runner)
trader = Agent(name="trader", system_prompt="短线交易员", runner=runner)

class DiscussionState(State):
    market: str = ""
    report: str = ""

h = Harness(project_path=".")
pr = await h.pipeline([
    FunctionTask(fn=fetch_data, output_key="market"),
    Dialogue(
        roles=[
            analyst.as_role(lambda ctx: f"分析行情: {ctx.state.market}"),
            trader.as_role(lambda ctx: f"回应分析: {ctx.last_from('analyst')}"),
        ],
        max_rounds=3,
    ),
    LLMTask("汇总讨论，给出最终建议", output_key="report"),
], state=DiscussionState())
```

### Service 示例

```python
from harness import (
    Harness, Agent, CronTrigger, EventTrigger,
    TriggerContext, Dialogue, FunctionTask, LLMTask, State,
)

h = Harness(project_path=".")

analyst = Agent(name="analyst", system_prompt="技术分析师", runner=runner)
trader = Agent(name="trader", system_prompt="短线交易员", runner=runner)

class MarketState(State):
    market: str = ""
    report: str = ""

async def discussion_handler(ctx: TriggerContext):
    return [
        FunctionTask(fn=fetch_data, output_key="market"),
        Dialogue(
            roles=[
                analyst.as_role(lambda c: f"分析: {c.state.market}"),
                trader.as_role(lambda c: f"回应: {c.last_from('analyst')}"),
            ],
            max_rounds=3,
        ),
        LLMTask("汇总结论", output_key="report"),
    ]

h.service(
    "stock-discussion",
    triggers=[
        CronTrigger("15 15 * * 1-5"),          # 收盘后
        EventTrigger("price_alert", filter=lambda d: abs(d["change"]) > 0.03),
    ],
    handler=discussion_handler,
    state_factory=MarketState,
)

# 启动长驻运行
await h.start()

# 外部系统可触发
await h.emit("price_alert", {"symbol": "600519", "change": 0.05})
```

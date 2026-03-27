# Harness v2 全量实现计划

> v2.0 底座已完成（2026-03-25）：State、Condition/Loop、output_key、tasks/ 拆分、v1/v2 双模式兼容，370 tests。
> 分支：`v2.0`，4 commits。
> 本计划覆盖剩余全部 v2 功能。

## Context

用户需要 Harness 框架级支持"讨论群"（辩论团）场景。核心需求：

1. **Agent = 工厂函数模式**：`create_agent(llm)` → `agent_node(state)` → 更新 state
   - 类比 TradingAgents：`def create_agent(llm): def agent_node(state): return {"field": value}; return agent_node`
2. **ctx.shared 共享状态**：agent 之间通过 `ctx.shared["key"]` 传递数据
3. **sub-state 空间**：每个 Agent 有独立 state 空间，`state.{agent_name}` 下
4. **内嵌投票机制**：每轮每个 agent 发言后自动在 sub-state 写入 vote/reason，框架聚合
5. **Debate = PipelineStep**：Debate 在 pipeline 中和其他 task 组合
6. **vote 结果写回 Pipeline State**：`state.debate_result = DebateResult(votes, consensus, summary)`

---

## 核心抽象设计

### 1. SharedState — Agent 间共享状态

```python
class SharedState:
    """ctx.shared 底座：字典语义，跨 Agent 共享。

    实现：包装 Pipeline State 的一个顶层 dict 字段（如 state._shared），
    对外暴露 dict-like 接口。
    """
    def __init__(self, state: "PipelineState"): ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def keys(self) -> Iterable[str]: ...
    def to_dict(self) -> dict: ...  # 快照用于 LLM prompt 注入
```

**设计决策**：`SharedState` 不是独立对象，是 Pipeline State 的一个 view。Agent 看到的 `ctx.shared` 就是 Pipeline State 的 `_shared` 字段。这样 Debate 结果直接留在 State 里，无需额外序列化。

### 2. Agent — 工厂函数模式

```python
# 工厂函数：接受 runner，返回 node 函数
def create_agent(
    name: str,
    system_prompt: str,
    runner: AbstractRunner,
    *,
    sub_state_schema: type[BaseModel] | None = None,
    tools: list[Tool] | None = None,
    vote_schema: type[BaseModel] | None = None,
) -> Callable[["AgentContext"], None]:
    """返回 agent node 函数。"""
    def agent_node(ctx: "AgentContext") -> None:
        # 1. 构建 prompt（system_prompt + ctx.history + ctx.shared）
        # 2. 调用 runner.execute(prompt, tools=tools)
        # 3. 将结果写入 ctx.agent_state（sub-state）
        # 4. 如果 vote_schema: 在 ctx.agent_state 中写入 vote 字段
        return  # 无返回值，状态直接写在 ctx.agent_state
    return agent_node
```

**AgentContext**（传给 node 函数的上下文）：

```python
@dataclass
class AgentContext:
    shared: SharedState          # ctx.shared，字典语义，跨 Agent 共享
    agent_state: BaseModel       # 该 Agent 的 sub-state（由 sub_state_schema 管理）
    history: list[dict]         # 该 Agent 的发言历史（message + round）
    round: int                  # 当前轮次
    # vote 自动写入 agent_state（由 vote_schema 定义字段）
```

**设计决策：工厂函数 vs 数据类**

- 不做数据类：工厂函数模式更灵活，和 TradingAgents 对齐
- `sub_state_schema` 控制该 Agent 的 state 结构（如 `AnalystState`）
- `vote_schema` 定义投票字段（如 `VoteOutput`），框架从 `agent_state` 读取
- Agent node **无返回值**：所有状态通过 `ctx.agent_state` 和 `ctx.shared` 写入，框架负责分发

### 3. Vote 机制

```python
# vote_schema 示例
class AnalystVoteOutput(BaseModel):
    vote: bool                        # True = 同意/通过，False = 反对
    reason: str                       # 投票理由
    confidence: float                  # 置信度 0-1

# agent_node 中投票的写入方式（agent_state 是 BaseModel 实例）
def analyst_node(ctx: AgentContext):
    response = ctx.runner.execute(...)
    ctx.agent_state.message = response  # 发言内容

    # 内嵌投票：框架自动从 agent_state 提取 vote 字段
    # （agent 可以手动写，也可以 LLM structured output 直接写）
    # 简化做法：agent_state.vote = VoteOutput(...)
    ctx.agent_state.vote = AnalystVoteOutput(
        vote=True, reason="趋势明确", confidence=0.85
    )
```

**Vote 聚合**（Debate 引擎内部）：

```python
@dataclass
class VoteResult:
    agent_name: str
    vote: bool
    reason: str
    confidence: float
    message: str

@dataclass
class DebateResult:
    votes: list[VoteResult]       # 每轮各 agent 的投票
    consensus: bool               # 多数一致？
    summary: str                 # 框架汇总的最终结论
    rounds_completed: int
```

### 4. Debate — PipelineStep

```python
@dataclass
class Debate(PipelineStep):
    """辩论团：多 Agent 多轮讨论 + 内嵌投票。

    类比 TradingAgents 的 debate 循环 + LangGraph 的 graph 模式。
    """
    agents: list[AgentNode]      # Agent node 函数列表（不是数据类）
    shared_schema: type[BaseModel] | None = None  # shared state 的结构
    max_rounds: int = 5
    vote_threshold: float = 0.5  # 超过此比例则结束
    output_key: str | None = None

    async def execute(self, state: State, ctx: SharedContext) -> DebateResult: ...
```

**SharedContext**（Debate 执行时传入的上下文）：

```python
@dataclass
class SharedContext:
    shared: SharedState
    runner: AbstractRunner
    storage: StorageProtocol | None = None
```

**Debate 执行流程**：

```
for round in range(max_rounds):
    round_votes = []

    for agent in agents:
        # 准备 AgentContext
        ctx = AgentContext(
            shared=self.shared,
            agent_state=agent.sub_state or empty_model(),
            history=agent.history,
            round=round,
        )

        # 执行 agent node
        await agent.node(ctx)

        # 提取 vote（从 agent_state）
        if hasattr(ctx.agent_state, 'vote'):
            round_votes.append(VoteResult(
                agent_name=agent.name,
                vote=ctx.agent_state.vote.vote,
                reason=ctx.agent_state.vote.reason,
                confidence=ctx.agent_state.vote.confidence,
                message=ctx.agent_state.message,
            ))

    # 投票聚合
    all_votes.extend(round_votes)
    agree_count = sum(1 for v in round_votes if v.vote)
    if agree_count / len(round_votes) >= vote_threshold:
        break

# 写入最终结果
debate_result = DebateResult(
    votes=all_votes,
    consensus=agree_count / len(all_votes) >= vote_threshold,
    summary=summarize_debate(all_votes),
    rounds_completed=round + 1,
)

if self.output_key:
    state._set_output(self.output_key, debate_result)
else:
    state.debate_result = debate_result

return debate_result
```

### 5. 状态层次结构

```
Pipeline State
├── _shared: dict                  # ctx.shared 的底座（SharedState 视图）
├── analyst: AnalystState           # Agent sub-state（由 sub_state_schema 管理）
│   ├── message: str
│   ├── vote: AnalystVoteOutput
│   └── history: list[dict]
├── trader: TraderState            # Agent sub-state
│   ├── message: str
│   ├── vote: TraderVoteOutput
│   └── history: list[dict]
└── debate_result: DebateResult    # 顶层输出（Debate 执行后写入）
```

---

## Phase 5: SharedState + AgentFactory

**目标**：引入 SharedState（ctx.shared）和 Agent 工厂函数模式。

### 新建文件

| 文件 | 内容 |
|------|------|
| `harness/_internal/agent/shared_state.py` | SharedState — dict-like view over Pipeline State |
| `harness/_internal/agent/context.py` | AgentContext — 传给 agent node 的上下文 |
| `harness/_internal/agent/factory.py` | `create_agent()` 工厂函数 |
| `harness/agent/__init__.py` | 导出 `create_agent`, `AgentContext`, `SharedState` |

### 核心 API

```python
# harness/agent/__init__.py
def create_agent(
    name: str,
    system_prompt: str,
    runner: AbstractRunner,
    *,
    sub_state_schema: type[BaseModel] | None = None,
    tools: list[Tool] | None = None,
    vote_schema: type[BaseModel] | None = None,
) -> Callable[["AgentContext"], None]:
    """工厂函数，返回 agent node 函数。"""
    ...

@dataclass
class AgentContext:
    shared: SharedState
    agent_state: BaseModel
    history: list[dict]
    round: int
```

### 修改文件

| 文件 | 变更 |
|------|------|
| `harness/__init__.py` | 导出 create_agent, SharedState, AgentContext |

### 测试（~30 个）

- `tests/unit/test_shared_state.py`：get/set/keys/snapshot、跨 State 隔离
- `tests/unit/test_agent_factory.py`：create_agent、生成的 node 函数行为、sub_state_schema、vote_schema

---

## Phase 6: Debate PipelineStep

**目标**：`Debate` 作为 PipelineStep，支持多 Agent 多轮 + 内嵌投票。

### 新建文件

| 文件 | 内容 |
|------|------|
| `harness/_internal/agent/debate_engine.py` | Debate 执行引擎：轮次循环 + 投票聚合 |
| `harness/tasks/debate.py` | Debate PipelineStep 数据类 |
| `tests/unit/test_debate.py` | Debate 单元测试 |
| `tests/integration/test_debate.py` | Debate 集成测试 |

### 核心 API

```python
# harness/tasks/debate.py
@dataclass
class VoteResult:
    agent_name: str
    vote: bool
    reason: str
    confidence: float
    message: str

@dataclass
class DebateResult:
    votes: list[VoteResult]
    consensus: bool
    summary: str
    rounds_completed: int

@dataclass
class Debate(PipelineStep):
    agents: list[Callable[[AgentContext], None]]
    shared_schema: type[BaseModel] | None = None
    max_rounds: int = 5
    vote_threshold: float = 0.5
    output_key: str | None = None
```

### 修改文件

| 文件 | 变更 |
|------|------|
| `harness/tasks/types.py` | `PipelineStep` union 加入 `Debate` |
| `harness/__init__.py` | 导出 Debate, VoteResult, DebateResult |
| `harness/harness.py` | `pipeline()` 支持 Debate 类型 |

### 测试（~35 个）

- 投票聚合逻辑（多数同意提前结束、等票、超时）
- vote_threshold 边界
- Agent 历史累积
- debate_result 写入 state

---

## Phase 7: Service 模式 + EventBus

**目标**：`h.service()` 长驻运行 + Cron/Event 触发。

### 新建文件

| 文件 | 内容 |
|------|------|
| `harness/_internal/event_bus.py` | EventBus — emit/on/off |
| `harness/triggers.py` | Cron + Event 触发器 |
| `harness/_internal/service.py` | ServiceRunner |

### 修改文件

| 文件 | 变更 |
|------|------|
| `harness/harness.py` | 新增 `service()`、`start()`/`stop()` 扩展 |
| `harness/__init__.py` | 导出 Cron, Event |

### 测试（~35 个）

- EventBus emit/on/off、多 handler
- Cron trigger、Cron + Event 混合
- Service shutdown

---

## Phase 8: Adapter Registry + 首批 Adapter

### 新建文件

| 文件 | 内容 |
|------|------|
| `harness/adapters/__init__.py` | re-export + register_adapter |
| `harness/adapters/base.py` | Adapter ABC + Registry |
| `harness/adapters/stock_data/akshare.py` | StockKline, StockRealtimeQuote |
| `harness/adapters/stock_data/__init__.py` | 注册 adapters |
| `harness/adapters/minimax/tts.py` | MinimaxTTS |
| `harness/adapters/minimax/__init__.py` | 注册 adapters |
| `harness/adapters/trending/weibo.py` | WeiboTrending |
| `harness/adapters/trending/__init__.py` | 注册 adapters |

### 测试（~30 个）

---

## Phase 9: ClaudeRunner (Agent SDK)

### 新建文件

| 文件 | 内容 |
|------|------|
| `harness/runners/claude.py` | ClaudeRunner |

### 测试（~25 个）

---

## Phase 10: NL Planner + CLI

### 新建文件

| 文件 | 内容 |
|------|------|
| `harness/planner/models.py` | ExecutionPlan, PlanStep |
| `harness/planner/planner.py` | Planner |
| `harness/planner/interaction.py` | PlanInteraction |
| `harness/planner/__init__.py` | re-export |

### 测试（~25 个）

---

## 总览

| Phase | 范围 | 新建文件 | 修改文件 | 新增测试 |
|-------|------|---------|---------|---------|
| 5 | SharedState + AgentFactory | 4 | 1 | ~30 |
| 6 | Debate PipelineStep | 4 | 3 | ~35 |
| 7 | Service + EventBus | 3 | 2 | ~35 |
| 8 | Adapter Registry + 首批 | 8 | 2 | ~30 |
| 9 | ClaudeRunner | 1 | 2 | ~25 |
| 10 | NL Planner | 4 | 2 | ~25 |
| **合计** | | **24** | **12** | **~180** |

## 验证

```python
# 讨论群最小示例（Phase 6 后）
from harness import Harness, State, Debate, create_agent, SharedState
from harness.tasks import ShellTask
from pydantic import BaseModel

class MarketData(BaseModel):
    data: str = ""

class AnalystVote(BaseModel):
    vote: bool
    reason: str
    confidence: float

class AnalystState(BaseModel):
    message: str = ""
    vote: AnalystVote | None = None
    history: list = []

analyst = create_agent(
    name="analyst",
    system_prompt="你是技术分析师...",
    runner=my_runner,
    sub_state_schema=AnalystState,
    vote_schema=AnalystVote,
)

trader = create_agent(
    name="trader",
    system_prompt="你是短线交易员...",
    runner=my_runner,
    sub_state_schema=TraderState,
    vote_schema=TraderVote,
)

# Pipeline 中使用 Debate
pr = await h.pipeline([
    ShellTask(cmd="fetch_market_data", output_key="market"),
    Debate(
        agents=[analyst, trader],
        shared_schema=MarketData,
        max_rounds=5,
        vote_threshold=0.5,
        output_key="debate_result",
    ),
])
```

## 导出清单

```python
# Phase 5
create_agent, SharedState, AgentContext

# Phase 6
Debate, VoteResult, DebateResult

# Phase 7
Cron, Event

# Phase 8
Adapter, register_adapter

# Phase 9
ClaudeRunner

# Phase 10
Planner
```

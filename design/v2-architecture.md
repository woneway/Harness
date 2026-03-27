# Harness v2 架构设计

> 程序员的 Dify — NL 驱动 + 代码级灵活 + 插件生态 + 生产就绪

## 设计原则

1. **工程化**：每个步骤可定义、可配置、可序列化
2. **可控**：用户清楚每一步在做什么，可以干预
3. **可复现**：相同输入 → 相同输出，所有参数可追溯
4. **可观测**：运行过程实时可见，结果有日志、有存储
5. **拒绝黑盒**：每轮 LLM 调用的输入/输出/推理过程都透明记录

---

## 1. 公开 API

### 1.1 NL 入口（主要接口）

```bash
# 命令行
$ harness run "基于今日微博热搜，生成一个60秒AI短视频"
$ harness run "建一个A股讨论群，4个角色，每天收盘后讨论"

# Python
from harness import Harness

h = Harness(project_path=".")
result = await h.natural("基于今日微博热搜，生成一个60秒AI短视频")
```

交互流程：
```
harness run "..."
  → Planner 分析需求
  → 检查 Adapter 可用性
  → 展示执行计划
  → 用户确认 (Y) / 生成项目 (g) / 修改建议 (e) / 取消 (n)
  → 全自动执行
```

### 1.2 代码入口（精细控制）

```python
from harness import (
    Harness, Flow, State,
    LLMTask, FunctionTask, ShellTask, PollingTask,
    Parallel, Dialogue, Role,
    Agent,
    Condition, Loop,
    TaskConfig, Memory,
)
from harness.adapters import minimax, stock_data

# ---------- 简单 pipeline（v1 兼容风格）----------
result = await h.pipeline([
    FunctionTask(fn=fetch_data),
    LLMTask("分析数据"),
    ShellTask(cmd="ffmpeg -i input.mp4 output.mp4"),
])

# ---------- 带条件和循环 ----------
result = await h.pipeline([
    FunctionTask(fn=fetch_data),
    LLMTask("分析数据质量", output_schema=QualityScore),
    Condition(
        check=lambda state: state["quality_score"].score < 0.8,
        if_true=[LLMTask("重新分析，更仔细")],
        if_false=[LLMTask("生成最终报告")],
    ),
])

# ---------- Flow 风格（复杂工作流）----------
class VideoState(State):
    topic: str = ""
    script: str = ""
    audio_url: str = ""
    video_url: str = ""
    output_path: str = ""

@h.flow(state_type=VideoState)
async def video_pipeline(state):
    # Step 1: 获取热点
    state.topic = await h.call(FunctionTask(fn=fetch_trending))

    # Step 2: 写脚本
    state.script = await h.call(LLMTask(f"为{state.topic}写60秒视频脚本"))

    # Step 3: 并行生成音频和视频
    state.audio_url, state.video_url = await h.parallel(
        h.call(minimax.tts(text=state.script, voice="male-qn")),
        h.call(minimax.video(prompt=state.script)),
    )

    # Step 4: 合成
    state.output_path = await h.call(
        ShellTask(cmd=f"ffmpeg -i {state.audio_url} -i {state.video_url} output.mp4")
    )

result = await video_pipeline()

# ---------- Agent 定义 ----------
analyst = Agent(
    name="技术分析师",
    role="分析K线形态、均线系统、MACD指标",
    personality="严谨、数据驱动、不做主观判断",
    tools=[stock_data.kline, stock_data.realtime_quote],
    runner=ClaudeRunner(),
)

# ---------- 讨论群（Service 模式）----------
h.service(
    name="stock-discussion",
    agents=[analyst, fundamental, trader, risk_mgr],
    triggers=[
        Cron("15 15 * * 1-5"),                          # 收盘后
        Event(stock_data.price_change, threshold=0.03),  # 涨跌超3%
    ],
    on_trigger=lambda data: Dialogue(
        background=f"今日行情数据:\n{data}",
        roles=[...],
        max_turns=8,
    ),
)
await h.start()  # 长驻运行
```

### 1.3 导出清单

```python
from harness import (
    # 核心
    Harness,

    # 任务类型
    LLMTask, FunctionTask, ShellTask, PollingTask,
    Parallel, Dialogue, Role,
    Condition, Loop,                    # v2 新增

    # Agent
    Agent,                              # v2 新增

    # Flow
    Flow, State,                        # v2 新增

    # 数据
    Result, PipelineResult,
    TaskConfig, Memory,

    # Runner
    AbstractRunner, RunnerResult,
    ClaudeRunner,                       # v2: 基于 Agent SDK（替换 ClaudeCliRunner）
    OpenAIRunner,
    AnthropicRunner,

    # Service 模式
    Cron, Event,                        # v2 新增

    # Registry
    adapter, register_adapter,          # v2 新增
)
```

---

## 2. 核心抽象

### 2.1 State — 共享状态

替代 v1 的 `results: list[Result]`，借鉴 LangGraph TypedDict + ADK output_key。

```python
from pydantic import BaseModel

class State(BaseModel):
    """Pipeline 执行期间的共享状态。

    所有任务可读可写，替代 results[N] 的隐式传递。
    自动持久化到 Storage，支持 resume。
    """
    class Config:
        # 允许任意字段
        extra = "allow"

# 用户自定义状态
class MarketState(State):
    date: str = ""
    market_data: str = ""
    analysis: str = ""
    report: str = ""
    quality_score: float = 0.0

# 任务通过 state 读写数据
def fetch_data(state: State) -> None:
    state.market_data = call_api()

# LLMTask 可以通过 output_key 自动写入 state
LLMTask(
    prompt=lambda state: f"分析: {state.market_data}",
    output_key="analysis",  # 结果自动写入 state.analysis
)
```

**关键设计决策**：
- State 是 Pydantic BaseModel，类型安全 + 可序列化
- 任务函数签名从 `fn(results: list) -> T` 变为 `fn(state: State) -> None`（直接写 state）
- LLMTask 的 `output_key` 自动将 LLM 输出写入 state 对应字段
- State 每步执行后自动快照到 Storage，支持任意步骤 resume

### 2.2 Agent — 持久化角色

借鉴 CrewAI 的角色三元组 + Claude Agent SDK 的 AgentDefinition。

```python
@dataclass
class Agent:
    name: str                           # 唯一标识
    role: str                           # 角色定义
    personality: str = ""               # 性格特征
    system_prompt: str = ""             # 完整 system prompt（优先于 role/personality 自动生成）
    tools: list[Tool] = field(default_factory=list)  # 可用工具
    runner: AbstractRunner | None = None  # LLM 后端
    memory: AgentMemory | None = None   # 长期记忆

    async def run(self, prompt: str, *, state: State | None = None) -> str:
        """单次执行：注入角色上下文 → 调用 runner → 返回结果"""
        ...

    async def chat(self, message: str) -> str:
        """多轮对话：维护 session"""
        ...
```

**与 v1 Dialogue Role 的关系**：
- v1 `Role` = name + system_prompt + prompt callable（轻量，仅在 Dialogue 中使用）
- v2 `Agent` = Role 的超集，可独立使用、有工具、有记忆

### 2.3 Condition / Loop — 流程控制

```python
@dataclass
class Condition:
    """条件分支。借鉴 LangGraph conditional_edges + Harness CI when。"""
    check: Callable[[State], bool]
    if_true: list[PipelineStep]
    if_false: list[PipelineStep] = field(default_factory=list)

@dataclass
class Loop:
    """循环执行。借鉴 ADK LoopAgent。"""
    body: list[PipelineStep]
    until: Callable[[State], bool]    # True 时退出循环
    max_iterations: int = 10
```

Pipeline 中使用：
```python
await h.pipeline([
    FunctionTask(fn=fetch_data),
    Loop(
        body=[
            LLMTask("分析数据", output_key="analysis"),
            LLMTask("评估分析质量", output_schema=QualityScore, output_key="quality"),
        ],
        until=lambda state: state.quality.score >= 0.8,
        max_iterations=3,
    ),
    LLMTask("基于分析生成最终报告", output_key="report"),
])
```

### 2.4 Adapter — 集成插件

借鉴 lm-evaluation-harness 的 Registry 模式。

```python
from harness.registry import register_adapter, Adapter

@register_adapter("minimax-tts")
class MinimaxTTS(Adapter):
    """MiniMax TTS 服务集成"""

    name = "minimax-tts"
    description = "MiniMax 语音合成"
    capabilities = ["tts", "voice-clone"]
    config_schema = MinimaxConfig  # Pydantic model for API key etc.

    async def execute(self, *, text: str, voice: str = "male-qn", **kwargs) -> str:
        """调用 TTS API，返回音频 URL"""
        ...

    def as_task(self, **kwargs) -> PollingTask:
        """转换为 PollingTask，用于 pipeline"""
        return PollingTask(
            submit_fn=self._submit,
            poll_fn=self._poll,
            success_condition=lambda r: r["status"] == "done",
            **kwargs,
        )

    def as_tool(self) -> Tool:
        """转换为 Agent Tool，用于 Agent.tools"""
        return Tool(
            name=self.name,
            description=self.description,
            fn=self.execute,
        )
```

CLI 管理：
```bash
$ harness adapter add minimax-tts        # 安装
$ harness adapter list                   # 列出已安装
$ harness adapter remove minimax-tts     # 卸载
$ harness adapter info minimax-tts       # 查看详情
```

**Adapter 目录结构**：
```
harness/adapters/
  __init__.py
  base.py              # Adapter 基类 + Registry
  minimax/
    __init__.py         # register_adapter("minimax-tts"), register_adapter("minimax-video")
    tts.py
    video.py
    config.py
  stock_data/
    __init__.py
    akshare.py          # akshare 数据源
    config.py
```

### 2.5 NL Planner — 自然语言规划器

```python
class Planner:
    """将自然语言需求转换为可执行计划。"""

    async def plan(self, request: str) -> ExecutionPlan:
        """
        1. 分析用户意图
        2. 检查已安装的 Adapter
        3. 拆解为步骤
        4. 返回结构化计划
        """
        ...

    async def refine(self, plan: ExecutionPlan, feedback: str) -> ExecutionPlan:
        """根据用户修改建议调整计划"""
        ...

    def to_pipeline(self, plan: ExecutionPlan) -> list[PipelineStep]:
        """将计划转换为可执行的 pipeline steps"""
        ...

    def to_project(self, plan: ExecutionPlan, output_dir: Path) -> Path:
        """将计划生成为独立的 Python 项目"""
        ...

@dataclass
class ExecutionPlan:
    """结构化执行计划 — 可序列化、可展示、可修改"""
    name: str
    description: str
    mode: Literal["pipeline", "service"]
    steps: list[PlanStep]
    required_adapters: list[str]
    missing_adapters: list[str]
    estimated_duration: str | None = None

@dataclass
class PlanStep:
    index: int
    description: str
    task_type: str              # "llm" | "function" | "shell" | "polling" | "parallel" | "dialogue"
    adapter: str | None = None  # 使用的 adapter
    config: dict = field(default_factory=dict)
```

Planner 的 LLM 调用：
```python
# Planner 自己也通过 Runner 调用 LLM
# system prompt 包含：
# 1. 所有已安装 Adapter 的元信息（name, description, capabilities）
# 2. 可用的任务类型和组合规则
# 3. 输出格式要求（ExecutionPlan JSON schema）
# output_schema = ExecutionPlan 保证结构化输出
```

### 2.6 EventBus — 事件触发

支持 Service 模式的事件驱动。借鉴 Temporal Signal + Prefect 事件引擎。

```python
class EventBus:
    """事件总线 — 连接外部事件源和内部处理器"""

    def on(self, event_type: str, handler: Callable) -> None:
        """注册事件处理器"""
        ...

    async def emit(self, event_type: str, data: Any) -> None:
        """发射事件"""
        ...

# 使用方式
class Cron:
    """定时触发器"""
    def __init__(self, expr: str):
        self.expr = expr  # "15 15 * * 1-5"

class Event:
    """事件触发器"""
    def __init__(self, source: Callable, *, threshold: float | None = None, **kwargs):
        self.source = source
        self.threshold = threshold
```

---

## 3. 执行模型

### 3.1 Pipeline 模式（一次性）

```
用户 → Harness.pipeline(steps) 或 Harness.natural("...")
  │
  ├─ 初始化 State（空或用户提供）
  ├─ 初始化 Storage（创建 run 记录）
  │
  ├─ for step in steps:
  │     ├─ Condition? → 评估 check(state) → 选择分支
  │     ├─ Loop? → 循环执行 body 直到 until(state) 或 max_iterations
  │     ├─ Parallel? → asyncio.gather 并发执行
  │     ├─ LLMTask? → Runner.execute() → output_key 写入 state
  │     ├─ FunctionTask? → fn(state) → 直接修改 state
  │     ├─ ShellTask? → subprocess → stdout 写入 state
  │     ├─ PollingTask? → submit + poll 循环
  │     └─ Dialogue? → 多角色讨论
  │     │
  │     ├─ 快照 state 到 Storage（可 resume）
  │     ├─ 记录 task_log（输入/输出/耗时/tokens）
  │     └─ 触发 progress_callback
  │
  ├─ 成功 → Notifier 通知
  └─ 失败 → 根据 failure_strategy 处理 → Notifier 通知
```

### 3.2 Service 模式（长驻）

```
用户 → Harness.service(name, agents, triggers, on_trigger)
  │
  ├─ 注册 triggers:
  │     ├─ Cron → APScheduler 定时任务
  │     └─ Event → EventBus 事件监听
  │
  ├─ Harness.start() → 进入事件循环
  │
  ├─ 触发时:
  │     ├─ 收集触发数据（行情、事件内容等）
  │     ├─ 调用 on_trigger(data) → 返回 PipelineStep（通常是 Dialogue）
  │     ├─ 执行 pipeline（同 3.1）
  │     ├─ 记录到 Storage
  │     └─ Notifier 通知
  │
  └─ Harness.stop() → 优雅退出
```

### 3.3 State 快照与 Resume

```python
# 每步执行后，state 快照到 Storage
# resume 时：
# 1. 从 Storage 恢复 state 快照
# 2. 跳过已完成的步骤
# 3. 从失败步骤继续执行

result = await h.pipeline([...])
# 如果失败:
result = await h.pipeline([...], resume_from=run_id)
# state 自动恢复到失败前的状态
```

### 3.4 失败策略

借鉴 Harness CI 的组合式策略：

```python
@dataclass
class FailureStrategy:
    retries: int = 2
    backoff_base: float = 2.0
    on_retry_exhausted: Literal["abort", "skip", "fallback"] = "abort"
    fallback: PipelineStep | None = None  # on_retry_exhausted="fallback" 时使用

# 使用
LLMTask(
    prompt="...",
    failure_strategy=FailureStrategy(
        retries=3,
        on_retry_exhausted="fallback",
        fallback=LLMTask("用更简单的方式分析"),
    ),
)
```

---

## 4. Runner 层

### 4.1 ClaudeRunner（基于 Agent SDK）

替换 v1 的 ClaudeCliRunner：

```python
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

class ClaudeRunner(AbstractRunner):
    """基于 Claude Agent SDK 的 Runner。

    替代 v1 的 CLI 子进程方式，获得：
    - 自定义 MCP 工具（@tool 装饰器）
    - Subagent 支持
    - 预算控制（max_turns, max_budget_usd）
    - 结构化输出
    - Hooks（PreToolUse, PostToolUse）
    - Session 管理（continue/resume/fork）
    """

    async def execute(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        session_id: str | None = None,
        output_schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> RunnerResult:
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            resume=session_id,
            output_format=self._schema_to_format(output_schema),
            mcp_servers=self._tools_to_mcp(tools),
            permission_mode="bypassPermissions",
            max_turns=kwargs.get("max_turns"),
            max_budget_usd=kwargs.get("max_budget_usd"),
        )

        text = ""
        result_session_id = None
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                result_session_id = message.session_id
                text = message.result or text

        return RunnerResult(
            text=text,
            session_id=result_session_id,
            tokens_used=...,
        )
```

### 4.2 Runner 接口（保持简洁）

借鉴 lm-evaluation-harness 的三操作模型 — Runner 接口尽量简单：

```python
class AbstractRunner(ABC):
    @abstractmethod
    async def execute(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        session_id: str | None = None,
        output_schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
        stream_callback: Callable[[str], None] | None = None,
        **kwargs,
    ) -> RunnerResult:
        """执行一次 LLM 调用。所有 Runner 的唯一接口。"""
        ...
```

OpenAIRunner 和 AnthropicRunner 保持不变。

---

## 5. 包结构

```
harness/
  __init__.py              # 公开 API 导出
  harness.py               # Harness 主类（pipeline / natural / service / start / stop）
  state.py                 # State 基类
  agent.py                 # Agent 定义

  tasks/
    __init__.py            # 导出所有任务类型
    base.py                # PipelineStep 协议
    llm.py                 # LLMTask
    function.py            # FunctionTask
    shell.py               # ShellTask
    polling.py             # PollingTask
    parallel.py            # Parallel
    dialogue.py            # Dialogue + Role
    condition.py           # Condition（v2 新增）
    loop.py                # Loop（v2 新增）

  runners/
    base.py                # AbstractRunner + RunnerResult
    claude.py              # ClaudeRunner（基于 Agent SDK）
    openai.py              # OpenAIRunner
    anthropic.py           # AnthropicRunner

  adapters/
    base.py                # Adapter 基类 + Registry + @register_adapter
    minimax/               # MiniMax TTS + Video
    stock_data/            # 股票数据（akshare）
    trending/              # 热搜数据

  planner/
    __init__.py
    planner.py             # Planner 核心逻辑
    models.py              # ExecutionPlan + PlanStep
    interaction.py         # 用户确认/修改交互

  engine/
    executor.py            # 任务派发 + 重试 + state 管理
    session.py             # SessionManager
    event_bus.py           # EventBus（v2 新增）

  storage/
    base.py                # StorageProtocol
    sql.py                 # SQLAlchemy async
    models.py              # ORM 模型

  scheduler/
    base.py                # AbstractScheduler
    apscheduler.py         # APScheduler v4

  notifier/
    base.py                # AbstractNotifier
    telegram.py            # TelegramNotifier

  memory/
    base.py                # Memory 基础（v1 memory.py 拆分）
    agent_memory.py        # Agent 长期记忆（v2 新增）

  cli.py                   # harness run / adapter / runs / migrate
  exceptions.py            # 所有异常类型
```

---

## 6. 可观测性设计

### 6.1 结构化日志

每次任务执行自动记录：

```python
@dataclass
class TaskLog:
    run_id: str
    task_index: str
    task_type: str
    # 输入
    prompt: str | None           # LLMTask 的 prompt
    state_snapshot_before: dict  # 执行前的 state
    # 输出
    output: Any
    state_snapshot_after: dict   # 执行后的 state
    raw_text: str | None         # LLM 原始输出
    # 性能
    tokens_used: int
    duration_seconds: float
    retry_count: int
    # 状态
    success: bool
    error: str | None
```

### 6.2 Progress 回调

```python
@dataclass
class ProgressEvent:
    event: Literal["plan_ready", "step_start", "step_complete", "step_error", "confirm_required"]
    step_index: str
    step_type: str
    step_description: str
    content: str | None = None
    state_snapshot: dict | None = None

# 使用
result = await h.pipeline(
    [...],
    progress_callback=lambda e: print(f"[{e.event}] {e.step_description}"),
)
```

### 6.3 CLI 查看

```bash
$ harness runs                         # 列出运行记录
$ harness runs <run_id>                # 查看详情
$ harness runs <run_id> --logs         # 查看每步日志
$ harness runs <run_id> --state        # 查看 state 变化历史
```

---

## 7. 与 v1 的关系

### 保留

| 组件 | 保留原因 |
|------|---------|
| StorageProtocol + SQLAlchemy 实现 | 纯基础设施，接口稳定 |
| AbstractScheduler + APScheduler | 纯基础设施 |
| AbstractNotifier + Telegram | 纯基础设施 |
| OpenAIRunner / AnthropicRunner | 接口兼容，实现成熟 |

### 重写

| 组件 | 重写原因 |
|------|---------|
| ClaudeCliRunner → ClaudeRunner | 切换到 Agent SDK |
| executor.py | State 模型完全不同，需要支持 Condition/Loop |
| task.py → tasks/ 目录 | 拆分为多文件，新增任务类型 |
| harness.py | 新增 natural() / service() / start() / stop() |
| memory.py → memory/ 目录 | 新增 Agent 长期记忆 |

### 新建

| 组件 | 说明 |
|------|------|
| State | 共享状态对象 |
| Agent | 持久化角色抽象 |
| Condition / Loop | 流程控制 |
| Adapter / Registry | 集成插件系统 |
| Planner | NL → 执行计划 |
| EventBus | 事件触发 |
| AgentMemory | 角色长期记忆 |

### 迁移指南

```python
# v1
await h.pipeline([
    FunctionTask(fn=lambda results: do_something()),
    LLMTask("分析", output_schema=MyOutput),
])

# v2（兼容模式 — 保留 results 参数）
await h.pipeline([
    FunctionTask(fn=lambda state: do_something(state)),
    LLMTask("分析", output_schema=MyOutput, output_key="analysis"),
])
```

---

## 8. 实现优先级

| Phase | 内容 | 依赖 |
|-------|------|------|
| **P0** | State + 新 executor + Condition + Loop | 无 |
| **P0** | ClaudeRunner（Agent SDK） | 无 |
| **P0** | tasks/ 拆分 + 保留 v1 任务类型 | State |
| **P1** | Adapter Registry + 首批 Adapter | State |
| **P1** | Agent 抽象 + AgentMemory | State, Runner |
| **P1** | Planner + 用户交互 | Adapter, Agent |
| **P2** | EventBus + Service 模式 | Scheduler, State |
| **P2** | CLI 增强（harness run / adapter） | Planner |
| **P3** | 更多 Adapter + 生态扩展 | Registry |

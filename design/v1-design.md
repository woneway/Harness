# Harness - 设计文档 v1

> 通用自动化流水线框架，内置 Claude Code runner

**状态：v1 已实现**（206 tests pass，2026-03-22）
**设计确认日期：2026-03-22**

---

## 一、定位

Harness 是一个 **AI-native 的通用自动化流水线框架**。

### 核心价值

> **声明你的流水线，框架负责可靠执行、记录、重试、调度、通知。**

### 差异化

| 框架 | 定位 | Harness 的区别 |
|------|------|--------------|
| Prefect / Airflow | 通用数据管道 | 太重，需要服务器，不懂 LLM |
| LangGraph / CrewAI | LLM Agent 编排 | LLM-only，无法表达 Shell/API 步骤 |
| **Harness** | **AI-native 管道，Local-first** | 简洁纯 Python API + LLM 是一等公民 + 开箱即用 |

**Claude Code runner 是默认 runner，也是核心卖点，但框架本身不依赖它。**

现有框架（LangGraph、CrewAI）均基于 LLM HTTP API。Harness 的 ClaudeCliRunner 直接调用 **Claude Code CLI 子进程**：

- **完整工具链**：文件读写、bash 执行、MCP 插件——不需要自己实现 tool calling
- **bypassPermissions**：真正的无人值守自动化（默认开启）
- **Claude Code session**：不只是对话历史，还有工具执行历史、文件系统状态

---

## 二、用户接入

用户只需要了解 **Task 类型** 和 `Harness`。

```python
from harness import Harness, LLMTask, FunctionTask, ShellTask, PollingTask, Parallel, Dialogue, Role
from pydantic import BaseModel

class ScriptResult(BaseModel):
    scenes: list[str]
    summary: str

h = Harness(project_path="/path/to/project")

# 单次调用（pipeline 的语法糖）
result = await h.run("分析并修复代码质量问题")

# 多步流水线：混合 LLM、函数、Shell、轮询、并行
results = await h.pipeline([
    LLMTask("生成5幕短视频脚本", output_schema=ScriptResult),
    Parallel([
        PollingTask(
            submit_fn=submit_video_generation,
            poll_fn=query_video_status,
            success_condition=lambda r: r["status"] == "Success",
            poll_interval=10,
            timeout=900,
        ),
        PollingTask(
            submit_fn=submit_tts,
            poll_fn=query_tts_status,
            success_condition=lambda r: r["status"] == "done",
            poll_interval=5,
            timeout=300,
        ),
    ]),
    FunctionTask(fn=ffmpeg_merge),
    ShellTask(cmd="notify-send '视频生成完成'"),
])

# 定时运行
h.schedule(tasks=[...], cron="0 2 * * *")
await h.start()
```

没有继承，没有魔法方法，没有配置文件。

---

## 三、核心 API

### 3.1 Harness

```python
class Harness:
    def __init__(
        self,
        project_path: str,
        *,
        runner: AbstractRunner | None = None,    # None = ClaudeCliRunner()
        system_prompt: str = "",
        storage_url: str | None = None,          # None = 自动用 project_path 拼绝对路径
        memory: Memory | None = None,
        notifier: AbstractNotifier | None = None,
        stream_callback: Callable[[str], None] | None = None,       # 解析后文本回调
        raw_stream_callback: Callable[[dict], None] | None = None,  # 原始 event dict 回调
        # 两者互斥；同时设置时抛 ValueError
        default_config: TaskConfig = TaskConfig(),
        env_overrides: dict[str, str] | None = None,  # 子进程环境变量覆写（空字符串表示删除）
    ):
        self._project_path = Path(project_path).resolve()
        self._storage_url = storage_url or (
            f"sqlite:///{self._project_path}/.harness/harness.db"
        )

    async def run(
        self,
        prompt: str | Callable,
        *,
        output_schema: type[BaseModel] | None = None,
        config: TaskConfig | None = None,
    ) -> Result:  # pipeline() 的单 LLMTask 语法糖
        """pipeline() 的单 LLMTask 语法糖。"""
        result = await self.pipeline([
            LLMTask(prompt, output_schema=output_schema, config=config)
        ])
        return result.results[0]

    async def pipeline(
        self,
        tasks: list[PipelineStep],
        *,
        name: str | None = None,          # 可选名称，存入 runs 表，便于 harness runs 识别
        resume_from: str | None = None,
    ) -> PipelineResult: ...

    def schedule(
        self,
        tasks: list[PipelineStep],
        *,
        cron: str,
        name: str | None = None,          # 可选名称，区分多个 scheduled pipeline
    ) -> None: ...

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
```

**Runner 解析优先级（仅对 LLMTask 有效）：**
```
LLMTask.runner（显式指定）
  → Harness.runner（实例默认）
    → ClaudeCliRunner()（框架默认）
```

### 3.2 Task 类型体系

```python
PipelineStep = LLMTask | FunctionTask | ShellTask | PollingTask | Parallel | Dialogue

# 公共基类（用户不直接实例化）
@dataclass
class BaseTask:
    config: TaskConfig | None = None
    stream_callback: Callable[[str], None] | None = None       # 覆写 Harness.stream_callback
    raw_stream_callback: Callable[[dict], None] | None = None  # 覆写 Harness.raw_stream_callback
    # 两者互斥；同时设置时抛 ValueError
```

#### LLMTask

调用语言模型，默认用 ClaudeCliRunner。

```python
@dataclass
class LLMTask(BaseTask):
    prompt: str | Callable[[list[Result]], str]
    system_prompt: str = ""
    output_schema: type[BaseModel] | None = None  # None = str，Pydantic model = 结构化输出
    runner: AbstractRunner | None = None           # None = 继承 Harness.runner
```

#### FunctionTask

执行纯 Python 函数，框架直接调用，不经过任何 runner。

```python
@dataclass
class FunctionTask(BaseTask):
    fn: Callable[[list[Result]], Any]
    output_schema: type[BaseModel] | None = None
```

- `output_schema` 的语义：对 `fn` 的返回值做类型校验（isinstance 检查）
- 校验失败时直接抛 `OutputSchemaError`，**不触发重试**——确定性函数重跑结果相同，重试无意义

#### ShellTask

执行 Shell 命令，框架用 `asyncio.create_subprocess_shell` 调用。

```python
@dataclass
class ShellTask(BaseTask):
    cmd: str | Callable[[list[Result]], str]
    cwd: str | None = None
    env: dict[str, str] | None = None
```

- `output` 为 `stdout`（str），`stderr` 附在 `Result.error` 里
- 非零退出码视为失败，进入重试流程

> **安全警告**：`cmd` 支持 `Callable[[list[Result]], str]`，意味着前序任务（尤其是 LLMTask）的输出可动态构造 shell 命令，存在**命令注入风险**。若前序为 LLMTask，建议改用 `args: list[str]` 形式（v1 可先文档警告，v2 提供 `ShellTask(args=[...])` 安全形式）。

#### PollingTask

提交异步任务后轮询结果，适用于视频生成、TTS、图像生成等外部 AI API。

```python
@dataclass
class PollingTask(BaseTask):
    submit_fn: Callable[[list[Result]], Any]            # 接收前序 Result 列表，提交任务，返回 job 句柄
    poll_fn: Callable[[Any], Any]                       # 接收 submit 返回值（job 句柄），返回当前状态
    success_condition: Callable[[Any], bool]            # 判断是否完成
    failure_condition: Callable[[Any], bool] | None = None  # 判断是否失败（区别于超时）
    poll_interval: float = 10.0
    timeout: int = 900
    output_schema: type[BaseModel] | None = None
```

#### Parallel

并发执行一组 Task（`asyncio.gather`），全部成功后继续。

```python
@dataclass
class Parallel(BaseTask):
    tasks: list[LLMTask | FunctionTask | ShellTask | PollingTask]  # v1 不支持嵌套 Parallel
    error_policy: Literal["all_or_nothing", "best_effort"] = "all_or_nothing"
    # all_or_nothing：任一失败立刻取消其他，抛 TaskFailedError
    # best_effort：收集所有结果，失败的标记 success=False，不中断
    max_retries: int = 2   # 整块 Parallel 的最大重试次数（all_or_nothing 策略生效）
```

**`Parallel` 的 `task_index` 用字符串路径表示：**
- 顺序步骤：`"0"`, `"1"`, `"2"`, ...
- Parallel 内子任务：`"2.0"`, `"2.1"`, `"2.2"`, ...（字符串比较，不会有浮点精度问题）
- 数据库中 `task_index` 列存为 VARCHAR

**断点续跑语义：** Parallel 块视为原子单元。续跑时若该块未全部成功，整体重跑。v1 不支持 Parallel 块内部的局部续跑。

**v1 不支持嵌套 `Parallel`**：`tasks` 类型不含 `Parallel`；框架在 `pipeline()` 入口做运行时校验，违反时抛 `InvalidPipelineError` 并给出清晰提示。

#### Dialogue

多角色对话，支持轮次模式和回合模式。v1 不支持嵌套在 Parallel 内部。

```python
@dataclass
class DialogueTurn:
    """一次角色发言记录。"""
    round: int       # 轮次，从 0 开始（回合模式下为发言序号）
    role_name: str
    content: str

@dataclass
class DialogueOutput:
    """Dialogue 的执行结果，作为 Result.output 存储。"""
    turns: list[DialogueTurn]   # 所有发言，按时间顺序
    rounds_completed: int       # 轮次模式：已完成轮数；回合模式：与 total_turns 相同
    total_turns: int            # 所有模式下的实际发言总次数
    final_speaker: str          # 最后发言的角色名
    final_content: str          # 最后发言的内容

@dataclass
class Role:
    """Dialogue 中的一个参与者。"""
    name: str
    system_prompt: str
    prompt: Callable[["DialogueContext"], str]  # 动态生成发言内容
    runner: AbstractRunner | None = None        # None 时继承 Harness 默认 runner

@dataclass
class Dialogue(BaseTask):
    roles: list[Role]
    background: str = ""         # 注入所有角色 system_prompt 前的背景信息
    max_rounds: int = 3          # 轮次模式的最大轮数；回合模式下用于计算默认 max_turns
    until: Callable[["DialogueContext"], bool] | None = None       # 每次发言后检查（内容条件）
    until_round: Callable[["DialogueContext"], bool] | None = None # 每轮所有角色发完后检查（轮次条件）
    # 回合模式专用：
    next_speaker: Callable[[list["DialogueTurn"]], str] | None = None  # None = 轮次模式
    max_turns: int | None = None  # None 时默认 max_rounds × len(roles)
    # 进度与流式回调：
    progress_callback: Callable[["DialogueProgressEvent"], None] | None = None
    role_stream_callback: Callable[[str, str], None] | None = None  # (role_name, chunk)
```

**两种模式：**

| 模式 | 触发条件 | 发言顺序 | 适用场景 |
|------|---------|---------|---------|
| **轮次模式**（默认） | `next_speaker=None` | 按 `roles` 列表循环 | 专家小组各自陈述、每人轮流发言 |
| **回合模式** | 设置 `next_speaker` | `next_speaker(history)` 动态决定 | 真正的辩论、角色互相点名 |

**`DialogueContext`**（`role.prompt` 的参数）：

```python
@dataclass
class DialogueContext:
    round: int                      # 当前轮次（从 0 开始）
    role_name: str                  # 当前发言角色名
    background: str                 # Dialogue.background
    history: list[DialogueTurn]     # 本次发言前的所有历史
    pipeline_results: list[Result]  # 上游 pipeline 结果

    def last_from(self, role_name: str) -> str | None: ...  # 获取指定角色最近发言
    def all_from(self, role_name: str) -> list[str]: ...    # 获取指定角色所有发言
```

**Dialogue 的 `task_index` 格式：**
- 轮次模式：`"{outer}.r{round}.{role_index}"`（如 `"2.r0.1"`）
- 回合模式：`"{outer}.t{turn}"`（如 `"2.t3"`）

**Dialogue 错误处理：**
- `role.prompt(ctx)` 抛异常：不重试，直接抛 `TaskFailedError`
- `runner.execute()` 失败/超时：按 `TaskConfig.max_retries` 重试
- `next_speaker` 返回无效角色名：立即抛 `TaskFailedError`（包含有效名称列表提示）

**session 策略：** 每个 Role 独立维护自己的 session，实现角色间上下文隔离。

### 3.3 向后兼容

```python
# task.py 底部
import warnings

def Task(*args, **kwargs):
    warnings.warn(
        "Task is deprecated, use LLMTask instead. Task will be removed in v2.",
        DeprecationWarning,
        stacklevel=2,
    )
    return LLMTask(*args, **kwargs)
```

### 3.4 AgentLeader

`AgentLeader` 是 `ClaudeCliRunner` 的封装，用于控制 agent 可见性。

```python
class AgentLeader(AbstractRunner):
    def __init__(self, agents: list[str]):
        self.agents = agents
        self._runner = ClaudeCliRunner()

    async def execute(self, prompt, *, system_prompt, **kwargs):
        constraint = (
            f"\n\n可用 agent 白名单：{', '.join(self.agents)}。"
            "不要调用列表之外的 agent。"
        )
        return await self._runner.execute(
            prompt, system_prompt=system_prompt + constraint, **kwargs
        )
```

**v1**：system_prompt 软约束；**v2**：隔离 HOME 目录硬约束（待 Claude Code 支持后替换）。

**用法：**

```python
h.pipeline([
    LLMTask("扫描项目"),
    LLMTask("修复代码", runner=AgentLeader(["code-reviewer", "tdd"])),
    LLMTask("运行测试"),
])
```

**Runner 继承树：**

```
AbstractRunner
  └── ClaudeCliRunner
        └── AgentLeader
  └── OpenAIRunner（将来）
  └── HailuoRunner（将来，HTTP runner 示例）
```

### 3.5 TaskConfig

`TaskConfig` 只包含与 runner 无关的通用配置。

```python
@dataclass(frozen=True)
class TaskConfig:
    timeout: int = 3600
    max_retries: int = 2
    backoff_base: float = 2.0    # 第n次重试等待 backoff_base^retry_count 秒
```

**优先级：Task.config > Harness.default_config > TaskConfig 默认值**

#### ClaudeCliRunner 专用配置

`PermissionMode` 等 Claude Code 专用参数在 `ClaudeCliRunner` 构造函数中指定，不出现在 `TaskConfig`：

```python
class PermissionMode(StrEnum):
    BYPASS       = "bypassPermissions"
    DEFAULT      = "default"
    ACCEPT_EDITS = "acceptEdits"
    DONT_ASK     = "dontAsk"
    PLAN         = "plan"

# 使用方式
h = Harness(project_path=".")
# 等同于 runner=ClaudeCliRunner(permission_mode=PermissionMode.BYPASS)（默认无人值守）

# 需要调整时：
h = Harness(project_path=".", runner=ClaudeCliRunner(permission_mode=PermissionMode.DEFAULT))
```

### 3.6 system_prompt 合并规则

```
final_system_prompt =
    Harness.system_prompt
    + ("\n\n" + LLMTask.system_prompt if LLMTask.system_prompt else "")
    + memory_injection
```

- `Harness.system_prompt` 是全局约束，始终保留
- `LLMTask.system_prompt` 是任务补充，追加在后
- `memory_injection` 始终在末尾

仅对 `LLMTask` 生效。

### 3.7 Result / PipelineResult

```python
@dataclass(frozen=True)
class Result:
    task_index: str                  # 顺序步骤："0","1","2"；Parallel 子任务："2.0","2.1"
    task_type: Literal["llm", "function", "shell", "polling", "dialogue"]
    output: BaseModel | str | Any    # LLMTask: BaseModel 或 str；Dialogue: DialogueOutput；其他: fn 返回值 / stdout
    raw_text: str | None             # LLMTask: Claude 原始输出；Dialogue: 最后发言内容；其他: None 或 stdout
    tokens_used: int                 # 非 LLM Task 为 0（Dialogue 累计所有发言 tokens）
    duration_seconds: float
    success: bool
    error: str | None

@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    name: str | None                 # pipeline() 的 name 参数，便于 harness runs 识别
    results: list[Result]
    total_tokens: int
    total_duration_seconds: float
```

### 3.8 Memory

```python
@dataclass
class Memory:
    history_runs: int = 3
    memory_file: str = ".harness/memory.md"
    max_tokens: int = 2000
```

Memory 注入仅对 `LLMTask` 有意义，框架在注入时自动跳过非 LLM Task。

---

## 四、Session 策略

Session 管理仅对 `LLMTask`（ClaudeCliRunner）有效，其他 Task 类型无 session 概念。

| 场景 | 行为 |
|------|------|
| `h.run()` | 独立 session |
| `h.pipeline()` 正常执行 | 所有 LLMTask 共享 session，非 LLM Task 不参与 |
| LLMTask 重试 | 生成新 session_id，后续 LLMTask 跟进 |
| 断点续跑 | 开启新 session |

**session 断开时兜底**：框架从 SQLite 读取前序所有成功 Task 的输出，注入到当前 LLMTask 的 prompt 前：

```
=== 前序任务输出 ===
Task 0 [llm]: {output.summary 或 raw_text[:300]}
Task 1 [function]: {str(output)[:200]}
Task 2 [shell]: {stdout[:200]}
```

---

## 五、断点续跑

```python
try:
    results = await h.pipeline([...])
except TaskFailedError as e:
    print(f"run_id: {e.run_id}, 失败在 Task {e.task_index}")

# 续跑：跳过已成功的步骤
results = await h.pipeline([...], resume_from=e.run_id)
```

```bash
harness runs           # 列出最近的 run
harness runs --failed  # 只看失败的
```

---

## 六、失败处理

| Task 类型 | 触发重试的情况 |
|---------|-------------|
| LLMTask | subprocess 异常 / 超时 / output_schema 校验失败 |
| FunctionTask | fn 抛出异常 |
| ShellTask | 非零退出码 / 超时 |
| PollingTask | submit_fn 异常 / failure_condition 触发 / 超时 |
| Dialogue（单次发言） | runner.execute() 异常 / 超时 |

| 情况 | 处理 |
|------|------|
| LLMTask output_schema 校验失败 | 重试，prompt 追加错误信息 |
| LLMTask prompt Callable 抛异常 | 不重试，直接抛 TaskFailedError |
| Dialogue role.prompt() 抛异常 | 不重试，直接抛 TaskFailedError |
| Dialogue next_speaker 返回无效角色名 | 不重试，直接抛 TaskFailedError |
| 超过 max_retries | 抛 TaskFailedError |
| Parallel 内 Task 失败（all_or_nothing）| 取消其余，抛 TaskFailedError |

---

## 七、stream_callback

仅对 `LLMTask` 有效。两种模式类型不同，拆分为两个独立参数避免类型歧义：

```python
# 模式1：接收解析后的文本片段（str）
Harness(stream_callback=lambda text: print(text, end="", flush=True))

# 模式2：接收原始 event dict（raw）
Harness(raw_stream_callback=lambda event: ...)
```

```python
class Harness:
    def __init__(
        self,
        ...
        stream_callback: Callable[[str], None] | None = None,       # 解析后文本
        raw_stream_callback: Callable[[dict], None] | None = None,  # 原始 event dict
        # 两者互斥；同时设置时抛 ValueError
    ): ...
```

Task 级别可覆写 Harness 级别的回调。执行每个 Task 前框架输出分隔标识：`=== Task {index} [{type}] ===`。

---

## 八、记忆管理

### 两部分记忆，分工明确

| | 运行历史 | memory.md |
|---|---|---|
| 谁维护 | 框架自动写 | Claude 自主更新（LLMTask 专属） |
| 内容 | 每次运行的结构化摘要 | 项目约定、决策、注意事项 |
| 增长控制 | 天然有界（只注入最近 N 条） | 两层约束（见下） |

### memory.md 更新机制（框架主导）

**框架负责写 memory.md，LLM 通过 output_schema 返回内容。** 这样对所有 runner（ClaudeCliRunner、OpenAIRunner 等）都适用。

```python
# output_schema 中声明可选的 memory_update 字段
class ScanResult(BaseModel):
    issues: list[str]
    summary: str
    memory_update: str | None = None   # LLM 可选填，框架写入 memory.md
```

框架在 LLMTask 执行后检查 `result.output.memory_update`，非 None 时写入 `.harness/memory.md`，触发增长控制。

**兜底（Claude Code runner 专属）**：对 `output_schema=None` 的末尾 LLMTask，追加 system_prompt 提示 Claude 直接编辑文件。此为可选的快捷路径，不是主要机制。

### memory.md 增长控制

1. **框架硬截断**：注入时超过 `max_tokens` 从头部裁掉
2. **框架合并**：写入 `memory_update` 时，若文件超过 `max_tokens` 的 80%，在写入前追加压缩提示，触发下一次 LLMTask 时整理（v1 可先跳过，只做硬截断）

### run summary 提取规则

- 优先取最后一个成功 Task 的 `output.summary` 或 `output['summary']`
- 没有则取 `str(output)[:300]`
- Task 失败时写入：`"Task {n} [{type}] 失败：{error_message}"`

### 注入格式

```
=== 最近运行历史 ===
2026-03-20: 修复了 auth bug，跳过数据库重构（需设计讨论）
2026-03-19: 添加接口限流，优化连接池

=== 项目记忆 ===
[.harness/memory.md 内容]
```

---

## 九、存储

### 初始化

首次调用 `Harness()` 时框架自动执行：
1. 创建 `{project_path}/.harness/` 目录
2. 在 `{project_path}/.gitignore` 中追加 `.harness/harness.db`（若不存在该行）
3. 创建数据库表

`memory.md` 不加入 `.gitignore`，用户自行决定是否提交。

### 默认：SQLite + WAL 模式

```python
engine = create_async_engine(storage_url, connect_args={"check_same_thread": False})
await conn.execute(text("PRAGMA journal_mode=WAL"))
```

### 迁移

```bash
harness migrate --to "postgresql+asyncpg://user:pass@host/dbname"
```

### 核心表

**`runs`**
```
id, project_path, name VARCHAR, started_at, completed_at, status, total_tokens, summary, error
```

**`task_logs`**
```
id, run_id, task_index VARCHAR, task_type, prompt_preview, output, raw_text,
tokens_used, duration_seconds, attempt, success, error, created_at
```

`task_index` 为 VARCHAR，存储字符串路径（`"0"`, `"2.0"`, `"2.1"`），避免浮点精度问题。

---

## 十、调度

```python
h.schedule(tasks=[...], cron="0 2 * * *")
await h.start()
```

```python
class AbstractScheduler(ABC):
    @abstractmethod
    def add_job(self, fn: Callable, cron: str) -> None: ...
    @abstractmethod
    async def start(self) -> None: ...
    @abstractmethod
    async def stop(self) -> None: ...
```

---

## 十一、ClaudeCliRunner

**CLI 调用：**

```
claude
  --permission-mode bypassPermissions
  --verbose
  --output-format stream-json
  --include-partial-messages
  --system-prompt <system_prompt>
  [--json-schema <schema>]
  [--resume <session_id>]   # 复用已有 session（新建 session 时省略此参数）
  -p <prompt>
```

> **注意**：Claude CLI 使用 `--resume <id>` 复用 session，不使用 `--session-id`。

**设计决策：**
- asyncio.create_subprocess_exec，非阻塞
- JSON lines 逐行解析，碰到 `result` event 提取最终输出
- 超时：先 SIGTERM，等 5 秒，未退出再 SIGKILL
- 启动检查：未安装抛 `ClaudeNotFoundError`；版本过低打印警告继续

**环境变量清理（仅作用于子进程）：**
```
移除：CLAUDECODE, CLAUDE_CODE_ENTRYPOINT, CLAUDE_CODE, CLAUDE_SESSION, CLAUDE_API_KEY
保留：ANTHROPIC_API_KEY
```

---

## 十二、包结构

```
harness/
  __init__.py          # 公开 API：Harness, LLMTask, FunctionTask, ShellTask,
                       #           PollingTask, Parallel, Dialogue, Role,
                       #           Task（LLMTask 已废弃别名，v2 移除）,
                       #           Result, PipelineResult, TaskConfig, Memory
                       #           （PermissionMode 从 harness.runners.claude_cli 导入）

  harness.py           # Harness 主类
  task.py              # 所有 Task 类型 + TaskConfig + Result + PipelineResult
                       # 含 DialogueTurn, DialogueOutput, Role, Dialogue
  memory.py            # Memory

  runners/
    base.py            # AbstractRunner
    claude_cli.py      # ClaudeCliRunner + PermissionMode（Claude 专用配置集中在此）
    agent_leader.py    # AgentLeader

  storage/
    base.py            # StorageProtocol
    sql.py             # SQLAlchemy async 实现（SQLite / MySQL / PG）
    models.py          # ORM 模型

  scheduler/
    base.py            # AbstractScheduler
    apscheduler.py     # APScheduler v4

  notifier/
    base.py            # AbstractNotifier
    telegram.py        # TelegramNotifier

  _internal/
    executor.py        # Task 派发 + 重试 + session 管理 + prompt 注入兜底
    parallel.py        # Parallel 执行逻辑（asyncio.gather）
    polling.py         # PollingTask 轮询逻辑
    dialogue.py        # Dialogue 多角色执行 + DialogueContext
    stream_parser.py   # stream-json 逐行解析
    session.py         # SessionManager
    exceptions.py      # TaskFailedError, ClaudeNotFoundError, ...

  cli.py               # harness migrate / harness runs
```

---

## 十三、元信息

| 项目 | 决策 |
|------|------|
| PyPI 包名 | `harness-ai` |
| import 名 | `from harness import ...` |
| 开源协议 | Apache 2.0 |
| Python 最低版本 | 3.11 |
| 核心依赖 | SQLAlchemy (async), APScheduler v4, pydantic |
| 异步驱动 | aiosqlite / aiomysql / asyncpg |
| 默认存储 | SQLite + WAL 模式 |
| 默认权限模式 | `PermissionMode.BYPASS` |

---

## 十四、实现状态（v1 已完成）

所有模块已实现，206 tests pass（2026-03-22）。详细实现记录见 `design/PLAN.md`（历史存档）。

**已实现：** `_internal/`（exceptions, stream_parser, session, executor, parallel, polling, dialogue）、`runners/`（base, claude_cli, agent_leader）、`storage/`（models, sql）、`task.py`（含 Dialogue/Role/DialogueTurn/DialogueOutput）、`memory.py`、`harness.py`、`scheduler/`、`notifier/`、`cli.py`

**v1 实现与原始设计的主要差异（均已在本文档同步）：**

| 项目 | 原设计 | 实际实现 |
|------|--------|---------|
| CLI session 参数 | 文档误写 `--session-id` | 实际为 `--resume <id>` |
| Task 类型 | 5 种（LLM/Function/Shell/Polling/Parallel） | 6 种，新增 `Dialogue` |
| `__init__.py` 导出 | 未含 Dialogue/Role | 已导出 `Dialogue`, `Role` |
| `env_overrides` | 未设计 | Harness 构造函数新增，透传给子进程 |
| Parallel.max_retries | 未设计 | 独立于 TaskConfig 的整块重试次数 |

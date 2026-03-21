# 设计反馈：基于视频生成管道的复用性分析

**日期：2026-03-21**
**来源：尝试将 Harness 应用于 MiniMax 短视频生成项目时的发现**

---

## 背景

尝试用 Harness 框架编排一个自动化短视频生成管道：

```
定时触发 → 抓热点 → M2.7生成脚本 → Hailuo生成视频 + TTS配音 + 音乐
        → ffmpeg合成 → Playwright发布抖音
```

复用分析结论：**基础设施层可用，执行层完全不适用。**

---

## 发现一：基础设施层与 Claude Code 耦合过紧

### 现状

Scheduler、Storage、Retry、Notifier 是**通用能力**，与 Claude Code 没有本质关联，但在当前设计里它们只能通过 `Harness` 主类获得，而 `Harness` 强依赖 `ClaudeCliRunner`。

### 影响

任何非 Claude Code 的自动化项目（MiniMax、OpenAI、纯 Python 函数）都无法复用这些基础设施，必须重新实现调度、存储、重试、通知。

### 建议

拆分为两层：

```
harness-core        ← 通用基础设施
  scheduler/        ← APScheduler 封装
  storage/          ← SQLite/PG 运行记录
  notifier/         ← Telegram 等通知
  pipeline/         ← Task/Result/Pipeline 抽象（不含 LLM 概念）
  retry/            ← 重试 + backoff

harness             ← Claude Code 专用层（依赖 harness-core）
  runners/
    claude_cli.py   ← ClaudeCliRunner
    agent_leader.py ← AgentLeader
  memory.py         ← memory.md 机制（Claude 专用）
  session.py        ← Session 管理（Claude 专用）
```

用其他 runner 的项目只需依赖 `harness-core`。

---

## 发现二：Task 模型是 LLM-only，无法表达非 LLM 步骤

### 现状

```python
@dataclass
class Task:
    prompt: str | Callable[[list[Result]], str]   # LLM-centric
    output_schema: type[BaseModel] | None          # 假设输出是文本/结构化数据
    runner: AbstractRunner | None                  # 假设有 runner
```

### 问题

真实管道里存在多种步骤类型：

| 步骤 | 类型 | 能否用 Task(prompt=...) 表达 |
|------|------|----------------------------|
| M2.7 生成脚本 | LLM | ✅ 自然 |
| Hailuo 生成视频 | HTTP API | ⚠️ 勉强 |
| ffmpeg 合并 | 系统命令 | ❌ 别扭 |
| Playwright 发布 | 浏览器自动化 | ❌ 别扭 |

### 建议

在 `harness-core` 的 pipeline 层引入 Task 类型多态：

```python
# LLM 任务（现有设计）
LLMTask(prompt="生成脚本", output_schema=ScriptResult)

# 函数任务（纯 Python）
FunctionTask(fn=merge_videos, output_schema=MergeResult)

# Shell 任务
ShellTask(cmd="ffmpeg -i input.mp4 output.mp4")
```

所有 Task 类型共享统一的 `Result` 和 pipeline 基础设施（重试、记录、通知）。

---

## 发现三：没有异步轮询模式

### 现状

`Runner.execute()` 假设**同步返回结果**。`TaskConfig.timeout` 是唯一的等待机制。

### 问题

Hailuo 视频生成是典型的异步任务：

```
POST /video_generation  →  { task_id: "xxx" }
    ↓ 等待 5~15 分钟
GET /query/video_generation?task_id=xxx  →  { status: "Success", url: "..." }
```

用 `timeout=900` 粗暴等待，无法感知任务进度，失败时也无法区分"超时"和"生成失败"。

### 建议

在 `harness-core` 里引入 `PollingTask`（或在 `FunctionTask` 上支持轮询配置）：

```python
PollingTask(
    submit_fn=submit_video_generation,
    poll_fn=query_video_status,
    success_condition=lambda r: r["status"] == "Success",
    poll_interval=10,       # 每 10 秒轮询一次
    timeout=900,
)
```

---

## 发现四：Pipeline 没有并行支持

### 现状

`pipeline(tasks)` 是严格顺序执行的。

### 问题

视频生成管道有天然的并行机会：

```
Scene 1 ──► Hailuo ──► 片段1 ─┐
Scene 2 ──► Hailuo ──► 片段2 ─┤
Scene 3 ──► Hailuo ──► 片段3 ─┴──► ffmpeg 合并
Scene 4 ──► TTS    ──► 音频1 ─┘
```

顺序执行 5 个场景需要 5×10 分钟 = 50 分钟；并行执行只需 10 分钟。

### 建议

支持并行分支语法：

```python
pipeline([
    LLMTask("生成5幕脚本", output_schema=ScriptResult),
    Parallel([
        PollingTask(submit_fn=generate_video, ...),   # 并发执行
        PollingTask(submit_fn=generate_tts, ...),
        PollingTask(submit_fn=generate_music, ...),
    ]),
    FunctionTask(fn=ffmpeg_merge),
    FunctionTask(fn=publish_to_douyin),
])
```

---

## 发现五：Memory 系统假设 LLM 有文件系统访问

### 现状

memory.md 的维护依赖 Claude 主动执行：
> "如有重要的项目约定，请更新 `.harness/memory.md`"

### 问题

非 Claude Code 的 LLM（MiniMax M2.7、OpenAI 等）是纯 HTTP API，**没有文件系统访问能力**，无法执行这个指令。

### 建议

memory.md 的更新应由**框架负责**，而不是 LLM：

```python
# 框架在 Task 执行后，从 output 提取 summary 写入 memory
# LLM 只负责在 output_schema 里返回 memory_update 字段（可选）
class TaskResult(BaseModel):
    result: ...
    memory_update: str | None = None   # LLM 可选填，框架写入
```

这样 memory 机制对所有 runner 都适用。

---

## 总结

| 发现 | 影响范围 | 优先级 |
|------|---------|--------|
| 基础设施与 Claude Code 耦合 | harness-core 拆分 | 高 |
| Task 模型 LLM-only | 引入 FunctionTask/ShellTask | 高 |
| 缺少异步轮询模式 | 引入 PollingTask | 中 |
| 缺少并行支持 | 引入 Parallel | 中 |
| Memory 依赖 LLM 文件系统 | 改为框架维护 | 低 |

**核心建议：** 将 Harness 从"Claude Code CLI 封装"重新定位为"通用自动化流水线框架，内置 Claude Code runner"。这样既保留原有价值，又大幅扩展适用场景。

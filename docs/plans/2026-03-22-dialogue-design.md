# Dialogue — 多角色辩论/批评循环设计

**日期**：2026-03-22
**状态**：已批准，待实现

---

## 背景

Harness v1 的 pipeline 通信机制是单向顺序传递：每个 Task 通过 `Callable[[list[Result]], str]` 读取前序结果，但无法在任务之间来回交流。

本设计解决"辩论/批评循环"场景：多个 LLM 角色轮流发言，每个角色有独立记忆，知道背景、知道对方说了什么，直到收敛或达到轮数上限。

---

## 设计决策

### 为什么不用 AgentLeader（Claude agent teams）

AgentLeader 把编排权交给 Claude 自己，Harness 对内部调用不可见：无 task_index、无存储日志、无法重试单步、成本不可预测。

Dialogue 保持 Harness 的核心价值：**Python 编排，框架保证可观测、可重试、可持久化**。

### 为什么不是 Loop

Loop 是通用循环结构，Dialogue 是有明确语义的专用类型：每个角色有独立 session（记忆）、共享背景、通过 `DialogueContext` 访问对方的发言。Loop 无法自然表达这些语义。

### session 策略

每个 Role 独立维护自己的 Claude CLI session（`--resume`）。角色按顺序交替调用，无并发，无竞争。

Claude Code session 存储在本地磁盘，顺序交替调用下 `--resume` 完全可靠。

---

## 数据模型

```python
# harness/task.py

@dataclass
class Role:
    """Dialogue 中的一个参与者。"""
    name: str                                      # 唯一标识，用于 session 键和 history 查询
    system_prompt: str                             # 角色的人格/职责定义
    prompt: Callable[[DialogueContext], str]       # 每轮发言前构造 prompt
    runner: Any | None = None                      # None 时继承 Harness 默认 runner


@dataclass
class Dialogue(BaseTask):
    """多角色循环对话，每个角色有独立 Claude session。

    执行流程：
    1. 按 roles 顺序，依次让每个角色发言（一轮）
    2. 一轮结束后检查 until 条件
    3. 达到 max_rounds 或 until 返回 True 时终止
    4. 产出单个 Result，output 为 DialogueOutput

    v1 限制：不支持嵌套在 Parallel 内部。
    """
    roles: list[Role] = field(default_factory=list)
    background: str = ""               # 注入每个角色 system_prompt 的前缀
    max_rounds: int = 3
    until: Callable[[DialogueContext], bool] | None = None
```

---

## DialogueContext 与 DialogueTurn

```python
# harness/_internal/dialogue.py

@dataclass(frozen=True)
class DialogueTurn:
    round: int         # 轮次（从 0 开始）
    role_name: str
    content: str


@dataclass
class DialogueContext:
    """每次调用 Role.prompt 时传入。"""
    round: int                          # 当前轮次
    role_name: str                      # 当前发言角色
    background: str                     # Dialogue.background
    history: list[DialogueTurn]         # 本次发言之前的所有历史
    pipeline_results: list[Result]      # 上游 pipeline 结果（与其他 Task 对齐）

    def last_from(self, role_name: str) -> str | None:
        """获取指定角色最近一次发言内容。"""
        for turn in reversed(self.history):
            if turn.role_name == role_name:
                return turn.content
        return None

    def all_from(self, role_name: str) -> list[str]:
        """获取指定角色所有历史发言。"""
        return [t.content for t in self.history if t.role_name == role_name]
```

---

## Dialogue 输出

Dialogue 执行完后，向 pipeline 的 `results` 列表追加**一个** `Result`：

```python
@dataclass(frozen=True)
class DialogueOutput:
    turns: list[DialogueTurn]      # 所有轮次所有角色的发言
    rounds_completed: int
    final_speaker: str             # 最后发言的角色名
    final_content: str             # 最后发言的内容
```

下游 `LLMTask` 可以通过 `results[-1].output` 访问完整对话记录，或直接用 `results[-1].output.final_content`。

**不在 Dialogue 上放 `output_schema`**。需要结构化输出时，在 Dialogue 后接一个 `LLMTask`：

```python
Dialogue(roles=[...], max_rounds=3),
LLMTask("基于上述辩论，输出结构化结论", output_schema=ConclusionSchema),
```

---

## `until` 触发时机

每轮所有角色发言完毕后检查一次。不在单个角色发言后检查，保证每轮语义完整。

---

## session 管理

`execute_dialogue` 内部维护：

```python
role_sessions: dict[str, str | None]  # role_name -> session_id
role_broken: dict[str, bool]          # role_name -> 是否断开
```

每次调用某角色前：
- 若该角色 session 正常：传入 `session_id` 做 `--resume`
- 若该角色 session 断开：框架自动将 `history` 中该角色的所有历史发言拼入 prompt 开头，作为恢复上下文，用户的 `prompt` callable 无需处理此边界

各角色 session 独立容错，互不影响。

---

## task_index 命名

```
"{pipeline_outer_index}.r{round}.{role_index}"
```

例：Dialogue 是 pipeline 第 2 步，共 2 个角色，跑 3 轮：

```
2.r0.0  →  round 0, analyzer
2.r0.1  →  round 0, critic
2.r1.0  →  round 1, analyzer
2.r1.1  →  round 1, critic
2.r2.0  →  round 2, analyzer
2.r2.1  →  round 2, critic
```

storage 层无需改动（task_index 已是 VARCHAR(50)）。

---

## system_prompt 合并规则

```
background + "\n\n" + role.system_prompt + "\n\n" + harness.system_prompt + "\n\n" + memory_injection
```

background 为空时跳过。

---

## 完整使用示例

```python
from harness import Harness, Dialogue, Role, LLMTask, FunctionTask

h = Harness(project_path=".")

await h.pipeline([
    FunctionTask(fn=load_codebase),          # 加载待审查代码

    Dialogue(
        background="审查高并发缓存模块，目标是找出所有竞态条件。",
        roles=[
            Role(
                name="analyzer",
                system_prompt="你是并发安全专家，负责识别问题并提出修复方案。",
                prompt=lambda ctx: (
                    "请分析代码中的并发问题。"
                    if ctx.round == 0
                    else f"批评者说：{ctx.last_from('critic')}\n\n请修正你的分析。"
                ),
            ),
            Role(
                name="critic",
                system_prompt="你是严格的代码审查者，负责挑战分析者的每一个结论。",
                prompt=lambda ctx: (
                    f"请挑战以下分析，指出漏洞：\n\n{ctx.last_from('analyzer')}"
                ),
            ),
        ],
        max_rounds=4,
        until=lambda ctx: "我同意" in (ctx.last_from("critic") or ""),
    ),

    LLMTask(
        prompt=lambda results: (
            f"基于以下辩论记录，输出最终修复方案：\n\n"
            + "\n".join(
                f"[{t.role_name} round {t.round}] {t.content}"
                for t in results[-1].output.turns
            )
        ),
    ),
])
```

---

## 与现有架构的兼容性

| 组件 | 变更 | 破坏性 |
|------|------|--------|
| `task.py` | 新增 `Role`、`Dialogue`、`DialogueOutput` | 无 |
| `harness.py` | 主循环新增 `elif isinstance(task, Dialogue)` | 无 |
| `_internal/dialogue.py` | 新文件，`execute_dialogue()` | 无 |
| `SessionManager` | 不动，Dialogue 独立管理 per-role sessions | 无 |
| `storage/` | 不动，task_index 格式已兼容 | 无 |
| `__init__.py` | 新增导出 `Dialogue`、`Role` | 无 |

---

## v1 限制（待 v2 解决）

- 不支持 3 个以上角色的群体讨论（技术上可行，但未测试）
- 不支持嵌套在 `Parallel` 内部
- `until` 只在整轮结束后检查，不支持单角色发言后提前退出
- 不支持动态增减角色

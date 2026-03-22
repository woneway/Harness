"""Memory — 项目记忆注入与更新。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.storage.base import StorageProtocol


@dataclass
class Memory:
    """项目记忆配置。

    Args:
        history_runs: 注入最近多少条 run summary。
        memory_file: memory 文件路径（相对于 project_path）。
        max_tokens: 注入内容的最大 token 数（字符数近似）。
    """

    history_runs: int = 3
    memory_file: str = ".harness/memory.md"
    max_tokens: int = 2000

    async def build_injection(
        self,
        storage: "StorageProtocol",
        project_path: Path,
    ) -> str:
        """构建注入到 system_prompt 的记忆字符串。

        格式：
            === 最近运行历史 ===
            2026-03-20: {summary}
            ...
            === 项目记忆 ===
            {memory.md 内容}

        超过 max_tokens 时从头部截断。
        """
        parts: list[str] = []

        # 运行历史
        try:
            runs = await storage.list_runs(
                str(project_path), limit=self.history_runs
            )
        except Exception:
            runs = []

        if runs:
            lines = ["=== 最近运行历史 ==="]
            for run in reversed(runs):  # 时间从旧到新
                started = run.get("started_at")
                date_str = str(started)[:10] if started else "?"
                summary = run.get("summary") or run.get("error") or "(无摘要)"
                lines.append(f"{date_str}: {summary}")
            parts.append("\n".join(lines))

        # memory.md 内容
        memory_path = project_path / self.memory_file
        if memory_path.exists():
            try:
                content = memory_path.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(f"=== 项目记忆 ===\n{content}")
            except Exception:
                pass

        if not parts:
            return ""

        combined = "\n\n".join(parts)

        # 超过 max_tokens 从头部截断
        if len(combined) > self.max_tokens:
            combined = combined[-self.max_tokens :]

        return combined

    def consolidation_system_prompt(self, project_path: Path) -> str:
        """生成 memory.md 整理提示，注入到末尾 LLMTask 的 system_prompt。

        当末尾 LLMTask 无 output_schema 时，框架自动追加此提示，
        告知 Claude 在完成主任务后将关键信息写入 memory.md。
        """
        memory_path = project_path / self.memory_file
        return (
            f"完成主要任务后，请将本次运行中值得记录的关键发现、决策和上下文"
            f"写入文件 `{memory_path}`。"
            "若文件不存在则创建，若已存在则追加或整理更新，内容简洁即可。"
        )

    def write_memory_update(
        self,
        project_path: Path,
        content: str,
    ) -> None:
        """将 content 追加写入 memory_file。

        写入后若文件超过 max_tokens 的 80%，做硬截断（v1 不做 LLM 压缩）。
        """
        memory_path = project_path / self.memory_file
        memory_path.parent.mkdir(parents=True, exist_ok=True)

        # 追加写入
        with memory_path.open("a", encoding="utf-8") as f:
            f.write("\n" + content)

        # 增长控制：硬截断
        threshold = int(self.max_tokens * 0.8)
        try:
            current = memory_path.read_text(encoding="utf-8")
            if len(current) > threshold:
                # 从末尾保留 threshold 个字符（保留最近内容）
                memory_path.write_text(current[-threshold:], encoding="utf-8")
        except Exception:
            pass

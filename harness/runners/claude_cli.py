"""ClaudeCliRunner — 调用 Claude Code CLI 子进程的 runner。"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
from enum import StrEnum
from typing import Callable

logger = logging.getLogger(__name__)

from harness._internal.exceptions import ClaudeNotFoundError
from harness._internal.stream_parser import StreamParser
from harness.runners.base import AbstractRunner, RunnerResult

# 子进程环境变量清理列表（不透传给 Claude Code 子进程）
_ENV_VARS_TO_REMOVE = frozenset(
    [
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE",
        "CLAUDE_SESSION",
        "CLAUDE_API_KEY",
    ]
)


class PermissionMode(StrEnum):
    """Claude Code CLI 的权限模式。"""

    BYPASS = "bypassPermissions"
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    DONT_ASK = "dontAsk"
    PLAN = "plan"


class ClaudeCliRunner(AbstractRunner):
    """通过 asyncio 子进程调用 Claude Code CLI。

    Args:
        permission_mode: Claude Code 权限模式，默认 BYPASS（无人值守自动化）。
    """

    def __init__(
        self,
        permission_mode: PermissionMode = PermissionMode.BYPASS,
    ) -> None:
        self.permission_mode = permission_mode
        self._claude_path: str | None = None
        self._checked = False

    def _get_subprocess_env(
        self,
        env_overrides: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """构建子进程环境变量，移除 Claude Code 相关变量并应用覆写。"""
        env = dict(os.environ)
        for key in _ENV_VARS_TO_REMOVE:
            env.pop(key, None)
        if env_overrides:
            for key, val in env_overrides.items():
                if val == "":
                    env.pop(key, None)
                else:
                    env[key] = val
        return env

    async def _ensure_claude(self) -> str:
        """检查 claude CLI 是否可用，返回路径。"""
        if self._checked:
            if self._claude_path is None:
                raise ClaudeNotFoundError()
            return self._claude_path

        path = shutil.which("claude")
        self._checked = True

        if path is None:
            self._claude_path = None
            raise ClaudeNotFoundError()

        self._claude_path = path

        # 版本检查：失败时只打印警告，不中断
        try:
            proc = await asyncio.create_subprocess_exec(
                path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            version_str = stdout.decode().strip()
            # 简单记录版本，暂不强制版本约束
        except Exception as e:
            print(f"[harness] Warning: Claude version check failed: {e}")

        return path

    async def execute(
        self,
        prompt: str,
        *,
        system_prompt: str,
        session_id: str | None,
        output_schema_json: str | None = None,
        stream_callback: Callable[[str], None] | None = None,
        raw_stream_callback: Callable[[dict], None] | None = None,
        env_overrides: dict[str, str] | None = None,
        **kwargs: object,
    ) -> RunnerResult:
        """调用 Claude Code CLI 执行 prompt。

        Args:
            prompt: 用户 prompt。
            system_prompt: 系统 prompt。
            session_id: Claude Code session ID，用于 session 复用。
            output_schema_json: JSON Schema 字符串，传给 --json-schema 参数。
            stream_callback: 接收解析后文本片段的回调。
            raw_stream_callback: 接收原始 event dict 的回调。
            env_overrides: 额外环境变量覆写，空值表示删除。
        """
        claude_path = await self._ensure_claude()

        args = [
            claude_path,
            "--permission-mode",
            str(self.permission_mode),
            "--verbose",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
        ]

        if system_prompt:
            args += ["--system-prompt", system_prompt]

        if output_schema_json:
            args += ["--json-schema", output_schema_json]

        if session_id:
            args += ["--resume", session_id]

        args += ["-p", prompt]

        env = self._get_subprocess_env(env_overrides)

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=10 * 1024 * 1024,  # 10MB：MCP 工具返回大 JSON 时单行可超过默认 64KB
        )

        parser = StreamParser(
            stream_callback=stream_callback,
            raw_stream_callback=raw_stream_callback,
        )

        # 同时读取 stdout（解析）和 stderr（drain），防止管道缓冲区满死锁。
        # stderr 内容在子进程异常退出时记录到日志供调试。
        assert proc.stdout is not None
        assert proc.stderr is not None
        stderr_chunks: list[str] = []

        async def _read_stdout() -> None:
            async for line in proc.stdout:  # type: ignore[union-attr]
                parser.feed(line.decode(errors="replace").rstrip("\n"))

        async def _collect_stderr() -> None:
            data = await proc.stderr.read()  # type: ignore[union-attr]
            if data:
                stderr_chunks.append(data.decode(errors="replace"))

        try:
            await asyncio.gather(_read_stdout(), _collect_stderr())
        except asyncio.CancelledError:
            # 取消时发送 SIGTERM → 等 5s → SIGKILL
            await self._terminate(proc)
            raise

        await proc.wait()

        if proc.returncode != 0 and stderr_chunks:
            stderr_text = "".join(stderr_chunks)
            logger.warning("claude exited %d, stderr: %s", proc.returncode, stderr_text[:1000])

        return RunnerResult(
            text=parser.final_text or "",
            tokens_used=parser.tokens_used,
            session_id=parser.session_id or session_id,
        )

    async def _terminate(self, proc: asyncio.subprocess.Process) -> None:
        """发送 SIGTERM，等待 5 秒后若未退出则 SIGKILL。"""
        try:
            proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        except ProcessLookupError:
            pass  # 进程已退出

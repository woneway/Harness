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


class MCPMode(StrEnum):
    """Claude CLI 子进程的 MCP 加载模式。"""

    DISABLE = "disable"  # 禁用所有 MCP，加载 --strict-mcp-config
    INHERIT = "inherit"  # 不加任何 MCP 参数，允许继承父进程配置
    SPECIFY = "specify"  # --strict-mcp-config + --mcp-config 指定配置文件


class ClaudeCliRunner(AbstractRunner):
    """通过 asyncio 子进程调用 Claude Code CLI。

    Args:
        permission_mode: Claude Code 权限模式，默认 BYPASS（无人值守自动化）。
        mcp_mode: MCP 加载模式。
            - DISABLE（默认）：禁用所有 MCP，加载 --strict-mcp-config
            - INHERIT：不加任何 MCP 参数，允许继承父进程配置
            - SPECIFY：严格模式，只加载 mcp_configs 指定的配置文件
        mcp_configs: 当 mcp_mode 为 SPECIFY 时，指定要加载的 MCP 配置文件路径。
    """

    def __init__(
        self,
        permission_mode: PermissionMode = PermissionMode.BYPASS,
        mcp_mode: MCPMode = MCPMode.DISABLE,
        mcp_configs: list[str] | None = None,
    ) -> None:
        self.permission_mode = permission_mode
        self.mcp_mode = mcp_mode
        self.mcp_configs = mcp_configs
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

        if self.mcp_mode == MCPMode.DISABLE:
            # 使用 --strict-mcp-config 覆盖所有 MCP 配置（~/.claude.json 等），
            # 防止子进程继承父进程的 MCP server 连接导致挂起。
            args += ["--strict-mcp-config"]
        elif self.mcp_mode == MCPMode.SPECIFY:
            # 严格模式：只加载指定的 MCP 配置，禁用所有继承的配置。
            if not self.mcp_configs:
                raise ValueError("mcp_mode=SPECIFY requires mcp_configs to be set")
            args += ["--strict-mcp-config"]
            for config in self.mcp_configs:
                args += ["--mcp-config", config]
        # MCPMode.INHERIT: 不加任何 MCP 参数，允许继承父进程配置

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

        # 优先用 result 事件的 text；为空时回退到最后一个 assistant 消息
        # （Claude 使用工具后 result 可能为空，但 assistant 消息有完整文本）
        effective_text = parser.final_text or parser._last_assistant_text or ""

        return RunnerResult(
            text=effective_text,
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

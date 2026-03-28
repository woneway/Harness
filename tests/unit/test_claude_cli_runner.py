"""tests/unit/test_claude_cli_runner.py — ClaudeCliRunner 单元测试（mock subprocess）。"""

from __future__ import annotations

import asyncio
import os
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness._internal.exceptions import ClaudeNotFoundError
from harness.runners.claude_cli import ClaudeCliRunner, MCPMode, PermissionMode, _ENV_VARS_TO_REMOVE


class TestPermissionMode:
    def test_values(self) -> None:
        assert PermissionMode.BYPASS == "bypassPermissions"
        assert PermissionMode.DEFAULT == "default"
        assert PermissionMode.PLAN == "plan"


class TestMCPMode:
    def test_mcp_mode_default(self) -> None:
        runner = ClaudeCliRunner()
        assert runner.mcp_mode == MCPMode.DISABLE

    def test_mcp_mode_explicit_inherit(self) -> None:
        runner = ClaudeCliRunner(mcp_mode=MCPMode.INHERIT)
        assert runner.mcp_mode == MCPMode.INHERIT

    def test_mcp_mode_specify(self) -> None:
        runner = ClaudeCliRunner(mcp_mode=MCPMode.SPECIFY, mcp_configs=["/path/to/config.json"])
        assert runner.mcp_mode == MCPMode.SPECIFY
        assert runner.mcp_configs == ["/path/to/config.json"]


class TestGetSubprocessEnv:
    def test_removes_claude_vars(self) -> None:
        runner = ClaudeCliRunner()
        with patch.dict(os.environ, {"CLAUDECODE": "1", "CLAUDE_CODE": "1", "PATH": "/usr/bin"}):
            env = runner._get_subprocess_env()
            assert "CLAUDECODE" not in env
            assert "CLAUDE_CODE" not in env
            assert "PATH" in env

    def test_keeps_normal_vars(self) -> None:
        runner = ClaudeCliRunner()
        with patch.dict(os.environ, {"MY_VAR": "hello"}, clear=False):
            env = runner._get_subprocess_env()
            assert env["MY_VAR"] == "hello"


class TestEnsureClaude:
    @pytest.mark.asyncio
    async def test_claude_not_found(self) -> None:
        runner = ClaudeCliRunner()
        with patch("shutil.which", return_value=None):
            with pytest.raises(ClaudeNotFoundError):
                await runner._ensure_claude()

    @pytest.mark.asyncio
    async def test_cached_not_found_raises(self) -> None:
        """第二次调用直接从缓存抛 ClaudeNotFoundError。"""
        runner = ClaudeCliRunner()
        with patch("shutil.which", return_value=None):
            with pytest.raises(ClaudeNotFoundError):
                await runner._ensure_claude()
        # 第二次不调用 which，直接从缓存抛
        with pytest.raises(ClaudeNotFoundError):
            await runner._ensure_claude()

    @pytest.mark.asyncio
    async def test_claude_found(self) -> None:
        runner = ClaudeCliRunner()
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"1.0.0", b""))

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            path = await runner._ensure_claude()
            assert path == "/usr/bin/claude"

    @pytest.mark.asyncio
    async def test_cached_found(self) -> None:
        runner = ClaudeCliRunner()
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"1.0.0", b""))

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await runner._ensure_claude()

        # 第二次应直接返回缓存
        path = await runner._ensure_claude()
        assert path == "/usr/bin/claude"

    @pytest.mark.asyncio
    async def test_version_check_failure_logs_warning(self, capsys) -> None:
        runner = ClaudeCliRunner()
        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("asyncio.create_subprocess_exec", side_effect=OSError("fail")):
            path = await runner._ensure_claude()
            assert path == "/usr/bin/claude"
        captured = capsys.readouterr()
        assert "Warning" in captured.out


class TestExecute:
    @pytest.mark.asyncio
    async def test_basic_execution(self) -> None:
        """基本执行流程：构建 args、启动子进程、解析输出。"""
        runner = ClaudeCliRunner()
        runner._checked = True
        runner._claude_path = "/usr/bin/claude"

        # mock stdout 返回一条 stream-json 结果行
        import json
        result_msg = json.dumps({
            "type": "result",
            "result": "hello world",
            "session_id": "sess-1",
            "cost_usd": 0.01,
            "duration_ms": 100,
            "total_cost_usd": 0.01,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

        mock_stdout = AsyncMock()
        mock_stdout.__aiter__ = lambda self: self
        mock_stdout.__anext__ = AsyncMock(side_effect=[
            (result_msg + "\n").encode(),
            StopAsyncIteration(),
        ])

        mock_stderr = AsyncMock()
        mock_stderr.read = AsyncMock(return_value=b"")

        mock_proc = AsyncMock()
        mock_proc.stdout = mock_stdout
        mock_proc.stderr = mock_stderr
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch.object(runner, "_get_subprocess_env", return_value={"PATH": "/usr/bin"}):
            result = await runner.execute(
                "test prompt",
                system_prompt="test sp",
                session_id=None,
            )

        assert result.text == "hello world"
        assert result.session_id == "sess-1"

    @pytest.mark.asyncio
    async def test_strict_mcp_config_flag(self) -> None:
        """mcp_mode=DISABLE（默认）时添加 --strict-mcp-config。"""
        runner = ClaudeCliRunner()
        runner._checked = True
        runner._claude_path = "/usr/bin/claude"

        captured_args = []

        async def mock_create(*args, **kwargs):
            captured_args.extend(args)
            mock_stdout = AsyncMock()
            mock_stdout.__aiter__ = lambda self: self
            mock_stdout.__anext__ = AsyncMock(side_effect=StopAsyncIteration())
            mock_stderr = AsyncMock()
            mock_stderr.read = AsyncMock(return_value=b"")
            proc = AsyncMock()
            proc.stdout = mock_stdout
            proc.stderr = mock_stderr
            proc.wait = AsyncMock(return_value=0)
            proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_create), \
             patch.object(runner, "_get_subprocess_env", return_value={}):
            await runner.execute("prompt", system_prompt="", session_id=None)

        assert "--strict-mcp-config" in captured_args

    @pytest.mark.asyncio
    async def test_inherit_mode_no_strict_mcp_config(self) -> None:
        """mcp_mode=INHERIT 时不添加 --strict-mcp-config。"""
        runner = ClaudeCliRunner(mcp_mode=MCPMode.INHERIT)
        runner._checked = True
        runner._claude_path = "/usr/bin/claude"

        captured_args = []

        async def mock_create(*args, **kwargs):
            captured_args.extend(args)
            mock_stdout = AsyncMock()
            mock_stdout.__aiter__ = lambda self: self
            mock_stdout.__anext__ = AsyncMock(side_effect=StopAsyncIteration())
            mock_stderr = AsyncMock()
            mock_stderr.read = AsyncMock(return_value=b"")
            proc = AsyncMock()
            proc.stdout = mock_stdout
            proc.stderr = mock_stderr
            proc.wait = AsyncMock(return_value=0)
            proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_create), \
             patch.object(runner, "_get_subprocess_env", return_value={}):
            await runner.execute("prompt", system_prompt="", session_id=None)

        assert "--strict-mcp-config" not in captured_args

    @pytest.mark.asyncio
    async def test_specify_mode_adds_config(self) -> None:
        """mcp_mode=SPECIFY 时添加 --strict-mcp-config 和 --mcp-config。"""
        runner = ClaudeCliRunner(mcp_mode=MCPMode.SPECIFY, mcp_configs=["/a.json", "/b.json"])
        runner._checked = True
        runner._claude_path = "/usr/bin/claude"

        captured_args = []

        async def mock_create(*args, **kwargs):
            captured_args.extend(args)
            mock_stdout = AsyncMock()
            mock_stdout.__aiter__ = lambda self: self
            mock_stdout.__anext__ = AsyncMock(side_effect=StopAsyncIteration())
            mock_stderr = AsyncMock()
            mock_stderr.read = AsyncMock(return_value=b"")
            proc = AsyncMock()
            proc.stdout = mock_stdout
            proc.stderr = mock_stderr
            proc.wait = AsyncMock(return_value=0)
            proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_create), \
             patch.object(runner, "_get_subprocess_env", return_value={}):
            await runner.execute("prompt", system_prompt="", session_id=None)

        assert "--strict-mcp-config" in captured_args
        assert "--mcp-config" in captured_args
        # 两个配置文件都出现
        config_indices = [i for i, a in enumerate(captured_args) if a == "--mcp-config"]
        assert len(config_indices) == 2
        assert captured_args[config_indices[0] + 1] == "/a.json"
        assert captured_args[config_indices[1] + 1] == "/b.json"

    @pytest.mark.asyncio
    async def test_specify_mode_raises_without_configs(self) -> None:
        """mcp_mode=SPECIFY 但 mcp_configs 为空时抛出 ValueError。"""
        runner = ClaudeCliRunner(mcp_mode=MCPMode.SPECIFY, mcp_configs=None)
        runner._checked = True
        runner._claude_path = "/usr/bin/claude"

        with pytest.raises(ValueError, match="mcp_mode=SPECIFY requires mcp_configs"):
            await runner.execute("prompt", system_prompt="", session_id=None)

    @pytest.mark.asyncio
    async def test_session_id_passed_as_resume(self) -> None:
        """session_id 被传为 --resume 参数。"""
        runner = ClaudeCliRunner()
        runner._checked = True
        runner._claude_path = "/usr/bin/claude"

        captured_args = []

        async def mock_create(*args, **kwargs):
            captured_args.extend(args)
            mock_stdout = AsyncMock()
            mock_stdout.__aiter__ = lambda self: self
            mock_stdout.__anext__ = AsyncMock(side_effect=StopAsyncIteration())
            mock_stderr = AsyncMock()
            mock_stderr.read = AsyncMock(return_value=b"")
            proc = AsyncMock()
            proc.stdout = mock_stdout
            proc.stderr = mock_stderr
            proc.wait = AsyncMock(return_value=0)
            proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_create), \
             patch.object(runner, "_get_subprocess_env", return_value={}):
            await runner.execute(
                "prompt",
                system_prompt="",
                session_id="my-session",
            )

        assert "--resume" in captured_args
        assert "my-session" in captured_args

    @pytest.mark.asyncio
    async def test_nonzero_exit_logs_warning(self, caplog) -> None:
        """非零退出码时记录 stderr 到日志。"""
        import logging

        runner = ClaudeCliRunner()
        runner._checked = True
        runner._claude_path = "/usr/bin/claude"

        mock_stdout = AsyncMock()
        mock_stdout.__aiter__ = lambda self: self
        mock_stdout.__anext__ = AsyncMock(side_effect=StopAsyncIteration())

        mock_stderr = AsyncMock()
        mock_stderr.read = AsyncMock(return_value=b"some error occurred")

        mock_proc = AsyncMock()
        mock_proc.stdout = mock_stdout
        mock_proc.stderr = mock_stderr
        mock_proc.wait = AsyncMock(return_value=1)
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch.object(runner, "_get_subprocess_env", return_value={}), \
             caplog.at_level(logging.WARNING, logger="harness.runners.claude_cli"):
            await runner.execute("p", system_prompt="", session_id=None)

        assert "some error occurred" in caplog.text


class TestTerminate:
    @pytest.mark.asyncio
    async def test_sigterm_then_wait(self) -> None:
        """正常关闭：SIGTERM 后进程退出。"""
        runner = ClaudeCliRunner()
        mock_proc = MagicMock()
        mock_proc.send_signal = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.kill = MagicMock()

        await runner._terminate(mock_proc)

        mock_proc.send_signal.assert_called_once_with(signal.SIGTERM)

    @pytest.mark.asyncio
    async def test_sigterm_timeout_then_sigkill(self) -> None:
        """SIGTERM 超时后发送 SIGKILL。"""
        runner = ClaudeCliRunner()
        mock_proc = MagicMock()
        mock_proc.send_signal = MagicMock()

        call_count = 0

        async def slow_wait():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 第一次 wait（SIGTERM 后）超时
                await asyncio.sleep(100)
            # 第二次 wait（SIGKILL 后）立即返回

        mock_proc.wait = slow_wait
        mock_proc.kill = MagicMock()

        await runner._terminate(mock_proc)

        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_already_exited(self) -> None:
        """进程已退出时不抛异常。"""
        runner = ClaudeCliRunner()
        mock_proc = MagicMock()
        mock_proc.send_signal = MagicMock(side_effect=ProcessLookupError)

        await runner._terminate(mock_proc)  # 不抛异常

"""ShellTask — Shell 命令任务。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from harness.tasks.base import BaseTask
from harness.tasks.result import Result


@dataclass
class ShellTask(BaseTask):
    """执行 Shell 命令，框架用 asyncio.create_subprocess_shell 调用。

    .. warning::
        ``cmd`` 支持 ``Callable[[list[Result]], str]``，意味着前序任务（尤其是
        LLMTask）的输出可动态构造 shell 命令，存在**命令注入风险**。若前序为
        LLMTask，建议对输出做严格校验后再传入。
    """

    cmd: str | Callable[[list[Result]], str] = ""
    cwd: str | None = None
    env: dict[str, str] | None = None
    output_key: str | None = None  # v2: 执行后将 result.output 写入 state 属性

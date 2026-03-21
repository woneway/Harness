"""Harness 自定义异常。"""

from __future__ import annotations


def _classify_error(error: str) -> tuple[str, str]:
    """根据错误信息返回 (category, hint)。

    category: 简短的错误分类标签
    hint: 用户可操作的修复建议
    """
    e = error.lower()

    # 网络 / 代理类
    if "socks" in e and ("missing" in e or "support" in e):
        return (
            "网络错误 [SOCKS代理]",
            "缺少 SOCKS 代理支持。运行: uv add 'requests[socks]'",
        )
    if "proxyerror" in e or "proxy" in e:
        return (
            "网络错误 [代理]",
            "代理连接失败。尝试: unset HTTP_PROXY HTTPS_PROXY ALL_PROXY，或使用 Harness(env_overrides={'ALL_PROXY': ''})",
        )
    if "connectionpool" in e or "max retries exceeded" in e:
        return (
            "网络错误 [连接超时]",
            "连接远程服务失败，请检查网络或代理配置",
        )
    if "timeout" in e or "timed out" in e:
        return (
            "超时",
            "任务超时。可通过 TaskConfig(timeout=N) 延长超时时间",
        )
    if "connectionrefused" in e or "connection refused" in e:
        return (
            "网络错误 [连接被拒]",
            "目标服务拒绝连接，请确认服务是否运行",
        )

    # 导入 / 依赖类
    if "modulenotfounderror" in e or "no module named" in e:
        pkg = ""
        if "no module named" in e:
            try:
                pkg = error.split("No module named")[1].strip().strip("'\"")
            except IndexError:
                pass
        hint = f"缺少依赖包 '{pkg}'。运行: uv add {pkg}" if pkg else "缺少依赖包，请检查 pyproject.toml"
        return ("依赖缺失", hint)
    if "importerror" in e:
        return ("导入错误", "依赖导入失败，请检查包是否正确安装")

    # 认证类
    if "401" in e or "unauthorized" in e or "authentication" in e:
        return ("认证失败", "API Key 或凭据无效，请检查环境变量")
    if "403" in e or "forbidden" in e:
        return ("权限不足", "没有访问权限，请检查 API 配额或账户权限")

    # 数据 / 代码类
    if "keyerror" in e:
        return ("KeyError", "字典键不存在，请检查数据结构")
    if "typeerror" in e:
        return ("TypeError", "类型不匹配，请检查函数参数或返回值类型")
    if "valueerror" in e:
        return ("ValueError", "数据值无效，请检查输入数据")
    if "attributeerror" in e:
        return ("AttributeError", "对象属性不存在，请检查数据结构")
    if "filenotfounderror" in e or "no such file" in e:
        return ("文件不存在", "指定文件路径不存在，请检查路径是否正确")
    if "permissionerror" in e:
        return ("权限错误", "没有文件读写权限")

    # prompt callable 失败
    if "prompt callable raised" in e:
        return ("Prompt 构建失败", "prompt 函数抛出异常，请检查 results 索引是否越界")

    return ("执行错误", "请检查 Task 函数或相关依赖")


class TaskFailedError(Exception):
    """Task 超过最大重试次数或不可恢复失败时抛出。"""

    def __init__(
        self,
        run_id: str,
        task_index: str,
        task_type: str,
        error: str,
        partial_results: list | None = None,
    ) -> None:
        self.run_id = run_id
        self.task_index = task_index
        self.task_type = task_type
        self.error = error
        self.partial_results = partial_results or []

        category, hint = _classify_error(error)
        msg = (
            f"\n"
            f"{'─' * 50}\n"
            f"  Task {task_index} [{task_type}] 失败\n"
            f"  Run: {run_id[:8]}\n"
            f"  类型: {category}\n"
            f"  提示: {hint}\n"
            f"{'─' * 50}\n"
            f"  原始错误: {error}\n"
            f"{'─' * 50}"
        )
        super().__init__(msg)


class ClaudeNotFoundError(Exception):
    """Claude CLI 未安装或未在 PATH 中时抛出。"""

    def __init__(self) -> None:
        super().__init__(
            "Claude CLI not found. Please install it: https://claude.ai/code"
        )


class InvalidPipelineError(Exception):
    """Pipeline 配置无效时抛出，例如 Parallel 嵌套。"""

    pass


class OutputSchemaError(Exception):
    """FunctionTask output_schema 校验失败时抛出，不触发重试。"""

    def __init__(self, task_index: str, expected_type: type, actual_type: type) -> None:
        self.task_index = task_index
        self.expected_type = expected_type
        self.actual_type = actual_type
        super().__init__(
            f"Task {task_index} output schema validation failed: "
            f"expected {expected_type.__name__}, got {actual_type.__name__}"
        )

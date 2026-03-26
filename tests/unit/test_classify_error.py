"""tests/unit/test_classify_error.py — _classify_error 所有分支覆盖。"""

from harness._internal.exceptions import _classify_error


class TestClassifyErrorNetwork:
    def test_socks_missing(self) -> None:
        cat, hint = _classify_error("PySocksError: SOCKS support is missing")
        assert cat == "网络错误 [SOCKS代理]"
        assert "uv add" in hint

    def test_socks_missing_variant(self) -> None:
        cat, _ = _classify_error("Missing socks module for proxy")
        assert cat == "网络错误 [SOCKS代理]"

    def test_proxy_error(self) -> None:
        cat, hint = _classify_error("ProxyError: cannot connect")
        assert cat == "网络错误 [代理]"
        assert "unset" in hint

    def test_proxy_keyword(self) -> None:
        cat, _ = _classify_error("proxy connection failed")
        assert cat == "网络错误 [代理]"

    def test_connection_pool(self) -> None:
        cat, _ = _classify_error("urllib3 ConnectionPool failed")
        assert cat == "网络错误 [连接超时]"

    def test_max_retries_exceeded(self) -> None:
        cat, hint = _classify_error("Max retries exceeded with url")
        assert cat == "网络错误 [连接超时]"
        assert "网络" in hint

    def test_timeout(self) -> None:
        cat, hint = _classify_error("Request timed out after 30s")
        assert cat == "超时"
        assert "TaskConfig" in hint

    def test_connection_refused(self) -> None:
        cat, hint = _classify_error("ConnectionRefusedError: [Errno 111]")
        assert cat == "网络错误 [连接被拒]"
        assert "服务" in hint

    def test_connection_refused_phrase(self) -> None:
        cat, _ = _classify_error("Connection refused on port 8080")
        assert cat == "网络错误 [连接被拒]"


class TestClassifyErrorImport:
    def test_module_not_found_with_name(self) -> None:
        cat, hint = _classify_error("ModuleNotFoundError: No module named 'pandas'")
        assert cat == "依赖缺失"
        assert "pandas" in hint
        assert "uv add" in hint

    def test_module_not_found_lowercase(self) -> None:
        """lowercase 'no module named' 匹配分类，但 split 用 title case 提取包名。"""
        cat, hint = _classify_error("no module named 'requests'")
        assert cat == "依赖缺失"
        # 大小写不匹配 split("No module named")，回退到通用提示
        assert "pyproject.toml" in hint

    def test_module_not_found_no_name(self) -> None:
        """modulenotfounderror 但无 'No module named' 片段。"""
        cat, hint = _classify_error("ModuleNotFoundError: something else")
        assert cat == "依赖缺失"
        assert "pyproject.toml" in hint

    def test_import_error(self) -> None:
        cat, hint = _classify_error("ImportError: cannot import name 'foo'")
        assert cat == "导入错误"
        assert "安装" in hint


class TestClassifyErrorAuth:
    def test_401(self) -> None:
        cat, hint = _classify_error("HTTP 401 Unauthorized")
        assert cat == "认证失败"
        assert "API Key" in hint

    def test_unauthorized_word(self) -> None:
        cat, _ = _classify_error("authentication required for this endpoint")
        assert cat == "认证失败"

    def test_403(self) -> None:
        cat, hint = _classify_error("HTTP 403 Forbidden")
        assert cat == "权限不足"
        assert "权限" in hint

    def test_forbidden_word(self) -> None:
        cat, _ = _classify_error("Access forbidden")
        assert cat == "权限不足"


class TestClassifyErrorFile:
    def test_file_not_found(self) -> None:
        cat, hint = _classify_error("FileNotFoundError: /tmp/missing.txt")
        assert cat == "文件不存在"
        assert "路径" in hint

    def test_no_such_file(self) -> None:
        cat, _ = _classify_error("No such file or directory: /tmp/x")
        assert cat == "文件不存在"

    def test_permission_error(self) -> None:
        cat, hint = _classify_error("PermissionError: [Errno 13]")
        assert cat == "权限错误"
        assert "权限" in hint


class TestClassifyErrorMisc:
    def test_prompt_callable(self) -> None:
        cat, hint = _classify_error("Prompt callable raised: IndexError")
        assert cat == "Prompt 构建失败"
        assert "prompt" in hint

    def test_fallback(self) -> None:
        cat, hint = _classify_error("some completely unknown error")
        assert cat == "执行错误"
        assert "Task" in hint or "依赖" in hint

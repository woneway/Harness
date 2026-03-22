"""tests/integration/test_storage.py — 存储层集成测试。"""

from __future__ import annotations

import pytest

from harness.storage.sql import SQLStorage


@pytest.fixture
async def storage(tmp_path) -> SQLStorage:
    url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    s = SQLStorage(url)
    await s.init()
    return s


class TestTaskIndexVarchar:
    @pytest.mark.asyncio
    async def test_sequential_task_index(self, storage: SQLStorage) -> None:
        """顺序步骤 task_index 存为字符串 '0', '1', '2'。"""
        await storage.save_run("run-1", "/project", "test")
        await storage.save_task_log("run-1", "0", "function", output="step0")
        await storage.save_task_log("run-1", "1", "llm", output="step1")
        await storage.save_task_log("run-1", "2", "shell", output="step2")

        logs = await storage.get_task_logs("run-1")
        indices = [l["task_index"] for l in logs]
        assert indices == ["0", "1", "2"]
        assert all(isinstance(i, str) for i in indices)

    @pytest.mark.asyncio
    async def test_parallel_task_index(self, storage: SQLStorage) -> None:
        """Parallel 子任务 task_index 存为字符串 '2.0', '2.1'。"""
        await storage.save_run("run-2", "/project", None)
        await storage.save_task_log("run-2", "2.0", "function", output="pa")
        await storage.save_task_log("run-2", "2.1", "polling", output="pb")

        logs = await storage.get_task_logs("run-2")
        indices = [l["task_index"] for l in logs]
        assert "2.0" in indices
        assert "2.1" in indices
        assert all(isinstance(i, str) for i in indices)


class TestRunsName:
    @pytest.mark.asyncio
    async def test_name_field_stored_and_retrieved(self, storage: SQLStorage) -> None:
        await storage.save_run("run-name-1", "/p", "my-pipeline")
        run = await storage.get_run("run-name-1")
        assert run is not None
        assert run["name"] == "my-pipeline"

    @pytest.mark.asyncio
    async def test_null_name_allowed(self, storage: SQLStorage) -> None:
        await storage.save_run("run-name-2", "/p", None)
        run = await storage.get_run("run-name-2")
        assert run is not None
        assert run["name"] is None


class TestRunSummaryExtraction:
    @pytest.mark.asyncio
    async def test_summary_updated(self, storage: SQLStorage) -> None:
        await storage.save_run("run-s1", "/p", None)
        await storage.update_run(
            "run-s1",
            status="success",
            summary="Fixed auth bug",
            total_tokens=100,
        )
        run = await storage.get_run("run-s1")
        assert run["summary"] == "Fixed auth bug"
        assert run["total_tokens"] == 100
        assert run["status"] == "success"

    @pytest.mark.asyncio
    async def test_failed_run_stores_error(self, storage: SQLStorage) -> None:
        await storage.save_run("run-f1", "/p", None)
        await storage.update_run(
            "run-f1",
            status="failed",
            error="Task 1 [llm] 失败：timeout",
        )
        run = await storage.get_run("run-f1")
        assert run["status"] == "failed"
        assert "timeout" in run["error"]


class TestListRuns:
    @pytest.mark.asyncio
    async def test_list_runs_by_project(self, storage: SQLStorage) -> None:
        await storage.save_run("r1", "/project/a", "run1")
        await storage.save_run("r2", "/project/a", "run2")
        await storage.save_run("r3", "/project/b", "run3")

        runs = await storage.list_runs("/project/a")
        assert len(runs) == 2
        ids = {r["id"] for r in runs}
        assert ids == {"r1", "r2"}

    @pytest.mark.asyncio
    async def test_list_failed_only(self, storage: SQLStorage) -> None:
        await storage.save_run("r1", "/p", None)
        await storage.update_run("r1", status="success")
        await storage.save_run("r2", "/p", None)
        await storage.update_run("r2", status="failed")

        runs = await storage.list_runs("/p", failed_only=True)
        assert len(runs) == 1
        assert runs[0]["id"] == "r2"

    @pytest.mark.asyncio
    async def test_success_only_task_logs(self, storage: SQLStorage) -> None:
        await storage.save_run("run-sl", "/p", None)
        await storage.save_task_log("run-sl", "0", "llm", success=True, output="ok")
        await storage.save_task_log("run-sl", "1", "shell", success=False, error="fail")
        await storage.save_task_log("run-sl", "2", "function", success=True, output="ok2")

        logs = await storage.get_task_logs("run-sl", success_only=True)
        assert len(logs) == 2
        assert all(l["success"] for l in logs)


class TestOutputSchemaClass:
    @pytest.mark.asyncio
    async def test_output_schema_class_stored_and_retrieved(self, storage: SQLStorage) -> None:
        """output_schema_class 字符串能正确存储并读取。"""
        await storage.save_run("run-sc1", "/p", None)
        await storage.save_task_log(
            "run-sc1",
            "0",
            "function",
            output='{"name": "test", "value": 1}',
            output_schema_class="myapp.models.MyModel",
        )
        logs = await storage.get_task_logs("run-sc1")
        assert len(logs) == 1
        assert logs[0]["output_schema_class"] == "myapp.models.MyModel"

    @pytest.mark.asyncio
    async def test_output_schema_class_nullable(self, storage: SQLStorage) -> None:
        """不传 output_schema_class 时默认为 None。"""
        await storage.save_run("run-sc2", "/p", None)
        await storage.save_task_log(
            "run-sc2",
            "0",
            "llm",
            output="some text",
        )
        logs = await storage.get_task_logs("run-sc2")
        assert len(logs) == 1
        assert logs[0]["output_schema_class"] is None

    @pytest.mark.asyncio
    async def test_output_schema_class_multiple_tasks(self, storage: SQLStorage) -> None:
        """混合有/无 schema 的任务均能正确存取。"""
        await storage.save_run("run-sc3", "/p", None)
        await storage.save_task_log(
            "run-sc3", "0", "function",
            output='{"x": 1}',
            output_schema_class="pkg.mod.Schema",
        )
        await storage.save_task_log(
            "run-sc3", "1", "llm",
            output="plain text",
        )
        logs = await storage.get_task_logs("run-sc3")
        assert logs[0]["output_schema_class"] == "pkg.mod.Schema"
        assert logs[1]["output_schema_class"] is None

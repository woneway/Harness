"""harness CLI — 查看运行历史、迁移数据。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print(
        "CLI dependencies missing. Install with: pip install typer rich",
        file=sys.stderr,
    )
    sys.exit(1)

from harness.storage.sql import SQLStorage

app = typer.Typer(name="harness", help="Harness — AI-native 流水线框架 CLI")
console = Console()


def _get_storage(project_path: str | None) -> SQLStorage:
    path = Path(project_path or ".").resolve()
    url = f"sqlite+aiosqlite:///{path}/.harness/harness.db"
    return SQLStorage(url)


@app.command("runs")
def list_runs(
    failed: bool = typer.Option(False, "--failed", help="只显示失败的 run"),
    limit: int = typer.Option(20, "--limit", "-n", help="最多显示多少条"),
    project: str = typer.Option(".", "--project", "-p", help="项目路径"),
) -> None:
    """列出最近的 pipeline 运行记录。"""

    async def _list() -> None:
        storage = _get_storage(project)
        project_path = str(Path(project).resolve())
        runs = await storage.list_runs(project_path, limit=limit, failed_only=failed)

        if not runs:
            console.print("[yellow]No runs found.[/yellow]")
            return

        table = Table(title="Harness Runs")
        table.add_column("Run ID", style="cyan", no_wrap=True, max_width=12)
        table.add_column("Name", style="white")
        table.add_column("Started", style="white")
        table.add_column("Status", style="white")
        table.add_column("Tokens", justify="right")
        table.add_column("Summary", style="dim", max_width=50)

        for run in runs:
            run_id = (run["id"] or "")[:8]
            name = run["name"] or "-"
            started = str(run["started_at"])[:19] if run["started_at"] else "-"
            status = run["status"] or "?"
            status_colored = (
                f"[green]{status}[/green]"
                if status == "success"
                else f"[red]{status}[/red]"
                if status == "failed"
                else status
            )
            tokens = str(run["total_tokens"] or 0)
            summary = (run["summary"] or "")[:50]

            table.add_row(run_id, name, started, status_colored, tokens, summary)

        console.print(table)

    asyncio.run(_list())


@app.command("migrate")
def migrate(
    to: str = typer.Option(..., "--to", help="目标数据库 DSN（如 postgresql+asyncpg://...）"),
    project: str = typer.Option(".", "--project", "-p", help="项目路径"),
) -> None:
    """将 SQLite 数据库迁移到 MySQL/PostgreSQL。"""

    async def _migrate() -> None:
        src_storage = _get_storage(project)
        dst_storage = SQLStorage(to)
        await dst_storage.init()

        project_path = str(Path(project).resolve())
        runs = await src_storage.list_runs(project_path, limit=10000)

        console.print(f"[cyan]Migrating {len(runs)} runs...[/cyan]")

        for run in runs:
            run_id = run["id"]
            try:
                await dst_storage.save_run(run_id, run["project_path"], run["name"])
                await dst_storage.update_run(
                    run_id,
                    status=run["status"],
                    total_tokens=run["total_tokens"],
                    summary=run["summary"],
                    error=run["error"],
                )
            except Exception as e:
                console.print(f"[red]  Failed to migrate run {run_id}: {e}[/red]")
                continue

            logs = await src_storage.get_task_logs(run_id)
            for log in logs:
                try:
                    await dst_storage.save_task_log(
                        run_id,
                        log["task_index"],
                        log["task_type"],
                        output=log.get("output"),
                        raw_text=log.get("raw_text"),
                        tokens_used=log.get("tokens_used", 0),
                        duration_seconds=log.get("duration_seconds", 0.0),
                        attempt=log.get("attempt", 1),
                        success=log.get("success", True),
                        error=log.get("error"),
                    )
                except Exception as e:
                    console.print(f"[yellow]  Warning: task log {log['task_index']}: {e}[/yellow]")

        console.print(f"[green]Migration complete! {len(runs)} runs migrated.[/green]")

    asyncio.run(_migrate())


if __name__ == "__main__":
    app()

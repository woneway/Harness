"""code_stats.py — 统计 Harness 自身代码量，验证框架基础功能。

运行：
    uv run python examples/code_stats/main.py

展示 v2 State 模式：所有步骤通过 state 共享数据，替代 results[N]。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from dataclasses import dataclass

from harness import Harness, FunctionTask, LLMTask, ShellTask, State


@dataclass
class ScanResult:
    files: list[str]
    file_count: int


@dataclass
class Report:
    file_count: int
    total_lines: int
    largest_file: str
    largest_lines: int


def scan_py_files(state: State) -> ScanResult:
    root = Path(__file__).resolve().parent.parent.parent / "harness"
    files = sorted(str(p) for p in root.rglob("*.py") if p.stat().st_size > 0)
    return ScanResult(files=files, file_count=len(files))


def make_wc_cmd(state: State) -> str:
    scan: ScanResult = state.scan  # type: ignore[attr-defined]
    paths = " ".join(f'"{f}"' for f in scan.files)
    return f"wc -l {paths}"


def make_report(state: State) -> Report:
    scan: ScanResult = state.scan  # type: ignore[attr-defined]
    wc_output: str = state.wc  # type: ignore[attr-defined]

    lines = [l.strip() for l in wc_output.strip().splitlines()]
    total = int(lines[-1].split()[0]) if len(lines) > 1 else 0

    largest_line = max(lines[:-1], key=lambda l: int(l.split()[0]))
    largest_count = int(largest_line.split()[0])
    largest_name = Path(largest_line.split()[1]).name

    return Report(
        file_count=scan.file_count,
        total_lines=total,
        largest_file=largest_name,
        largest_lines=largest_count,
    )


def make_analysis_prompt(state: State) -> str:
    report: Report = state.report  # type: ignore[attr-defined]
    scan: ScanResult = state.scan  # type: ignore[attr-defined]
    file_list = "\n".join(Path(f).name for f in scan.files)
    return f"""以下是一个 Python 开源框架（Harness）的代码统计数据，请给出简短的分析：

文件数：{report.file_count}
总行数：{report.total_lines}
最大文件：{report.largest_file}（{report.largest_lines} 行）
所有文件：
{file_list}

请从以下角度分析（每点一句话）：
1. 代码规模评估（小/中/大型项目）
2. 最大文件是否存在过大风险
3. 一个改进建议"""


async def main() -> None:
    h = Harness(
        project_path=".",
        stream_callback=lambda text: print(text, end="", flush=True),
    )

    s = State()
    pr = await h.pipeline(
        [
            FunctionTask(fn=scan_py_files, output_key="scan"),
            ShellTask(cmd=make_wc_cmd, output_key="wc"),
            FunctionTask(fn=make_report, output_key="report"),
            LLMTask(prompt=make_analysis_prompt),
        ],
        state=s,
        name="code-stats",
    )

    report: Report = s.report  # type: ignore[attr-defined]
    print("\n\n📊 Harness 代码统计报告")
    print(f"  .py 文件数：{report.file_count}")
    print(f"  总行数：{report.total_lines:,}")
    print(f"  最大文件：{report.largest_file} ({report.largest_lines} 行)")
    print(f"  耗时：{pr.total_duration_seconds:.2f}s")
    print(f"  Run ID：{pr.run_id[:8]}")


if __name__ == "__main__":
    asyncio.run(main())

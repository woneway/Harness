"""code_stats.py — 统计 Harness 自身代码量，验证框架基础功能。

运行：
    python examples/code_stats.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from dataclasses import dataclass

from harness import Harness, FunctionTask, LLMTask, ShellTask


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


def scan_py_files(results: list) -> ScanResult:
    root = Path(__file__).parent.parent / "harness"
    files = sorted(str(p) for p in root.rglob("*.py") if p.stat().st_size > 0)
    return ScanResult(files=files, file_count=len(files))


def make_wc_cmd(results: list) -> str:
    scan: ScanResult = results[0].output
    paths = " ".join(f'"{f}"' for f in scan.files)
    return f"wc -l {paths}"


def make_report(results: list) -> Report:
    scan: ScanResult = results[0].output
    wc_output: str = results[1].output

    # wc -l 最后一行是 total
    lines = [l.strip() for l in wc_output.strip().splitlines()]
    total = int(lines[-1].split()[0]) if len(lines) > 1 else 0

    # 找最大文件（排除 total 行）
    largest_line = max(lines[:-1], key=lambda l: int(l.split()[0]))
    largest_count = int(largest_line.split()[0])
    largest_name = Path(largest_line.split()[1]).name

    return Report(
        file_count=scan.file_count,
        total_lines=total,
        largest_file=largest_name,
        largest_lines=largest_count,
    )


def make_analysis_prompt(results: list) -> str:
    report: Report = results[2].output
    scan: ScanResult = results[0].output
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

    pr = await h.pipeline(
        [
            FunctionTask(fn=scan_py_files),
            ShellTask(cmd=make_wc_cmd),
            FunctionTask(fn=make_report),
            LLMTask(prompt=make_analysis_prompt),
        ],
        name="code-stats",
    )

    report: Report = pr.results[2].output
    print("\n\n📊 Harness 代码统计报告")
    print(f"  .py 文件数：{report.file_count}")
    print(f"  总行数：{report.total_lines:,}")
    print(f"  最大文件：{report.largest_file} ({report.largest_lines} 行)")
    print(f"  耗时：{pr.total_duration_seconds:.2f}s")
    print(f"  Run ID：{pr.run_id[:8]}")


if __name__ == "__main__":
    asyncio.run(main())

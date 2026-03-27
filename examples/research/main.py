"""SecondBrain 开源项目调研工具 — CLI 入口。

三种输入模式：
  uv run python -m examples.research.main "langchain"                                    # 项目名
  uv run python -m examples.research.main "https://github.com/langchain-ai/langchain"   # URL
  uv run python -m examples.research.main "cft0808/edict"                                # owner/repo
  uv run python -m examples.research.main "langchain vs llamaindex vs haystack"           # 对比
  uv run python -m examples.research.main "Python AI 编排框架"                             # 主题
"""

import asyncio
import sys
import time

from harness import Harness, Result
from harness.runners.claude_cli import ClaudeCliRunner, PermissionMode
from harness.tasks import PipelineStep

from .pipeline import build_pipeline

# 步骤类型到中文名的映射
_STEP_NAMES = {
    "FunctionTask": "函数",
    "LLMTask": "LLM",
    "Condition": "条件",
    "Parallel": "并行",
    "Discussion": "讨论",
}

_step_start_time: float = 0.0


def _on_step_start(step: PipelineStep, task_index: str) -> None:
    """步骤开始时打印进度。"""
    global _step_start_time
    _step_start_time = time.monotonic()
    step_type = type(step).__name__
    name = _STEP_NAMES.get(step_type, step_type)
    print(f"\n⏳ Step {task_index} [{name}] 开始...", flush=True)


def _on_step_complete(
    step: PipelineStep,
    task_index: str,
    result: Result | list[Result],
) -> None:
    """步骤完成时打印结果摘要。"""
    elapsed = time.monotonic() - _step_start_time
    step_type = type(step).__name__
    name = _STEP_NAMES.get(step_type, step_type)

    if isinstance(result, list):
        tokens = sum(r.tokens_used for r in result)
        status = "✓" if all(r.success for r in result) else "✗"
    else:
        tokens = result.tokens_used
        status = "✓" if result.success else "✗"

    token_str = f" | {tokens:,} tokens" if tokens > 0 else ""
    print(f"\n{status} Step {task_index} [{name}] 完成 ({elapsed:.1f}s{token_str})", flush=True)


async def main() -> None:
    if len(sys.argv) < 2:
        print("用法: uv run python -m examples.research.main <调研目标>")
        print()
        print("示例:")
        print('  uv run python -m examples.research.main "langchain"')
        print('  uv run python -m examples.research.main "cft0808/edict"')
        print('  uv run python -m examples.research.main "langchain vs llamaindex vs haystack"')
        print('  uv run python -m examples.research.main "Python AI 编排框架"')
        sys.exit(1)

    raw_input = " ".join(sys.argv[1:])
    print(f"调研目标: {raw_input}")
    print("=" * 60)

    runner = ClaudeCliRunner(permission_mode=PermissionMode.BYPASS)
    steps, state = build_pipeline(runner)

    # 注入用户输入
    state.raw_input = raw_input

    h = Harness(
        project_path=".",
        runner=runner,
        system_prompt=(
            "你是一位专业的开源项目调研分析师。"
            "你擅长通过搜索、阅读文档和分析数据来全面评估开源项目。"
            "回答使用中文，技术术语保留英文。"
            "回答尽量简洁精炼，避免冗余。"
        ),
    )

    pipeline_start = time.monotonic()
    print("\n开始调研 Pipeline...")
    pr = await h.pipeline(
        steps,
        state=state,
        on_step_start=_on_step_start,
        on_step_complete=_on_step_complete,
    )
    total_elapsed = time.monotonic() - pipeline_start

    # 输出结果
    print("\n" + "=" * 60)
    print("调研完成")
    print("=" * 60)
    print(f"总耗时: {total_elapsed:.1f}s | 总 Token: {pr.total_tokens:,}")

    if state.discussion:
        print(f"Discussion: {state.discussion.rounds_completed} 轮, "
              f"收敛: {state.discussion.converged}")

    if state.output_path:
        print(f"报告已保存: {state.output_path}")
    else:
        print("报告未保存（路径为空）")

    print(f"\nPipeline 结果: {len(pr.results)} 个步骤完成")


if __name__ == "__main__":
    asyncio.run(main())

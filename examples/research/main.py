"""SecondBrain 开源项目调研工具 — CLI 入口。

三种输入模式：
  uv run python -m examples.research.main "langchain"                                    # 项目名
  uv run python -m examples.research.main "https://github.com/langchain-ai/langchain"   # URL
  uv run python -m examples.research.main "langchain vs llamaindex vs haystack"           # 对比
  uv run python -m examples.research.main "Python AI 编排框架"                             # 主题
"""

import asyncio
import sys

from harness import Harness
from harness.runners.claude_cli import ClaudeCliRunner, PermissionMode

from .pipeline import build_pipeline


async def main() -> None:
    if len(sys.argv) < 2:
        print("用法: uv run python -m examples.research.main <调研目标>")
        print()
        print("示例:")
        print('  uv run python -m examples.research.main "langchain"')
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
        ),
    )

    print("\n开始调研 Pipeline...")
    pr = await h.pipeline(steps, state=state)

    # 输出结果
    print("\n" + "=" * 60)
    print("调研完成")
    print("=" * 60)

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

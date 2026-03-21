"""video_pipeline.py — 视频生成管道示例（mock submit/poll）。

展示 Harness 的核心能力：
  LLMTask → Parallel[PollingTask × 2] → FunctionTask → ShellTask

运行方式：
    python examples/video_pipeline.py
"""

from __future__ import annotations

import asyncio
from pydantic import BaseModel

from harness import Harness, LLMTask, FunctionTask, ShellTask, TaskConfig
from harness.task import Parallel, PollingTask


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class ScriptResult(BaseModel):
    """LLM 生成的视频脚本结构。"""

    scenes: list[str]
    summary: str
    memory_update: str | None = None  # 框架自动写入 memory.md


# ---------------------------------------------------------------------------
# Mock 函数（实际使用时替换为真实 API 调用）
# ---------------------------------------------------------------------------


def submit_video_generation(results: list) -> str:
    """提交视频生成任务，返回 job handle。"""
    script: ScriptResult = results[0].output
    print(f"  [video] Submitting video job with {len(script.scenes)} scenes")
    return "video-job-handle-001"


def query_video_status(handle: str) -> dict:
    """查询视频生成状态。"""
    print(f"  [video] Querying status for {handle}")
    return {"status": "Success", "url": "https://example.com/video.mp4"}


def submit_tts(results: list) -> str:
    """提交 TTS 配音任务，返回 job handle。"""
    script: ScriptResult = results[0].output
    text = " ".join(script.scenes)
    print(f"  [tts] Submitting TTS for {len(text)} chars")
    return "tts-job-handle-001"


def query_tts_status(handle: str) -> dict:
    """查询 TTS 状态。"""
    print(f"  [tts] Querying status for {handle}")
    return {"status": "done", "audio_url": "https://example.com/audio.mp3"}


def ffmpeg_merge(results: list) -> str:
    """合并视频和音频（模拟 ffmpeg 调用）。"""
    video_result = next(r for r in results if r.task_type == "polling" and "video" in str(r.output))
    print(f"  [ffmpeg] Merging video and audio...")
    return "output.mp4"


# ---------------------------------------------------------------------------
# Pipeline 定义
# ---------------------------------------------------------------------------


async def main() -> None:
    h = Harness(
        project_path=".",
        system_prompt="你是一个专业的短视频内容创作者。请用中文回答。",
        stream_callback=lambda text: print(text, end="", flush=True),
    )

    pipeline_tasks = [
        # 步骤 0：LLM 生成脚本（需要 claude CLI）
        # 在示例中使用 FunctionTask 模拟，避免依赖 Claude CLI
        FunctionTask(
            fn=lambda r: ScriptResult(
                scenes=["开场白", "主体内容", "结尾升华"],
                summary="三幕式短视频脚本",
                memory_update="用户偏好：三幕结构，简洁有力",
            )
        ),

        # 步骤 1：并行生成视频和音频（模拟 PollingTask）
        Parallel(
            tasks=[
                PollingTask(
                    submit_fn=submit_video_generation,
                    poll_fn=query_video_status,
                    success_condition=lambda r: r.get("status") == "Success",
                    poll_interval=0.01,  # 示例中用极短间隔
                    timeout=30,
                    config=TaskConfig(max_retries=1),
                ),
                PollingTask(
                    submit_fn=submit_tts,
                    poll_fn=query_tts_status,
                    success_condition=lambda r: r.get("status") == "done",
                    poll_interval=0.01,
                    timeout=30,
                    config=TaskConfig(max_retries=1),
                ),
            ],
            error_policy="all_or_nothing",
        ),

        # 步骤 2：合并视频和音频
        FunctionTask(fn=ffmpeg_merge),

        # 步骤 3：发送桌面通知
        ShellTask(cmd="echo '视频生成完成：output.mp4'"),
    ]

    print("🎬 启动视频生成管道...")
    result = await h.pipeline(pipeline_tasks, name="video-demo")
    print(f"\n✅ 完成！Run ID: {result.run_id}")
    print(f"   总耗时: {result.total_duration_seconds:.2f}s")
    print(f"   总 tokens: {result.total_tokens}")
    print(f"   步骤数: {len(result.results)}")


if __name__ == "__main__":
    asyncio.run(main())

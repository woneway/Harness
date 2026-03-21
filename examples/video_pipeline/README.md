# video_pipeline — 视频生成管道示例（mock）

演示 Harness 并发编排能力：`LLMTask → Parallel[PollingTask × 2] → FunctionTask → ShellTask`。

## Pipeline 结构

```
FunctionTask [生成脚本]           — mock：返回三幕式脚本结构
    ↓
Parallel [
  PollingTask [视频生成]          — 提交任务 → 轮询状态 → 返回视频 URL
  PollingTask [TTS 配音]          — 提交任务 → 轮询状态 → 返回音频 URL
]
    ↓
FunctionTask [合并]               — mock ffmpeg 合并视频+音频
    ↓
ShellTask [通知]                  — echo 完成提示
```

> 当前所有 API 调用均为 mock，替换 `submit_video_generation` / `submit_tts` 等函数即可对接真实服务（如 Hailuo、MiniMax TTS）。

## 运行

```bash
uv run python examples/video_pipeline/main.py
```

## 输出示例

```
🎬 启动视频生成管道...
  [video] Submitting video job with 3 scenes
  [tts] Submitting TTS for 15 chars
  [video] Querying status for video-job-handle-001
  [tts] Querying status for tts-job-handle-001
  [ffmpeg] Merging video and audio...

✅ 完成！Run ID: a1b2c3d4
   总耗时: 0.05s
   步骤数: 4
```

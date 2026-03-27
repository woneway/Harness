"""Discussion position schema — 每个 Agent 对项目的评估立场。"""

from pydantic import BaseModel


class ProjectEvaluation(BaseModel):
    """Discussion 中每个 Agent 每轮输出的结构化立场。"""

    project_name: str           # 被评估的项目名
    score: float                # 0-10 综合评分
    strengths: list[str]        # 优势（最多 3 条）
    weaknesses: list[str]       # 劣势（最多 3 条）
    best_for: str               # 最适合的场景（一句话）
    recommendation: str         # "推荐" | "可选" | "不推荐"

"""输入解析：判断用户输入类型，标准化为调研目标。

三种输入模式：
- project:    单个项目名或 GitHub URL（如 "langchain" 或 "https://github.com/langchain-ai/langchain"）
- comparison: 多个项目对比（如 "langchain vs llamaindex vs haystack"）
- topic:      调研主题（如 "Python AI 编排框架"）
"""

from __future__ import annotations

import json
import re
from typing import Any

from harness import State


_GITHUB_URL_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/\s]+)"
)
# owner/repo 格式（如 "cft0808/edict" 或 "langchain-ai/langchain"）
_OWNER_REPO_RE = re.compile(
    r"(?:^|\s)(?P<owner>[A-Za-z0-9._-]+)/(?P<repo>[A-Za-z0-9._-]+)(?:\s|$)"
)
_VS_RE = re.compile(r"\bvs\.?\b", re.IGNORECASE)


def parse_input(state: State) -> str:
    """解析 raw_input，填充 input_type / target_project / target_url / competitors。

    Returns:
        JSON 字符串（方便调试），同时通过 state 副作用设置字段。
    """
    raw = state.raw_input.strip()
    result: dict[str, Any] = {}

    # 情况 1: 包含 "vs" 分隔符 → comparison 模式
    if _VS_RE.search(raw):
        parts = _VS_RE.split(raw)
        projects = [p.strip() for p in parts if p.strip()]
        state.input_type = "comparison"
        state.target_project = projects[0]
        state.competitors = [
            {"name": p, "url": "", "description": ""} for p in projects[1:]
        ]
        result = {
            "input_type": "comparison",
            "target": projects[0],
            "competitors": projects[1:],
        }
        return json.dumps(result, ensure_ascii=False)

    # 情况 2a: 包含 GitHub URL → project 模式
    m = _GITHUB_URL_RE.search(raw)
    if m:
        owner, repo = m.group("owner"), m.group("repo")
        state.input_type = "project"
        state.target_project = repo
        state.target_url = f"https://github.com/{owner}/{repo}"
        result = {
            "input_type": "project",
            "target": repo,
            "url": state.target_url,
        }
        return json.dumps(result, ensure_ascii=False)

    # 情况 2b: 包含 owner/repo 格式 → project 模式
    m = _OWNER_REPO_RE.search(raw)
    if m:
        owner, repo = m.group("owner"), m.group("repo")
        state.input_type = "project"
        state.target_project = repo
        state.target_url = f"{owner}/{repo}"
        result = {
            "input_type": "project",
            "target": repo,
            "url": state.target_url,
        }
        return json.dumps(result, ensure_ascii=False)

    # 情况 3: 单个词/短语且看起来像项目名（无空格或只有少量词）
    words = raw.split()
    if len(words) <= 2 and raw.isascii():
        state.input_type = "project"
        state.target_project = raw.lower().replace(" ", "-")
        result = {"input_type": "project", "target": state.target_project}
        return json.dumps(result, ensure_ascii=False)

    # 情况 4: 其他 → topic 模式
    state.input_type = "topic"
    state.target_project = ""
    result = {"input_type": "topic", "topic": raw}
    return json.dumps(result, ensure_ascii=False)

"""GitHub 定量数据采集：通过 gh CLI 获取项目指标。"""

from __future__ import annotations

import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from harness import State

logger = logging.getLogger(__name__)


def _fetch_repo_metrics(owner_repo: str) -> dict:
    """调用 gh api 获取单个 repo 的指标（同步）。"""
    cmd = [
        "gh", "api", f"repos/{owner_repo}",
        "--jq",
        "{ full_name, description, homepage, "
        "stars: .stargazers_count, forks: .forks_count, "
        "open_issues: .open_issues_count, language, "
        "license: .license.spdx_id, "
        "created_at, pushed_at: .pushed_at, topics }",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        logger.warning("gh api timed out for %s", owner_repo)
        return {"error": f"Timeout fetching {owner_repo}"}

    if proc.returncode != 0:
        logger.warning("gh api failed for %s: %s", owner_repo, proc.stderr[:200])
        return {"error": f"Failed to fetch {owner_repo}", "stderr": proc.stderr[:200]}

    return json.loads(proc.stdout)


def _extract_owner_repo(project_name: str, url: str) -> str | None:
    """从 URL 提取 owner/repo。支持完整 URL 和 owner/repo 格式。"""
    import re

    if not url:
        return None

    # 完整 URL: https://github.com/owner/repo
    if "github.com/" in url:
        return url.rstrip("/").split("github.com/")[1]

    # owner/repo 格式（如 "langchain-ai/langchain"）
    if re.match(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$", url.strip()):
        return url.strip()

    return None


def collect_github(state: State) -> str:
    """采集所有项目的 GitHub 指标，写入 state.github_metrics。

    需要 state.target_project / target_url / competitors 已填充。
    """
    projects_to_fetch: list[tuple[str, str]] = []  # [(display_name, owner/repo)]

    # 主项目
    owner_repo = _extract_owner_repo(state.target_project, state.target_url)
    if owner_repo:
        projects_to_fetch.append((state.target_project, owner_repo))

    # 竞品
    for comp in state.competitors:
        comp_or = _extract_owner_repo(comp.get("name", ""), comp.get("url", ""))
        if comp_or:
            projects_to_fetch.append((comp["name"], comp_or))

    # 并发采集（线程池）
    metrics: dict = {}
    if projects_to_fetch:
        with ThreadPoolExecutor(max_workers=5) as pool:
            future_to_name = {
                pool.submit(_fetch_repo_metrics, or_): name
                for name, or_ in projects_to_fetch
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    metrics[name] = future.result()
                except Exception as e:
                    metrics[name] = {"error": str(e)}

    state.github_metrics = metrics
    return json.dumps(metrics, ensure_ascii=False, indent=2)

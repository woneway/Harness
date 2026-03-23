"""SessionManager — 为 pipeline 内所有 LLMTask 维护共享 session_id。"""

from __future__ import annotations


class SessionManager:
    """管理 pipeline 内的 Claude Code session ID。

    正常执行时所有 LLMTask 复用同一 session_id。
    重试或续跑时清空 session_id，让 Claude CLI 开启全新 session。
    """

    def __init__(self, initial_session_id: str | None = None) -> None:
        self._session_id: str | None = initial_session_id
        self._broken = False

    @property
    def current_session_id(self) -> str | None:
        """当前有效的 session ID。"""
        return self._session_id

    def update(self, session_id: str | None) -> None:
        """Runner 执行后更新 session ID（从 RunnerResult 中获取）。"""
        if session_id is not None:
            self._session_id = session_id
            self._broken = False

    def mark_broken(self) -> None:
        """标记当前 session 已断开，下次调用将开启全新 session。"""
        self._broken = True
        self._session_id = None

    @property
    def is_broken(self) -> bool:
        """是否处于断开后的新 session 状态。"""
        return self._broken

    def reset(self) -> None:
        """重置为全新 session（用于续跑场景）。"""
        self._broken = True
        self._session_id = None

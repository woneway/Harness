"""AbstractNotifier — 通知器抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractNotifier(ABC):
    """通知器抽象接口。"""

    @abstractmethod
    async def notify(
        self,
        title: str,
        body: str,
        *,
        success: bool,
    ) -> None:
        """发送通知。

        Args:
            title: 通知标题。
            body: 通知正文。
            success: True 表示成功，False 表示失败。
        """

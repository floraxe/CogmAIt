from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from app.domain.session_context import SessionContext


class ShortTermMemoryStore(ABC):
    """短期记忆接口：负责单轮会话上下文拼装。"""

    @abstractmethod
    def add(self, message: Dict[str, str]) -> None:
        pass

    @abstractmethod
    def prepend(self, message: Dict[str, str]) -> None:
        pass

    @abstractmethod
    def snapshot(self) -> List[Dict[str, str]]:
        pass


class LongTermMemoryStore(ABC):
    """长期记忆接口：用于可插拔外部记忆源（DB/向量库）。"""

    @abstractmethod
    def recall(self, user_id: Optional[str], query: str) -> List[Dict[str, str]]:
        pass


class SessionShortTermMemory(ShortTermMemoryStore):
    """基于 SessionContext 的短期记忆实现。"""

    def __init__(self, max_messages: int = 500) -> None:
        self._context = SessionContext(max_messages=max_messages)

    def add(self, message: Dict[str, str]) -> None:
        self._context.add_message(message)

    def prepend(self, message: Dict[str, str]) -> None:
        self._context.prepend_message(message)

    def snapshot(self) -> List[Dict[str, str]]:
        return self._context.get_messages()


class NullLongTermMemory(LongTermMemoryStore):
    """空实现：默认不注入长期记忆。"""

    def recall(self, user_id: Optional[str], query: str) -> List[Dict[str, str]]:
        return []


class MemoryManager:
    """统一记忆编排器，隔离 agents.py 中的消息状态管理细节。"""

    def __init__(
        self,
        short_term: Optional[ShortTermMemoryStore] = None,
        long_term: Optional[LongTermMemoryStore] = None,
    ) -> None:
        self.short_term = short_term or SessionShortTermMemory()
        self.long_term = long_term or NullLongTermMemory()

    def add_system_prompt(self, prompt: str) -> None:
        self.short_term.add({"role": "system", "content": prompt})

    def prepend_context(self, prompt: str) -> None:
        self.short_term.prepend({"role": "system", "content": prompt})

    def add_web_context(self, prompt: str) -> None:
        self.short_term.add({"role": "system", "content": prompt})

    def add_history(self, messages: List[Dict[str, str]]) -> None:
        for message in messages:
            self.short_term.add(message)

    def add_tool_result(self, prompt: str) -> None:
        self.short_term.add({"role": "system", "content": prompt})

    def add_long_term_recall(self, user_id: Optional[str], query: str) -> None:
        recalled = self.long_term.recall(user_id=user_id, query=query)
        for message in recalled:
            self.short_term.add(message)

    def messages(self) -> List[Dict[str, str]]:
        return self.short_term.snapshot()

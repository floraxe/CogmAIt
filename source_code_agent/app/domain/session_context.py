from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Optional, Set


class SessionContext:
    """
    Abstract Function (AF):
        SessionContext 表示一次智能体对话中可安全传递给模型的短期上下文消息序列。
        它封装了 messages 的构建过程，负责把 system/user/assistant 消息按顺序收集起来，
        并以只读拷贝形式暴露给调用方，避免外部代码直接污染内部状态。

    Representation Invariant (RI):
        1) _messages 是一个列表，且列表长度不超过 _max_messages。
        2) _messages 中每个元素必须是 dict，并且至少包含 role/content 字段。
        3) role 只能是 _allowed_roles 中的值（默认: system, user, assistant）。
        4) content 必须是字符串（允许空字符串，由上游业务决定是否过滤空消息）。
        5) _max_messages 必须是正整数。
        6) _allowed_roles 必须是非空集合，且元素全部为字符串。
    """

    def __init__(
        self,
        messages: Optional[Iterable[Dict[str, Any]]] = None,
        max_messages: int = 50,
        allowed_roles: Optional[Set[str]] = None,
    ) -> None:
        self._max_messages = max_messages
        self._allowed_roles = allowed_roles or {"system", "user", "assistant"}
        self._messages = []

        if messages:
            for message in messages:
                self.add_message(message)
        self._check_rep()

    def add_message(self, message: Dict[str, Any]) -> None:
        # 防御式写入：先复制外部入参，避免共享引用。
        copied_message = copy.deepcopy(message)
        normalized = self._normalize_message(copied_message)
        if len(self._messages) >= self._max_messages:
            raise ValueError(f"消息数量超过上限: {self._max_messages}")
        self._messages.append(normalized)
        self._check_rep()

    def add_messages(self, messages: Iterable[Dict[str, Any]]) -> None:
        for message in messages:
            self.add_message(message)

    def prepend_message(self, message: Dict[str, Any]) -> None:
        copied_message = copy.deepcopy(message)
        normalized = self._normalize_message(copied_message)
        if len(self._messages) >= self._max_messages:
            raise ValueError(f"消息数量超过上限: {self._max_messages}")
        self._messages.insert(0, normalized)
        self._check_rep()

    def get_messages(self) -> list[Dict[str, str]]:
        # 防御式返回，避免外部拿到引用后直接修改内部状态。
        return copy.deepcopy(self._messages)

    def _normalize_message(self, message: Dict[str, Any]) -> Dict[str, str]:
        if not isinstance(message, dict):
            raise TypeError("message 必须是 dict")

        role = message.get("role")
        content = message.get("content")

        if role not in self._allowed_roles:
            raise ValueError(f"不支持的 role: {role}")
        if not isinstance(content, str):
            raise TypeError("message.content 必须是字符串")

        return {"role": role, "content": content}

    def _check_rep(self) -> None:
        if not isinstance(self._max_messages, int) or self._max_messages <= 0:
            raise AssertionError("RI 违背: _max_messages 必须为正整数")
        if not isinstance(self._allowed_roles, set) or not self._allowed_roles:
            raise AssertionError("RI 违背: _allowed_roles 必须为非空集合")
        if not all(isinstance(role, str) for role in self._allowed_roles):
            raise AssertionError("RI 违背: _allowed_roles 元素必须是字符串")
        if len(self._messages) > self._max_messages:
            raise AssertionError("RI 违背: _messages 长度超出上限")

        for msg in self._messages:
            if not isinstance(msg, dict):
                raise AssertionError("RI 违背: message 不是 dict")
            if "role" not in msg or "content" not in msg:
                raise AssertionError("RI 违背: message 缺少 role/content")
            if msg["role"] not in self._allowed_roles:
                raise AssertionError("RI 违背: role 非法")
            if not isinstance(msg["content"], str):
                raise AssertionError("RI 违背: content 不是字符串")

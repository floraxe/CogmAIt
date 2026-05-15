import json
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StreamChunkAdapter(ABC):
    @abstractmethod
    def supports(self, chunk: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def normalize(self, chunk: Any) -> Tuple[str, Any, str]:
        raise NotImplementedError


class OpenAICompatibleStreamAdapter(StreamChunkAdapter):
    """适配 OpenAI 风格 choices/delta 流式块。"""

    def supports(self, chunk: Any) -> bool:
        return isinstance(chunk, dict) and bool(chunk.get("choices"))

    def normalize(self, chunk: Any) -> Tuple[str, Any, str]:
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        if delta.get("tool_calls"):
            return "tool_calls", chunk, ""
        if choice.get("finish_reason") == "tool_calls":
            return "tool_call_result", chunk, ""
        content = delta.get("content", "") or ""
        return "message_chunk", chunk, content


class JsonStringStreamAdapter(StreamChunkAdapter):
    def supports(self, chunk: Any) -> bool:
        return isinstance(chunk, str)

    def normalize(self, chunk: Any) -> Tuple[str, Any, str]:
        content = ""
        try:
            content = json.loads(chunk)["choices"][0]["delta"].get("content", "")
        except Exception:
            logger.debug("failed to parse string stream chunk", exc_info=True)
        return "message_chunk", chunk, content


class StreamChunkNormalizer:
    """按注册顺序尝试各适配器（适配器模式）。"""

    def __init__(self, adapters: Optional[List[StreamChunkAdapter]] = None) -> None:
        self._adapters = adapters or [
            OpenAICompatibleStreamAdapter(),
            JsonStringStreamAdapter(),
        ]

    def normalize(self, chunk: Any, provider_id: Optional[str] = None) -> Tuple[str, Any, str]:
        del provider_id  # 预留按 provider 选择适配器
        for adapter in self._adapters:
            if adapter.supports(chunk):
                return adapter.normalize(chunk)
        return "message_chunk", chunk, ""

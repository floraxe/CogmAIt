import logging
import traceback
from typing import Any, AsyncGenerator, Dict, Union

from app.services.inference.context import InferenceContext
from app.services.inference.strategies.base import ModelInferenceStrategy

logger = logging.getLogger(__name__)

_CHAT_PAYLOAD_KEYS = frozenset(
    {"messages", "temperature", "max_tokens", "model_type", "stream"}
)


class ChatInferenceStrategy(ModelInferenceStrategy):
    async def execute(
        self, context: InferenceContext
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        payload = context.payload
        model = context.model
        provider = context.provider

        if "messages" not in payload:
            return {"error": "请求中缺少必要参数: messages"}

        payload.setdefault("temperature", 0.7)
        payload.setdefault("max_tokens", 1000)

        try:
            stream = payload.get("stream", False)
            extra_kwargs = {
                k: v for k, v in payload.items() if k not in _CHAT_PAYLOAD_KEYS
            }
            response = await provider.chat_completion(
                api_key=model.api_key,
                base_url=model.base_url,
                model=model.name,
                messages=payload.get("messages", []),
                temperature=payload.get("temperature", 0.7),
                max_tokens=payload.get("max_tokens", 1000),
                stream=stream,
                **extra_kwargs,
            )
            if stream:
                logger.debug("chat stream response type: %s", type(response))
                return response
            return response
        except Exception as exc:
            logger.exception("chat inference failed model=%s", model.id)
            traceback.print_exc()
            return {"error": f"执行模型推理时出错: {str(exc)}"}

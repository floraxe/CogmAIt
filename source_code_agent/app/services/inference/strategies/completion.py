from typing import Any, Dict

from app.services.inference.context import InferenceContext
from app.services.inference.strategies.base import ModelInferenceStrategy


class CompletionInferenceStrategy(ModelInferenceStrategy):
    async def execute(self, context: InferenceContext) -> Dict[str, Any]:
        payload = context.payload
        model = context.model
        provider = context.provider

        prompt = payload.get("prompt", "")
        if not prompt:
            return {"error": "请求中缺少提示输入"}

        return await provider.text_completion(
            api_key=model.api_key,
            base_url=model.base_url,
            model_name=model.name,
            prompt=prompt,
            temperature=payload.get("temperature", 0.7),
            max_tokens=payload.get("max_tokens", 1000),
        )

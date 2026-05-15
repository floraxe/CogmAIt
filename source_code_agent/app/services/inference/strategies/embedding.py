import json
import logging
from typing import Any, Dict, List, Union

from app.services.inference.context import InferenceContext
from app.services.inference.strategies.base import ModelInferenceStrategy

logger = logging.getLogger(__name__)


def _extract_embeddings(embeddings_result: Any) -> List[List[float]]:
    embeddings: List[List[float]] = []
    if isinstance(embeddings_result, dict):
        if "embeddings" in embeddings_result:
            embeddings = embeddings_result.get("embeddings", [])
        elif "data" in embeddings_result:
            data = embeddings_result.get("data", [])
            if data and isinstance(data, list):
                embeddings = [
                    item.get("embedding", [])
                    for item in data
                    if isinstance(item, dict) and "embedding" in item
                ]
    elif isinstance(embeddings_result, list):
        embeddings = embeddings_result

    if not embeddings and isinstance(embeddings_result, dict) and "embeddings" in embeddings_result:
        embeddings = embeddings_result["embeddings"]
    return embeddings


class EmbeddingInferenceStrategy(ModelInferenceStrategy):
    async def execute(self, context: InferenceContext) -> Dict[str, Any]:
        payload = context.payload
        model = context.model
        provider = context.provider

        texts = payload.get("input", [])
        if not texts:
            return {"error": "请求中缺少文本输入"}

        embeddings_result = await provider.embedding(
            api_key=model.api_key,
            base_url=model.base_url,
            model=model.name,
            text=texts,
        )

        embeddings = _extract_embeddings(embeddings_result)
        if not embeddings:
            logger.warning(
                "embedding parse failed: %s",
                json.dumps(embeddings_result, ensure_ascii=False)[:500]
                if isinstance(embeddings_result, dict)
                else str(embeddings_result)[:500],
            )
            return {
                "error": "无法从API返回中提取embedding数据",
                "raw_response": str(embeddings_result)[:1000],
            }

        return {
            "success": True,
            "embeddings": embeddings,
            "model": model.name,
            "provider": model.provider,
            "dimensions": len(embeddings[0]) if embeddings else 0,
        }

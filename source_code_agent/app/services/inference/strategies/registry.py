from typing import Dict, Optional

from app.services.inference.strategies.base import ModelInferenceStrategy
from app.services.inference.strategies.chat import ChatInferenceStrategy
from app.services.inference.strategies.completion import CompletionInferenceStrategy
from app.services.inference.strategies.embedding import EmbeddingInferenceStrategy

_REGISTRY: Dict[str, ModelInferenceStrategy] = {
    "chat": ChatInferenceStrategy(),
    "embedding": EmbeddingInferenceStrategy(),
    "completion": CompletionInferenceStrategy(),
}


def get_inference_strategy(model_type: str) -> Optional[ModelInferenceStrategy]:
    return _REGISTRY.get(model_type)

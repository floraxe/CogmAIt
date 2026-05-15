from app.services.inference.strategies.base import ModelInferenceStrategy
from app.services.inference.strategies.chat import ChatInferenceStrategy
from app.services.inference.strategies.completion import CompletionInferenceStrategy
from app.services.inference.strategies.embedding import EmbeddingInferenceStrategy
from app.services.inference.strategies.registry import get_inference_strategy

__all__ = [
    "ModelInferenceStrategy",
    "ChatInferenceStrategy",
    "EmbeddingInferenceStrategy",
    "CompletionInferenceStrategy",
    "get_inference_strategy",
]

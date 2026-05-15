from dataclasses import dataclass
from typing import Any, Dict

from app.models.model import Model
from app.providers.base import ModelProvider


@dataclass
class InferenceContext:
    model: Model
    provider: ModelProvider
    payload: Dict[str, Any]
    model_type: str

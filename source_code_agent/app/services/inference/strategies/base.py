from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Union

from app.services.inference.context import InferenceContext


class ModelInferenceStrategy(ABC):
    @abstractmethod
    async def execute(
        self, context: InferenceContext
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        raise NotImplementedError

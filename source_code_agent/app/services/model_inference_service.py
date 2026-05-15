from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.services.inference.facade import ModelInferenceFacade, model_inference_facade
from app.services.inference.stream_adapter import StreamChunkNormalizer


class ModelInferenceService:
    def __init__(
        self,
        facade: Optional[ModelInferenceFacade] = None,
        chunk_normalizer: Optional[StreamChunkNormalizer] = None,
    ) -> None:
        self._facade = facade or model_inference_facade
        self._chunk_normalizer = chunk_normalizer or StreamChunkNormalizer()

    @staticmethod
    def build_stream_payload(messages: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": messages, "stream": True, **config}

    async def run_stream(self, db: Session, model_id: str, payload: Dict[str, Any]):
        return await self._facade.run(db, model_id, payload)

    def normalize_stream_chunk(
        self,
        chunk: Any,
        provider_id: Optional[str] = None,
    ) -> Tuple[str, Any, str]:
        return self._chunk_normalizer.normalize(chunk, provider_id)

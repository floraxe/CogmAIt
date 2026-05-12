from typing import Any, Dict, List, Tuple

from sqlalchemy.orm import Session

from app.utils.model import execute_model_inference


class ModelInferenceService:
    @staticmethod
    def build_stream_payload(messages: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": messages, "stream": True, **config}

    @staticmethod
    async def run_stream(db: Session, model_id: str, payload: Dict[str, Any]):
        return await execute_model_inference(db, model_id, payload)

    @staticmethod
    def normalize_stream_chunk(chunk: Any) -> Tuple[str, Any, str]:
        if isinstance(chunk, dict) and chunk.get("choices"):
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            if delta.get("tool_calls"):
                return "tool_calls", chunk, ""
            if choice.get("finish_reason") == "tool_calls":
                return "tool_call_result", chunk, ""
            content = delta.get("content", "") or ""
            return "message_chunk", chunk, content

        content = ""
        if isinstance(chunk, str):
            try:
                import json
                content = json.loads(chunk)["choices"][0]["delta"].get("content", "")
            except Exception:
                content = ""
        return "message_chunk", chunk, content

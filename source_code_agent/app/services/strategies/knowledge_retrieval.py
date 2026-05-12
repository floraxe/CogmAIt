"""
知识库检索策略。

完全自包含：不依赖 RetrievalAugmentationService，
直接调用 get_knowledge / EmbeddingManager / execute_model_inference。
"""
import logging
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.services.strategy_base import BaseRetrievalStrategy, StrategyContext, StrategyResult
from app.services import chat_events as ev
from app.utils.knowledge import get_knowledge, get_knowledge_file
from app.utils.embedding import EmbeddingManager
from app.utils.model import execute_model_inference

logger = logging.getLogger(__name__)


class KnowledgeRetrievalStrategy(BaseRetrievalStrategy):
    def is_active(self, agent: Any) -> bool:
        return bool(getattr(agent, "knowledge_bases", None))

    async def execute(self, context: StrategyContext) -> StrategyResult:
        result = StrategyResult()
        result.events.append(ev.knowledge_search_start().to_dict())

        similarity_threshold = context.config.get("similarity_threshold", 0.7)
        top_k = context.config.get("top_k", 5)
        retrieval_results: List[Dict[str, Any]] = []
        total_kb = len(context.agent.knowledge_bases)

        for idx, kb in enumerate(context.agent.knowledge_bases, start=1):
            result.events.append(ev.knowledge_progress(idx, total_kb).to_dict())

            knowledge = get_knowledge(context.db, kb.id)
            if not knowledge or not knowledge.embedding_model:
                continue

            result.events.append(ev.embedding_start().to_dict())
            embedding_resp = await execute_model_inference(
                context.db,
                knowledge.embedding_model,
                {"input": [context.user_message], "model_type": "embedding"},
            )
            if "error" in embedding_resp:
                result.events.append(ev.embedding_error().to_dict())
                continue

            embeddings = embedding_resp.get("embeddings", [])
            if not embeddings:
                continue

            vector_store = EmbeddingManager.get_vector_store()
            if not vector_store or vector_store.client is None:
                result.events.append(ev.vector_store_unavailable().to_dict())
                continue

            result.events.append(ev.vector_search_start().to_dict())
            hits = vector_store.search_similar(
                knowledge_id=kb.id,
                query_vector=embeddings[0],
                limit=top_k,
                filter_expr=None,
            )
            for hit in hits:
                score = max(0.0, min(1.0, hit.get("score", 0)))
                if score < similarity_threshold:
                    continue
                file_info = get_knowledge_file(context.db, hit.get("file_id", ""))
                file_name = file_info.original_filename if file_info else "未知文件"
                retrieval_results.append({
                    "content": hit.get("text", "")[:512] + "...",
                    "score": score,
                    "source_file": file_name,
                    "file_id": hit.get("file_id", ""),
                    "knowledge_id": kb.id,
                    "knowledge_name": knowledge.name,
                    "chunk_id": hit.get("chunk_index", 0),
                    "type": "document",
                })

        if retrieval_results:
            context.memory.add_knowledge_context(
                "\n".join(f"[{r['knowledge_name']}] {r['content']}" for r in retrieval_results)
            )
            result.sources.extend(retrieval_results)

        result.events.append(ev.vector_search_complete(retrieval_results).to_dict())
        return result

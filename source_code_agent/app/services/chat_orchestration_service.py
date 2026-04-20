from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os
import tempfile

from sqlalchemy.orm import Session

from app.models.file import File as FileModel
from app.utils.file_processor import extract_text_from_file_path
from app.core.minio_client import get_file_stream
from app.utils.model import execute_model_inference
from app.utils.web_search import search_web, get_web_search_client
from app.utils.knowledge import get_knowledge, get_knowledge_file
from app.utils.embedding import EmbeddingManager


@dataclass
class DocumentContextResult:
    formatted_contexts: List[str] = field(default_factory=list)
    processed_messages: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)


class DocumentContextService:
    def __init__(self, max_content_length: int = 3000):
        self.max_content_length = max_content_length

    async def process_files(self, db: Session, file_ids: List[str]) -> DocumentContextResult:
        result = DocumentContextResult()
        for file_id in file_ids:
            try:
                file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
                if not file_record:
                    result.error_messages.append(f"文件不存在: {file_id}")
                    continue

                if file_record.status != "processed":
                    if file_record.status == "processing":
                        result.error_messages.append(f"文件 {file_record.original_filename} 正在处理中，请稍后再试")
                    else:
                        result.error_messages.append(
                            f"文件 {file_record.original_filename} 未处理完成，状态: {file_record.status}"
                        )
                    continue

                file_content = await self._get_file_content(file_record)
                if not file_content:
                    result.error_messages.append(f"获取文件 {file_record.original_filename} 内容时出错")
                    continue

                if len(file_content) > self.max_content_length:
                    file_content = (
                        file_content[: self.max_content_length]
                        + f"\n\n[内容过长，已截断。原文共 {len(file_content)} 字符]"
                    )

                formatted_content = (
                    f"--- 文件: {file_record.original_filename} ({file_record.file_type}) ---\n\n"
                    f"{file_content}\n\n"
                )
                result.formatted_contexts.append(formatted_content)
                result.processed_messages.append(f"已处理文件: {file_record.original_filename}")
            except Exception as exc:
                result.error_messages.append(f"处理文件ID {file_id} 时出错: {str(exc)}")

        return result

    async def load_plain_text_contexts(self, db: Session, file_ids: List[str]) -> List[str]:
        contexts: List[str] = []
        for file_id in file_ids:
            file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
            if not file_record or not file_record.text_content:
                continue
            contexts.append(
                f"--- 文件: {file_record.original_filename} ({file_record.file_type}) ---\n\n"
                f"{file_record.text_content}\n\n"
            )
        return contexts

    @staticmethod
    def build_system_context(formatted_contexts: List[str]) -> Optional[str]:
        if not formatted_contexts:
            return None
        combined_content = "\n".join(formatted_contexts)
        return (
            "以下是用户上传的文件内容，这是非常重要的上下文信息，请务必仔细阅读并在回答问题时参考这些内容。"
            "如果用户询问关于文件内容的问题，请直接基于这些内容回答：\n\n"
            f"{combined_content}"
        )

    async def _get_file_content(self, file_record: FileModel) -> str:
        if file_record.text_content:
            return file_record.text_content

        if not file_record.path or "/" not in file_record.path:
            return ""

        bucket, object_name = file_record.path.split("/", 1)
        response = get_file_stream(bucket, object_name)
        if not response:
            return ""

        temp_file_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_record.file_type}") as temp_file:
                temp_file.write(response.read())
                temp_file_path = temp_file.name

            content = await extract_text_from_file_path(temp_file_path)
            if content and not content.startswith("提取文件文本内容时出错"):
                return content

            try:
                with open(temp_file_path, "r", encoding="utf-8") as file_obj:
                    return file_obj.read()
            except UnicodeDecodeError:
                with open(temp_file_path, "r", encoding="latin-1") as file_obj:
                    return file_obj.read()
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

        return ""


class ModelInferenceService:
    @staticmethod
    def build_stream_payload(messages: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "messages": messages,
            "stream": True,
            **config,
        }

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


@dataclass
class RetrievalAugmentationResult:
    events: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    web_search_results: List[Dict[str, Any]] = field(default_factory=list)


class RetrievalAugmentationService:
    async def run(
        self,
        memory: Any,
        db: Session,
        agent: Any,
        user_message: str,
        config: Dict[str, Any],
    ) -> RetrievalAugmentationResult:
        result = RetrievalAugmentationResult()
        await self._run_web_search(memory, agent, user_message, result)
        await self._run_knowledge_retrieval(memory, db, agent, user_message, config, result)
        return result

    async def _run_web_search(
        self,
        memory: Any,
        agent: Any,
        user_message: str,
        result: RetrievalAugmentationResult,
    ) -> None:
        if not getattr(agent, "enable_web_search", False):
            return

        result.events.append(
            {
                "event": "web_search",
                "data": {"object": "chat.completion.web_search", "status": "正在联网搜索最新信息"},
                "sleep": 0.5,
            }
        )
        try:
            search_results = await search_web(user_message)
            raw_results = search_results.get("results", [])
            if raw_results:
                result.web_search_results = raw_results
                web_search_context = get_web_search_client().format_search_results(search_results)
                memory.add_web_context(web_search_context)
                for item in raw_results:
                    result.sources.append(
                        {
                            "content": item.get("content", ""),
                            "score": 1.0,
                            "source_file": item.get("title", "网络搜索结果"),
                            "url": item.get("url", ""),
                            "type": "web_search",
                        }
                    )
                result.events.append(
                    {
                        "event": "web_search_complete",
                        "data": {
                            "object": "chat.completion.web_search_complete",
                            "status": "已找到相关信息",
                            "results_count": len(raw_results),
                            "webList": raw_results,
                        },
                        "sleep": 0.5,
                    }
                )
            else:
                result.events.append(
                    {
                        "event": "web_search_complete",
                        "data": {
                            "object": "chat.completion.web_search_complete",
                            "status": "未找到相关网络信息",
                            "results_count": 0,
                            "webList": [],
                        },
                        "sleep": 0.1,
                    }
                )
        except Exception as exc:
            result.events.append(
                {
                    "event": "web_search_complete",
                    "data": {
                        "object": "chat.completion.web_search_complete",
                        "status": "网络搜索过程中发生错误",
                        "error": str(exc),
                        "webList": [],
                    },
                    "sleep": 0.1,
                }
            )

    async def _run_knowledge_retrieval(
        self,
        memory: Any,
        db: Session,
        agent: Any,
        user_message: str,
        config: Dict[str, Any],
        result: RetrievalAugmentationResult,
    ) -> None:
        if not getattr(agent, "knowledge_bases", None):
            return

        result.events.append(
            {
                "event": "knowledge_search",
                "data": {"object": "chat.completion.knowledge_search", "status": "正在检索知识库相关内容"},
                "sleep": 0.1,
            }
        )

        similarity_threshold = config.get("similarity_threshold", 0.7)
        top_k = config.get("top_k", 5)
        retrieval_results: List[Dict[str, Any]] = []
        total_kb = len(agent.knowledge_bases)

        for idx, kb in enumerate(agent.knowledge_bases, start=1):
            result.events.append(
                {
                    "event": "knowledge_progress",
                    "data": {
                        "object": "chat.completion.knowledge_progress",
                        "status": "正在检索知识库",
                        "progress": f"{idx}/{total_kb}",
                    },
                    "sleep": 0.1,
                }
            )

            knowledge = get_knowledge(db, kb.id)
            if not knowledge or not knowledge.embedding_model:
                continue

            result.events.append(
                {
                    "event": "embedding",
                    "data": {"object": "chat.completion.embedding", "status": "正在计算语义向量"},
                    "sleep": 0.1,
                }
            )
            query_embedding_result = await execute_model_inference(
                db, knowledge.embedding_model, {"input": [user_message], "model_type": "embedding"}
            )
            if "error" in query_embedding_result:
                result.events.append(
                    {
                        "event": "embedding_error",
                        "data": {
                            "object": "chat.completion.embedding_error",
                            "status": "向量计算失败",
                            "error": query_embedding_result["error"],
                        },
                        "sleep": 0.1,
                    }
                )
                continue

            embeddings = query_embedding_result.get("embeddings", [])
            if not embeddings:
                continue
            query_embedding = embeddings[0]

            vector_store = EmbeddingManager.get_vector_store()
            if not vector_store or vector_store.client is None:
                result.events.append(
                    {
                        "event": "vector_store_error",
                        "data": {"object": "chat.completion.vector_store_error", "status": "向量存储服务不可用"},
                        "sleep": 0.1,
                    }
                )
                continue

            result.events.append(
                {
                    "event": "vector_search",
                    "data": {"object": "chat.completion.vector_search", "status": "正在检索相似文档"},
                    "sleep": 0.1,
                }
            )
            hits = vector_store.search_similar(
                knowledge_id=kb.id, query_vector=query_embedding, limit=top_k, filter_expr=None
            )
            for hit in hits:
                file_id = hit.get("file_id", "")
                file_info = get_knowledge_file(db, file_id)
                file_name = file_info.original_filename if file_info else "未知文件"
                score = hit.get("score", 0)
                score = max(0, min(1, score))
                if score < similarity_threshold:
                    continue
                retrieval_results.append(
                    {
                        "content": hit.get("text", "").replace(" ", "").replace("\n", "").replace("\\n", "")[:512] + "...",
                        "score": score,
                        "source_file": file_name.replace(" ", "").replace("\n", "").replace("\\n", ""),
                        "file_id": file_id,
                        "knowledge_id": kb.id,
                        "knowledge_name": knowledge.name.replace(" ", "").replace("\n", "").replace("\\n", ""),
                        "chunk_id": hit.get("chunk_index", 0),
                        "type": "document",
                    }
                )

        if retrieval_results:
            knowledge_context_lines = [
                f"[{item['knowledge_name']}] {item['content']}" for item in retrieval_results
            ]
            memory.add_knowledge_context("\n".join(knowledge_context_lines))
            result.sources.extend(retrieval_results)

        result.events.append(
            {
                "event": "vector_search_complete",
                "data": {
                    "object": "chat.completion.vector_search_complete",
                    "status": f"知识库检索完成，找到{len(retrieval_results)}条相关内容",
                    "results_count": len(retrieval_results),
                    "ragList": retrieval_results,
                },
                "sleep": 0.5,
            }
        )

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator
import os
import tempfile
import json
import time
import traceback
import re

from sqlalchemy.orm import Session

from app.models.file import File as FileModel
from app.models.agent import AgentShareToken
from app.domain.memory import MemoryManager
from app.utils.file_processor import extract_text_from_file_path
from app.core.minio_client import get_file_stream
from app.utils.model import execute_model_inference
from app.utils.web_search import search_web, get_web_search_client
from app.utils.knowledge import get_knowledge, get_knowledge_file
from app.utils.embedding import EmbeddingManager
from app.utils.llm_knowledge_extractor import LLMKnowledgeExtractor
from app.utils.neo4j_utils import get_neo4j_service
from app.utils.config import get_neo4j_config
from app.utils import agent as agent_utils
from app.services.strategy_base import (
    BaseRetrievalStrategy,
    StrategyContext,
    StrategyResult,
)


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

    async def run_web_search_only(
        self,
        memory: Any,
        agent: Any,
        user_message: str,
    ) -> RetrievalAugmentationResult:
        result = RetrievalAugmentationResult()
        await self._run_web_search(memory, agent, user_message, result)
        return result

    async def run_knowledge_retrieval_only(
        self,
        memory: Any,
        db: Session,
        agent: Any,
        user_message: str,
        config: Dict[str, Any],
    ) -> RetrievalAugmentationResult:
        result = RetrievalAugmentationResult()
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


@dataclass
class McpOrchestrationResult:
    events: List[Dict[str, Any]] = field(default_factory=list)
    tool_result_prompt: Optional[str] = None


class McpOrchestrationService:
    async def run(
        self,
        db: Session,
        agent: Any,
        user_message: str,
        model_id: str,
        current_user_id: str,
    ) -> McpOrchestrationResult:
        result = McpOrchestrationResult()
        mcp_service_list = list(getattr(agent, "mcp_services", []) or [])
        if not mcp_service_list:
            return result

        result.events.append(
            {
                "event": "mcp_processing",
                "data": {"object": "chat.completion.mcp_processing", "status": "正在处理MCP服务请求"},
                "sleep": 0.1,
            }
        )

        try:
            from app.utils.mcp import call_mcp_service, analyze_mcp_service_needs

            detection_result = await analyze_mcp_service_needs(
                db=db,
                model_id=model_id,
                user_message=user_message,
                available_services=mcp_service_list,
            )
            if not detection_result or not detection_result.get("call_mcp", False):
                reason = (detection_result or {}).get("reason", "无原因")
                if any(key in reason for key in ["无法解析", "解析错误", "解析失败", "解析JSON"]):
                    result.events.append(
                        {
                            "event": "mcp_error",
                            "data": {
                                "object": "chat.completion.mcp_error",
                                "error": f"解析MCP服务需求失败: {reason}",
                            },
                            "sleep": 0.1,
                        }
                    )
                return result

            service_id = detection_result.get("service_id")
            function_name = detection_result.get("function_name")
            params = detection_result.get("params", {})
            if not service_id or not function_name:
                result.events.append(
                    {
                        "event": "mcp_error",
                        "data": {
                            "object": "chat.completion.mcp_error",
                            "error": "MCP服务ID或函数名称为空",
                        },
                        "sleep": 0.1,
                    }
                )
                return result

            result.events.append(
                {
                    "event": "mcp_call",
                    "data": {
                        "object": "chat.completion.mcp_call",
                        "service_id": service_id,
                        "function_name": function_name,
                        "params": params,
                    },
                    "sleep": 0.1,
                }
            )

            service = next((svc for svc in mcp_service_list if svc.id == service_id), None)
            if not service:
                result.events.append(
                    {
                        "event": "mcp_error",
                        "data": {
                            "object": "chat.completion.mcp_error",
                            "error": f"请求的MCP服务ID {service_id} 不在当前智能体关联的服务列表中",
                        },
                        "sleep": 0.1,
                    }
                )
                return result

            mcp_result = await call_mcp_service(
                db=db,
                service_id=service_id,
                function_name=function_name,
                params=params,
                user_id=current_user_id,
            )
            service_name = service.name if hasattr(service, "name") else service.get("name", "Unknown")
            result.events.append(
                {
                    "event": "mcp_result",
                    "data": {
                        "object": "chat.completion.mcp_result",
                        "service": service_name,
                        "function": function_name,
                        "result": mcp_result,
                    },
                    "sleep": 0.1,
                }
            )
            result.tool_result_prompt = (
                f"以下是调用MCP服务 '{service_name}' 的函数 '{function_name}' 的结果，请使用这些结果回答用户的问题:\n\n"
                f"```json\n{json.dumps(mcp_result, ensure_ascii=False, indent=2)}\n```"
            )
            return result
        except Exception as exc:
            result.events.append(
                {
                    "event": "mcp_error",
                    "data": {"object": "chat.completion.mcp_error", "error": f"处理MCP服务时出错: {str(exc)}"},
                    "sleep": 0.1,
                }
            )
            return result


class ChatResponseService:
    @staticmethod
    async def ensure_file_guidance(
        memory: Any,
        final_messages: List[Dict[str, Any]],
        file_ids: List[str],
        db: Session,
        document_service: DocumentContextService,
        user_message: str,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        has_file_content = any(
            "以下是用户上传的文件内容" in msg.get("content", "")
            for msg in final_messages
            if msg.get("role") == "system"
        )
        if not has_file_content and file_ids:
            fallback_contexts = await document_service.load_plain_text_contexts(db, file_ids)
            fallback_system_context = document_service.build_system_context(fallback_contexts)
            if fallback_system_context:
                memory.prepend_context(fallback_system_context)
                has_file_content = True
                final_messages = memory.messages()

        if has_file_content and any(
            key in user_message
            for key in ["文档说的什么", "文档说了什么", "文件内容是什么", "文件说了什么", "文件说的什么"]
        ):
            memory.add_system_prompt(
                "用户正在询问文件内容。请直接回答文件的内容是什么，不要回避或者说找不到相关信息。文件内容已经在之前的系统消息中提供。"
            )
            final_messages = memory.messages()
        return final_messages, has_file_content

    @staticmethod
    def build_extra_data(
        response_time: int,
        used_tokens: int,
        sources: List[Dict[str, Any]],
        web_search_results: List[Dict[str, Any]],
        has_file_content: bool,
    ) -> Dict[str, Any]:
        extra_data: Dict[str, Any] = {
            "response_time_ms": response_time,
            "tokens_used": used_tokens,
        }
        if sources:
            extra_data["sources"] = sources
        if web_search_results:
            extra_data["web_results"] = web_search_results
        if has_file_content:
            extra_data["has_file_content"] = True
        return extra_data


class GraphRetrievalService:
    async def stream_graph_events(
        self,
        db: Session,
        agent: Any,
        user_message: str,
        model_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not getattr(agent, "graphs", None):
            return

        graph_list = {"nodes": [], "links": []}
        yield {
            "event": "graph_search",
            "data": '{"object": "chat.completion.graph_search", "status": "正在查询知识图谱"}',
            "sleep": 0.1,
        }
        try:
            neo4j_config = get_neo4j_config()
            yield {
                "event": "graph_connecting",
                "data": '{"object": "chat.completion.graph_connecting", "status": "正在连接知识图谱数据库"}',
                "sleep": 0.1,
            }
            neo4j_service = get_neo4j_service(
                uri=neo4j_config.get("uri"),
                username=neo4j_config.get("username"),
                password=neo4j_config.get("password"),
                database=neo4j_config.get("database"),
                force_new=True,
            )
            if not neo4j_service or not neo4j_service.driver or not neo4j_service.is_connected():
                yield {
                    "event": "graph_connecting_error",
                    "data": '{"object": "chat.completion.graph_connection_error", "status": "Neo4j服务初始化失败"}',
                    "sleep": 0.1,
                }
                return

            yield {
                "event": "graph_connected",
                "data": '{"object": "chat.completion.graph_connected", "status": "知识图谱数据库连接成功"}',
                "sleep": 0.1,
            }

            extractor = LLMKnowledgeExtractor(db=db)
            for gb in agent.graphs:
                graph = agent_utils.get_graph(db, gb.id)
                if not graph or not graph.neo4j_subgraph:
                    continue

                schema = None
                try:
                    from app.utils.graph import get_graph_schema

                    schema = get_graph_schema(db, graph.id)
                    yield {
                        "event": "graph_schema",
                        "data": '{"object": "chat.completion.graph_schema", "status": "已获取知识图谱结构定义"}',
                        "sleep": 0.1,
                    }
                except Exception as exc:
                    yield {
                        "event": "graph_schema_error",
                        "data": json.dumps(
                            {
                                "object": "chat.completion.graph_schema_error",
                                "status": "获取图谱结构定义失败",
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        ),
                        "sleep": 0.1,
                    }

                yield {
                    "event": "graph_analysis",
                    "data": '{"object": "chat.completion.graph_analysis", "status": "正在分析问题与知识图谱的关联"}',
                    "sleep": 0.1,
                }

                extraction_prompt = f"""
                请分析用户问题并给出Neo4j Cypher查询，仅返回JSON:
                {{
                  "cypher": "MATCH (n) RETURN n LIMIT 15"
                }}
                用户问题: "{user_message}"
                图谱Schema: {json.dumps(schema, ensure_ascii=False) if schema else "未定义schema"}
                """
                extraction_result = await execute_model_inference(
                    db,
                    model_id,
                    {
                        "messages": [
                            {"role": "system", "content": "你是一个生成Neo4j Cypher查询的助手。"},
                            {"role": "user", "content": extraction_prompt},
                        ],
                        "model_type": "chat",
                    },
                )

                yield {
                    "event": "graph_query_generated",
                    "data": '{"object": "chat.completion.graph_query_generated", "status": "已生成知识图谱查询语句"}',
                    "sleep": 0.1,
                }

                cypher_query = self._extract_cypher(extraction_result, user_message, graph.neo4j_subgraph)
                if not cypher_query:
                    yield {
                        "event": "graph_search_error",
                        "data": '{"object": "chat.completion.graph_search_error", "status": "无法生成有效的知识图谱查询"}',
                        "sleep": 0.1,
                    }
                    continue

                yield {
                    "event": "graph_search",
                    "data": '{"object": "chat.completion.graph_search", "status": "正在查询知识图谱..."}',
                    "sleep": 0.5,
                }
                try:
                    with neo4j_service.driver.session(database=neo4j_service.database) as session:
                        records = list(session.run(cypher_query))
                    if not records:
                        yield {
                            "event": "graph_search_complete",
                            "data": '{"object": "chat.completion.graph_search_complete", "status": "知识图谱中未找到相关信息", "graphList": {"nodes": [], "links": []}}',
                            "sleep": 0.1,
                        }
                        continue

                    graph_list = self._build_graph_list(records)
                    yield {
                        "event": "graph_search_complete",
                        "data": json.dumps(
                            {
                                "object": "chat.completion.graph_search_complete",
                                "status": "知识图谱搜索完成",
                                "graphList": graph_list,
                            },
                            ensure_ascii=False,
                        ),
                        "sleep": 0.5,
                    }
                except Exception as exc:
                    yield {
                        "event": "graph_search_error",
                        "data": json.dumps(
                            {
                                "object": "chat.completion.graph_search_error",
                                "status": "执行知识图谱查询失败",
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        ),
                        "sleep": 0.1,
                    }
            yield {
                "event": "graph_search_complete",
                "data": json.dumps(
                    {
                        "object": "chat.completion.graph_search_complete",
                        "status": "知识图谱查询完成",
                        "graphList": graph_list,
                    },
                    ensure_ascii=False,
                ),
                "sleep": 0.1,
            }
        except Exception as exc:
            yield {
                "event": "graph_search_error",
                "data": json.dumps(
                    {
                        "object": "chat.completion.graph_search_error",
                        "status": "知识图谱查询失败",
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                "sleep": 0.1,
            }

    async def run(self, db: Session, agent: Any, user_message: str, model_id: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        async for item in self.stream_graph_events(db=db, agent=agent, user_message=user_message, model_id=model_id):
            events.append(item)
        return events

    @staticmethod
    def _extract_cypher(extraction_result: Any, user_message: str, subgraph_name: str) -> Optional[str]:
        subgraph_id = (subgraph_name or "").lower().replace(" ", "_").replace("-", "_")
        message_content = ""
        if isinstance(extraction_result, dict):
            choices = extraction_result.get("choices") or []
            if choices:
                message_content = choices[0].get("message", {}).get("content", "")
        elif isinstance(extraction_result, str):
            message_content = extraction_result

        if message_content:
            code_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', message_content)
            try:
                payload = json.loads(code_match.group(1) if code_match else message_content)
                if isinstance(payload, dict) and payload.get("cypher"):
                    cypher = payload["cypher"]
                    return GraphRetrievalService._normalize_cypher(cypher, subgraph_id, user_message)
            except Exception:
                pass

            for pattern in [
                r'```cypher\s*(MATCH[\s\S]+?)\s*```',
                r'```\s*(MATCH[\s\S]+?)\s*```',
                r'(MATCH\s*\([^)]+\)[\s\S]+?RETURN[^;]+)',
            ]:
                matches = re.findall(pattern, message_content, re.IGNORECASE)
                if matches:
                    return GraphRetrievalService._normalize_cypher(matches[0].strip(), subgraph_id, user_message)

        fallback_entity = user_message.strip().replace("'", "")
        fallback = f"MATCH (n) WHERE n.graph_id = '{subgraph_id}' AND n.name CONTAINS '{fallback_entity}' RETURN n LIMIT 20"
        return fallback

    @staticmethod
    def _normalize_cypher(cypher_query: str, subgraph_id: str, user_message: str) -> str:
        cypher_query = re.sub(r"MATCH\s*\(\w+:\w+\)", "MATCH (n)", cypher_query)
        cypher_query = cypher_query.replace("...", "").strip().replace('"', "'").replace("''", "'")
        if "WHERE" in cypher_query.upper():
            where_pos = cypher_query.upper().find("WHERE")
            return_pos = cypher_query.upper().find("RETURN", where_pos)
            where_part = cypher_query[where_pos:return_pos] if return_pos != -1 else cypher_query[where_pos:]
            if "graph_id" not in where_part.lower():
                if return_pos != -1:
                    new_where = f"WHERE n.graph_id = '{subgraph_id}' AND " + where_part[5:].strip()
                    cypher_query = cypher_query[:where_pos] + new_where + cypher_query[return_pos:]
                else:
                    cypher_query = cypher_query.replace("WHERE", f"WHERE n.graph_id = '{subgraph_id}' AND ")
        else:
            return_pos = cypher_query.upper().find("RETURN")
            if return_pos > 0:
                cypher_query = cypher_query[:return_pos] + f" WHERE n.graph_id = '{subgraph_id}' " + cypher_query[return_pos:]
            else:
                entity = user_message.strip().replace("'", "")
                cypher_query = f"MATCH (n) WHERE n.graph_id = '{subgraph_id}' AND n.name CONTAINS '{entity}' RETURN n LIMIT 20"
        return cypher_query

    @staticmethod
    def _build_graph_list(records: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
        nodes: Dict[str, Dict[str, Any]] = {}
        links: Dict[str, Dict[str, Any]] = {}
        for record in records:
            for _, value in record.items():
                if value is None:
                    continue
                if hasattr(value, "id") and hasattr(value, "labels"):
                    node_id = str(value.id)
                    props = dict(value.properties) if hasattr(value, "properties") else {}
                    node_name = props.get("name") or props.get("title") or f"节点{node_id}"
                    node_type = list(value.labels)[0] if getattr(value, "labels", None) else "Entity"
                    nodes[node_id] = {
                        "id": node_id,
                        "name": str(node_name),
                        "symbolSize": 50,
                        "category": str(node_type),
                        "properties": props,
                    }
                elif hasattr(value, "type") and hasattr(value, "start_node") and hasattr(value, "end_node"):
                    start_id = str(value.start_node.id)
                    end_id = str(value.end_node.id)
                    rel_type = str(value.type)
                    link_id = f"{start_id}_{rel_type}_{end_id}"
                    links[link_id] = {
                        "source": start_id,
                        "target": end_id,
                        "value": rel_type,
                        "properties": dict(value.properties) if hasattr(value, "properties") else {},
                    }
        return {"nodes": list(nodes.values()), "links": list(links.values())}


class WebSearchStrategy(BaseRetrievalStrategy):
    def __init__(self, retrieval_service: RetrievalAugmentationService):
        self.retrieval_service = retrieval_service

    async def execute(self, context: StrategyContext) -> StrategyResult:
        result = await self.retrieval_service.run_web_search_only(
            memory=context.memory,
            agent=context.agent,
            user_message=context.user_message,
        )
        return StrategyResult(
            events=result.events,
            sources=result.sources,
            web_search_results=result.web_search_results,
        )


class KnowledgeRetrievalStrategy(BaseRetrievalStrategy):
    def __init__(self, retrieval_service: RetrievalAugmentationService):
        self.retrieval_service = retrieval_service

    async def execute(self, context: StrategyContext) -> StrategyResult:
        result = await self.retrieval_service.run_knowledge_retrieval_only(
            memory=context.memory,
            db=context.db,
            agent=context.agent,
            user_message=context.user_message,
            config=context.config,
        )
        return StrategyResult(
            events=result.events,
            sources=result.sources,
            web_search_results=result.web_search_results,
        )


class GraphRetrievalStrategy(BaseRetrievalStrategy):
    def __init__(self, graph_service: GraphRetrievalService):
        self.graph_service = graph_service

    async def execute(self, context: StrategyContext) -> StrategyResult:
        events = await self.graph_service.run(
            db=context.db,
            agent=context.agent,
            user_message=context.user_message,
            model_id=context.model_id,
        )
        return StrategyResult(events=events)


@dataclass
class ChatPipelineRequest:
    db: Session
    agent_id: str
    messages: List[Any]
    session_id: str
    config_override: Dict[str, Any]
    file_ids: List[str]
    current_user_id: str
    access_type: str
    share_token: Optional[str] = None
    api_key_id: Optional[str] = None


@dataclass
class ChatPipelineState:
    request: ChatPipelineRequest
    agent: Any = None
    model_id: str = ""
    user_message: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    response_content: str = ""
    used_tokens: int = 0
    sources: List[Dict[str, Any]] = field(default_factory=list)
    web_search_results: List[Dict[str, Any]] = field(default_factory=list)
    has_file_content: bool = False
    share_token_id: Optional[str] = None
    memory: Any = field(default_factory=lambda: None)
    final_messages: List[Dict[str, Any]] = field(default_factory=list)


class ChatPipelineOrchestrator:
    def __init__(self):
        self.document_service = DocumentContextService()
        self.inference_service = ModelInferenceService()
        self.retrieval_service = RetrievalAugmentationService()
        self.mcp_service = McpOrchestrationService()
        self.response_service = ChatResponseService()
        self.graph_service = GraphRetrievalService()

    async def stream(self, request: ChatPipelineRequest) -> AsyncGenerator[Dict[str, Any], None]:
        state = ChatPipelineState(request=request)
        try:
            yield from_event("status", {"object": "chat.completion.status", "status": "开始处理请求"})
            audit_events = await self._audit(state)
            for event in audit_events:
                yield event

            strategy_events = await self._run_strategies(state)
            for event in strategy_events:
                yield event

            async for event in self._run_inference(state):
                yield event

            filter_events = await self._run_filter(state)
            for event in filter_events:
                yield event
        except Exception as exc:
            traceback.print_exc()
            yield {
                "event": "error",
                "data": json.dumps({"error": f"生成响应时出错: {str(exc)}"}, ensure_ascii=False),
            }
            time.sleep(0.1)
            yield {"event": "done", "data": "[DONE]"}

    async def _audit(self, state: ChatPipelineState) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        request = state.request
        state.user_message = (
            request.messages[-1].content
            if request.messages and request.messages[-1].role == "user"
            else ""
        )
        state.agent = agent_utils.get_agent(request.db, request.agent_id)
        if not state.agent:
            raise ValueError("智能体不存在")
        if not state.user_message:
            raise ValueError("请求中缺少用户消息")

        state.model_id = state.agent.model_id
        if not state.model_id:
            raise ValueError("该智能体未关联模型，请先在智能体设置中关联一个对话模型")
        if not agent_utils.get_model(request.db, state.model_id):
            raise ValueError(f"模型不存在: {state.model_id}")

        state.config = {**(state.agent.config or {}), **request.config_override}
        state.start_time = time.time()
        state.memory = MemoryManager()
        state.final_messages = state.memory.messages()

        if request.file_ids:
            events.append(from_event("status", {"object": "chat.completion.status", "status": "正在处理上传文件"}, sleep=0.1))
            file_context_result = await self.document_service.process_files(request.db, request.file_ids)
            for status_msg in file_context_result.processed_messages:
                events.append(from_event("file_processing", {"status": status_msg}, sleep=0.1))
            for error_msg in file_context_result.error_messages:
                events.append(from_event("file_processing", {"status": error_msg}, sleep=0.1))
            file_system_context = self.document_service.build_system_context(file_context_result.formatted_contexts)
            if file_system_context:
                state.memory.prepend_context(file_system_context)
                state.final_messages = state.memory.messages()
                state.has_file_content = True

        if state.agent.system_prompt:
            state.memory.add_system_prompt(state.agent.system_prompt)
            state.final_messages = state.memory.messages()
        return flatten_events(events)

    async def _run_strategies(self, state: ChatPipelineState) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = [from_event("think", {"object": "chat.completion.think", "status": "AI开始思考该如何回答您的问题"}, sleep=0.1)]
        strategy_context = StrategyContext(
            memory=state.memory,
            db=state.request.db,
            agent=state.agent,
            user_message=state.user_message,
            model_id=state.model_id,
            config=state.config,
        )
        active_strategies: List[BaseRetrievalStrategy] = []
        if state.agent.enable_web_search:
            active_strategies.append(WebSearchStrategy(self.retrieval_service))
        if state.agent.knowledge_bases:
            active_strategies.append(KnowledgeRetrievalStrategy(self.retrieval_service))
        if state.agent.graphs:
            active_strategies.append(GraphRetrievalStrategy(self.graph_service))

        for strategy in active_strategies:
            strategy_result = await strategy.execute(strategy_context)
            for item in strategy_result.events:
                payload = item["data"] if isinstance(item["data"], str) else json.dumps(item["data"], ensure_ascii=False)
                events.append({"event": item["event"], "data": payload})
            if strategy_result.sources:
                state.sources.extend(strategy_result.sources)
            if strategy_result.web_search_results:
                state.web_search_results = strategy_result.web_search_results

        history_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in state.request.messages
            if msg.role in ["user", "assistant", "system"]
        ]
        state.memory.add_history(history_messages)
        state.final_messages = state.memory.messages()
        return events

    async def _run_inference(self, state: ChatPipelineState) -> AsyncGenerator[Dict[str, Any], None]:
        mcp_result = await self.mcp_service.run(
            db=state.request.db,
            agent=state.agent,
            user_message=state.user_message,
            model_id=state.model_id,
            current_user_id=state.request.current_user_id,
        )
        for event_item in mcp_result.events:
            yield {"event": event_item["event"], "data": json.dumps(event_item["data"], ensure_ascii=False)}
            time.sleep(event_item.get("sleep", 0.1))
        if mcp_result.tool_result_prompt:
            state.memory.add_tool_result(mcp_result.tool_result_prompt)
            state.final_messages = state.memory.messages()

        yield {"event": "reasoning", "data": '{"object": "chat.completion.reasoning", "status": "AI正在整合信息推理回答"}'}
        yield {
            "event": "info",
            "data": json.dumps(
                {"object": "chat.completion.info", "sources": state.sources, "web_search_results": state.web_search_results},
                ensure_ascii=False,
            ),
        }
        yield {"event": "answer", "data": '{"object": "chat.completion.answer", "status": "AI开始生成答案"}'}

        state.final_messages, state.has_file_content = await self.response_service.ensure_file_guidance(
            memory=state.memory,
            final_messages=state.final_messages,
            file_ids=state.request.file_ids,
            db=state.request.db,
            document_service=self.document_service,
            user_message=state.user_message,
        )

        payload = self.inference_service.build_stream_payload(state.final_messages, state.config)
        model_response = await self.inference_service.run_stream(state.request.db, state.model_id, payload)
        async for chunk in model_response:
            event_name, event_data, delta_content = self.inference_service.normalize_stream_chunk(chunk)
            if delta_content:
                state.response_content += delta_content
            yield {"event": event_name, "data": event_data}

    async def _run_filter(self, state: ChatPipelineState) -> List[Dict[str, Any]]:
        request = state.request
        if request.access_type == "share" and request.share_token:
            share_token_obj = request.db.query(AgentShareToken).filter(AgentShareToken.token == request.share_token).first()
            if share_token_obj:
                state.share_token_id = share_token_obj.id

        response_time = int((time.time() - state.start_time) * 1000)
        extra_data = self.response_service.build_extra_data(
            response_time=response_time,
            used_tokens=state.used_tokens,
            sources=state.sources,
            web_search_results=state.web_search_results,
            has_file_content=state.has_file_content,
        )
        try:
            agent_utils.create_chat_history(
                db=request.db,
                agent_id=request.agent_id,
                session_id=request.session_id,
                user_id=request.current_user_id,
                user_message=state.user_message,
                agent_response=state.response_content,
                tokens_used=state.used_tokens,
                response_time=response_time,
                extra_data=extra_data,
                access_type=request.access_type,
                api_key_id=request.api_key_id,
                share_token_id=state.share_token_id,
                model_id=state.model_id,
            )
        except Exception:
            traceback.print_exc()
        return [
            {"event": "status", "data": '{"object": "chat.completion.status", "status": "回答完成"}'},
            {"event": "done", "data": "[DONE]"},
        ]


def from_event(event: str, data: Dict[str, Any], sleep: float = 0.1) -> Dict[str, Any]:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    item = {"event": event, "data": payload}
    time.sleep(sleep)
    return item


def flatten_events(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in items if item]

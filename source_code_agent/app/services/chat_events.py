"""
SSE 事件工厂模块。

所有 SSE 事件在此统一构建，确保：
- 事件格式集中维护，前端协议改动只需修改此文件
- 错误事件不向客户端暴露原始异常信息
- 类型安全，方便 IDE 补全和静态检查
"""
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SseEvent:
    """单个 SSE 事件的类型化包装。"""

    __slots__ = ("event", "data")

    def __init__(self, event: str, data: Dict[str, Any]) -> None:
        self.event = event
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        return {"event": self.event, "data": json.dumps(self.data, ensure_ascii=False)}


# ── 通用状态 ──────────────────────────────────────────────────────────────────

def status(msg: str) -> SseEvent:
    return SseEvent("status", {"object": "chat.completion.status", "status": msg})


def think(msg: str = "AI开始思考该如何回答您的问题") -> SseEvent:
    return SseEvent("think", {"object": "chat.completion.think", "status": msg})


def reasoning() -> SseEvent:
    return SseEvent("reasoning", {"object": "chat.completion.reasoning", "status": "AI正在整合信息推理回答"})


def answer_start() -> SseEvent:
    return SseEvent("answer", {"object": "chat.completion.answer", "status": "AI开始生成答案"})


def done_signal() -> Dict[str, str]:
    return {"event": "done", "data": "[DONE]"}


def error(msg: str = "生成响应时出错") -> SseEvent:
    return SseEvent("error", {"error": msg})


def info(sources: List[Dict[str, Any]], web_results: List[Dict[str, Any]]) -> SseEvent:
    return SseEvent("info", {
        "object": "chat.completion.info",
        "sources": sources,
        "web_search_results": web_results,
    })


# ── 文件处理 ──────────────────────────────────────────────────────────────────

def file_processing(msg: str) -> SseEvent:
    return SseEvent("file_processing", {"object": "chat.completion.file_processing", "status": msg})


# ── 联网搜索 ──────────────────────────────────────────────────────────────────

def web_search_start() -> SseEvent:
    return SseEvent("web_search", {"object": "chat.completion.web_search", "status": "正在联网搜索最新信息"})


def web_search_complete(results: List[Dict[str, Any]]) -> SseEvent:
    return SseEvent("web_search_complete", {
        "object": "chat.completion.web_search_complete",
        "status": "已找到相关信息" if results else "未找到相关网络信息",
        "results_count": len(results),
        "webList": results,
    })


def web_search_error() -> SseEvent:
    return SseEvent("web_search_complete", {
        "object": "chat.completion.web_search_complete",
        "status": "网络搜索过程中发生错误",
        "results_count": 0,
        "webList": [],
    })


# ── 知识库检索 ────────────────────────────────────────────────────────────────

def knowledge_search_start() -> SseEvent:
    return SseEvent("knowledge_search", {"object": "chat.completion.knowledge_search", "status": "正在检索知识库相关内容"})


def knowledge_progress(current: int, total: int) -> SseEvent:
    return SseEvent("knowledge_progress", {
        "object": "chat.completion.knowledge_progress",
        "status": "正在检索知识库",
        "progress": f"{current}/{total}",
    })


def embedding_start() -> SseEvent:
    return SseEvent("embedding", {"object": "chat.completion.embedding", "status": "正在计算语义向量"})


def embedding_error() -> SseEvent:
    return SseEvent("embedding_error", {"object": "chat.completion.embedding_error", "status": "向量计算失败"})


def vector_search_start() -> SseEvent:
    return SseEvent("vector_search", {"object": "chat.completion.vector_search", "status": "正在检索相似文档"})


def vector_store_unavailable() -> SseEvent:
    return SseEvent("vector_store_error", {"object": "chat.completion.vector_store_error", "status": "向量存储服务不可用"})


def vector_search_complete(results: List[Dict[str, Any]]) -> SseEvent:
    return SseEvent("vector_search_complete", {
        "object": "chat.completion.vector_search_complete",
        "status": f"知识库检索完成，找到{len(results)}条相关内容",
        "results_count": len(results),
        "ragList": results,
    })


# ── 图谱检索 ──────────────────────────────────────────────────────────────────

def graph_search_start() -> SseEvent:
    return SseEvent("graph_search", {"object": "chat.completion.graph_search", "status": "正在查询知识图谱"})


def graph_connecting() -> SseEvent:
    return SseEvent("graph_connecting", {"object": "chat.completion.graph_connecting", "status": "正在连接知识图谱数据库"})


def graph_connection_error() -> SseEvent:
    return SseEvent("graph_connecting_error", {"object": "chat.completion.graph_connection_error", "status": "Neo4j服务初始化失败"})


def graph_connected() -> SseEvent:
    return SseEvent("graph_connected", {"object": "chat.completion.graph_connected", "status": "知识图谱数据库连接成功"})


def graph_schema_ready() -> SseEvent:
    return SseEvent("graph_schema", {"object": "chat.completion.graph_schema", "status": "已获取知识图谱结构定义"})


def graph_schema_error() -> SseEvent:
    return SseEvent("graph_schema_error", {"object": "chat.completion.graph_schema_error", "status": "获取图谱结构定义失败"})


def graph_analysis() -> SseEvent:
    return SseEvent("graph_analysis", {"object": "chat.completion.graph_analysis", "status": "正在分析问题与知识图谱的关联"})


def graph_query_generated() -> SseEvent:
    return SseEvent("graph_query_generated", {"object": "chat.completion.graph_query_generated", "status": "已生成知识图谱查询语句"})


def graph_query_invalid() -> SseEvent:
    return SseEvent("graph_search_error", {"object": "chat.completion.graph_search_error", "status": "无法生成有效的知识图谱查询"})


def graph_search_running() -> SseEvent:
    return SseEvent("graph_search", {"object": "chat.completion.graph_search", "status": "正在查询知识图谱..."})


def graph_search_complete(graph_list: Dict[str, Any]) -> SseEvent:
    return SseEvent("graph_search_complete", {
        "object": "chat.completion.graph_search_complete",
        "status": "知识图谱搜索完成",
        "graphList": graph_list,
    })


def graph_search_empty() -> SseEvent:
    return SseEvent("graph_search_complete", {
        "object": "chat.completion.graph_search_complete",
        "status": "知识图谱中未找到相关信息",
        "graphList": {"nodes": [], "links": []},
    })


def graph_search_error() -> SseEvent:
    return SseEvent("graph_search_error", {
        "object": "chat.completion.graph_search_error",
        "status": "执行知识图谱查询失败",
    })


# ── MCP ───────────────────────────────────────────────────────────────────────

def mcp_processing() -> SseEvent:
    return SseEvent("mcp_processing", {"object": "chat.completion.mcp_processing", "status": "正在处理MCP服务请求"})


def mcp_call(service_id: str, function_name: str, params: Dict[str, Any]) -> SseEvent:
    return SseEvent("mcp_call", {
        "object": "chat.completion.mcp_call",
        "service_id": service_id,
        "function_name": function_name,
        "params": params,
    })


def mcp_result(service: str, function: str, result: Any) -> SseEvent:
    return SseEvent("mcp_result", {
        "object": "chat.completion.mcp_result",
        "service": service,
        "function": function,
        "result": result,
    })


def mcp_error(msg: str = "MCP服务处理失败") -> SseEvent:
    return SseEvent("mcp_error", {"object": "chat.completion.mcp_error", "error": msg})

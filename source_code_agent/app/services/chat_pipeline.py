"""
对话流水线编排器。

四阶段 Pipeline：Audit -> Strategies -> Inference -> Filter
编排器只负责调度，不包含任何业务实现细节。
"""
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.agent import AgentShareToken
from app.domain.memory import MemoryManager
from app.services import agent_service as agent_utils
from app.services.strategy_base import BaseRetrievalStrategy, StrategyContext
from app.services.document_context_service import DocumentContextService
from app.services.model_inference_service import ModelInferenceService
from app.services.mcp_service import McpOrchestrationService
from app.services.chat_response_service import ChatResponseService
from app.services.graph_retrieval_service import GraphRetrievalService
from app.services.strategies import WebSearchStrategy, KnowledgeRetrievalStrategy, GraphRetrievalStrategy
from app.services import chat_events as ev

logger = logging.getLogger(__name__)


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
    """
    四阶段对话流水线编排器。

    所有服务通过构造函数注入，便于单元测试时替换为 fake/stub 实现。
    生产代码直接 ChatPipelineOrchestrator() 即可，所有依赖均有默认值。

    Args:
        document_service:   文件上下文处理（默认 DocumentContextService）
        inference_service:  模型流式推理（默认 ModelInferenceService）
        mcp_service:        MCP 工具编排（默认 McpOrchestrationService）
        response_service:   回答收尾与数据构建（默认 ChatResponseService）
        graph_service:      图谱检索（默认 GraphRetrievalService）
        strategy_registry:  检索增强策略列表（默认三种策略）
    """

    def __init__(
        self,
        *,
        document_service: Optional[DocumentContextService] = None,
        inference_service: Optional[ModelInferenceService] = None,
        mcp_service: Optional[McpOrchestrationService] = None,
        response_service: Optional[ChatResponseService] = None,
        graph_service: Optional[GraphRetrievalService] = None,
        strategy_registry: Optional[List[BaseRetrievalStrategy]] = None,
    ) -> None:
        self.document_service = document_service or DocumentContextService()
        self.inference_service = inference_service or ModelInferenceService()
        self.mcp_service = mcp_service or McpOrchestrationService()
        self.response_service = response_service or ChatResponseService()
        self.graph_service = graph_service or GraphRetrievalService()

        # 策略注册表：通过 is_active(agent) 动态激活，满足开闭原则
        # 传入 strategy_registry 可完整替换，便于测试
        self._strategy_registry: List[BaseRetrievalStrategy] = strategy_registry or [
            WebSearchStrategy(),
            KnowledgeRetrievalStrategy(),
            GraphRetrievalStrategy(self.graph_service),
        ]

    async def stream(self, request: ChatPipelineRequest) -> AsyncGenerator[Dict[str, Any], None]:
        state = ChatPipelineState(request=request)
        try:
            yield ev.status("开始处理请求").to_dict()

            for event in await self._audit(state):
                yield event

            for event in await self._run_strategies(state):
                yield event

            async for event in self._run_inference(state):
                yield event

            for event in await self._run_filter(state):
                yield event

        except Exception as exc:
            logger.exception("Pipeline 执行异常")
            yield ev.error(f"生成响应时出错: {str(exc)}").to_dict()
            yield ev.done_signal()

    # ── Audit ─────────────────────────────────────────────────────────────────

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
            events.append(ev.status("正在处理上传文件").to_dict())
            file_context_result = await self.document_service.process_files(request.db, request.file_ids)
            for msg in file_context_result.processed_messages:
                events.append(ev.file_processing(msg).to_dict())
            for msg in file_context_result.error_messages:
                events.append(ev.file_processing(msg).to_dict())
            file_system_context = self.document_service.build_system_context(file_context_result.formatted_contexts)
            if file_system_context:
                state.memory.prepend_context(file_system_context)
                state.final_messages = state.memory.messages()
                state.has_file_content = True

        if state.agent.system_prompt:
            state.memory.add_system_prompt(state.agent.system_prompt)
            state.final_messages = state.memory.messages()

        return events

    # ── Strategies ────────────────────────────────────────────────────────────

    async def _run_strategies(self, state: ChatPipelineState) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = [ev.think().to_dict()]

        strategy_context = StrategyContext(
            memory=state.memory,
            db=state.request.db,
            agent=state.agent,
            user_message=state.user_message,
            model_id=state.model_id,
            config=state.config,
        )

        active_strategies = [s for s in self._strategy_registry if s.is_active(state.agent)]
        for strategy in active_strategies:
            strategy_result = await strategy.execute(strategy_context)
            events.extend(strategy_result.events)
            if strategy_result.sources:
                state.sources.extend(strategy_result.sources)
            if strategy_result.web_search_results:
                state.web_search_results = strategy_result.web_search_results

        history_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in state.request.messages
            if msg.role in ("user", "assistant", "system")
        ]
        state.memory.add_history(history_messages)
        state.final_messages = state.memory.messages()

        return events

    # ── Inference ─────────────────────────────────────────────────────────────

    async def _run_inference(self, state: ChatPipelineState) -> AsyncGenerator[Dict[str, Any], None]:
        mcp_result = await self.mcp_service.run(
            db=state.request.db,
            agent=state.agent,
            user_message=state.user_message,
            model_id=state.model_id,
            current_user_id=state.request.current_user_id,
        )
        for event_item in mcp_result.events:
            yield event_item
        if mcp_result.tool_result_prompt:
            state.memory.add_tool_result(mcp_result.tool_result_prompt)
            state.final_messages = state.memory.messages()

        yield ev.reasoning().to_dict()
        yield ev.info(state.sources, state.web_search_results).to_dict()
        yield ev.answer_start().to_dict()

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

    # ── Filter ────────────────────────────────────────────────────────────────

    async def _run_filter(self, state: ChatPipelineState) -> List[Dict[str, Any]]:
        request = state.request
        if request.access_type == "share" and request.share_token:
            share_token_obj = (
                request.db.query(AgentShareToken)
                .filter(AgentShareToken.token == request.share_token)
                .first()
            )
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
            logger.exception("持久化聊天历史失败")

        return [
            ev.status("回答完成").to_dict(),
            ev.done_signal(),
        ]

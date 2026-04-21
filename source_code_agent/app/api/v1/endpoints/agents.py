from fastapi import APIRouter, Depends, HTTPException, Query, status, Body, Header
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import time
import json
import traceback  # 确保导入 traceback 模块
import logging
from sse_starlette.sse import EventSourceResponse

# 配置日志
logger = logging.getLogger(__name__)
from app.utils.model import execute_model_inference
from app.db.session import get_db
from app.utils.deps import get_current_active_user
from app.utils import agent as agent_utils
from app.schemas.agent import (
    AgentCreate, 
    AgentUpdate, 
    AgentResponse, 
    AgentListResponse,
    AgentChatRequest,
    AgentChatMessage,
    AgentChatResponse
)
from app.utils import format_datetime  # 添加这行导入
from app.models.agent import AgentChatHistory, AgentShareToken
from app.domain.memory import MemoryManager
from pydantic import ValidationError
from app.services.chat_orchestration_service import (
    DocumentContextService,
    ModelInferenceService,
    RetrievalAugmentationService,
    McpOrchestrationService,
    ChatResponseService,
    GraphRetrievalService,
    WebSearchStrategy,
    KnowledgeRetrievalStrategy,
    GraphRetrievalStrategy,
)
from app.services.strategy_base import StrategyContext

router = APIRouter()


def _resolve_agent_access(db: Session, agent_id: str, token: Optional[str]):
    """解析聊天访问目标智能体与访问类型。"""
    if token:
        agent = agent_utils.get_agent_by_share_token(db, token)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="分享令牌无效或已禁用"
            )
        return agent, agent.id, True

    agent = agent_utils.get_agent(db=db, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在"
        )
    return agent, agent_id, False


def _build_streaming_response(chat_request: AgentChatRequest, generator):
    if not chat_request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前仅支持流式请求"
        )
    return EventSourceResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


async def _chat_orchestration_generator(
    *,
    db: Session,
    agent_id: str,
    chat_request: AgentChatRequest,
    current_user_id: str,
    access_type: str,
    share_token: Optional[str] = None,
    api_key_id: Optional[str] = None,
):
    try:
        yield {"event": "status", "data": json.dumps({"object": "chat.completion.status", "status": "开始处理请求"}, ensure_ascii=False)}
        time.sleep(0.1)

        messages = chat_request.messages
        user_message = messages[-1].content if messages and messages[-1].role == "user" else ""
        session_id = chat_request.session_id or f"session_{int(time.time())}"
        config_override = chat_request.config or {}
        file_ids = chat_request.file_ids or []

        agent = agent_utils.get_agent(db, agent_id)
        if not agent:
            yield {"event": "error", "data": json.dumps({"error": "智能体不存在"}, ensure_ascii=False)}
            time.sleep(0.1)
            yield {"event": "done", "data": "[DONE]"}
            time.sleep(0.1)
            return

        if not user_message:
            yield {"event": "error", "data": json.dumps({"error": "请求中缺少用户消息"}, ensure_ascii=False)}
            time.sleep(0.1)
            yield {"event": "done", "data": "[DONE]"}
            time.sleep(0.1)
            return

        model_id = agent.model_id
        if not model_id:
            yield {"event": "error", "data": json.dumps({"error": "该智能体未关联模型，请先在智能体设置中关联一个对话模型"}, ensure_ascii=False)}
            time.sleep(0.1)
            yield {"event": "done", "data": "[DONE]"}
            time.sleep(0.1)
            return

        model = agent_utils.get_model(db, model_id)
        if not model:
            yield {"event": "error", "data": f'{{"error": "模型不存在: {model_id}"}}'}
            time.sleep(0.1)
            yield {"event": "done", "data": "[DONE]"}
            time.sleep(0.1)
            return

        agent_config = agent.config or {}
        config = {**agent_config, **config_override}
        start_time = time.time()

        response_content = ""
        used_tokens = 0
        sources = []
        web_search_results = []
        memory = MemoryManager()
        final_messages = memory.messages()
        document_service = DocumentContextService()
        inference_service = ModelInferenceService()
        retrieval_service = RetrievalAugmentationService()
        mcp_service = McpOrchestrationService()
        response_service = ChatResponseService()
        graph_service = GraphRetrievalService()
        has_file_content = False

        if file_ids:
            logger.info(f"开始处理文件，文件ID列表: {file_ids}")
            yield {"event": "status", "data": json.dumps({"object": "chat.completion.status", "status": "正在处理上传文件"}, ensure_ascii=False)}
            time.sleep(0.1)
            file_context_result = await document_service.process_files(db, file_ids)
            for status_msg in file_context_result.processed_messages:
                yield {"event": "file_processing", "data": json.dumps({"status": status_msg}, ensure_ascii=False)}
                time.sleep(0.1)
            for error_msg in file_context_result.error_messages:
                yield {"event": "file_processing", "data": json.dumps({"status": error_msg}, ensure_ascii=False)}
                time.sleep(0.1)

            file_system_context = document_service.build_system_context(file_context_result.formatted_contexts)
            if file_system_context:
                memory.prepend_context(file_system_context)
                final_messages = memory.messages()
                has_file_content = True

        yield {"event": "think", "data": json.dumps({"object": "chat.completion.think", "status": "AI开始思考该如何回答您的问题"}, ensure_ascii=False)}
        time.sleep(0.1)
        if agent.system_prompt:
            memory.add_system_prompt(agent.system_prompt)
            final_messages = memory.messages()

        strategy_context = StrategyContext(
            memory=memory,
            db=db,
            agent=agent,
            user_message=user_message,
            model_id=model_id,
            config=config,
        )
        active_strategies = []
        if agent.enable_web_search:
            active_strategies.append(WebSearchStrategy(retrieval_service))
        if agent.knowledge_bases:
            active_strategies.append(KnowledgeRetrievalStrategy(retrieval_service))
        if agent.graphs:
            active_strategies.append(GraphRetrievalStrategy(graph_service))

        for strategy in active_strategies:
            strategy_result = await strategy.execute(strategy_context)
            for event_item in strategy_result.events:
                data_payload = (
                    event_item["data"]
                    if isinstance(event_item["data"], str)
                    else json.dumps(event_item["data"], ensure_ascii=False)
                )
                yield {"event": event_item["event"], "data": data_payload}
                time.sleep(event_item.get("sleep", 0.1))
            if strategy_result.sources:
                sources.extend(strategy_result.sources)
            if strategy_result.web_search_results:
                web_search_results = strategy_result.web_search_results
        final_messages = memory.messages()

        history_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role in ["user", "assistant", "system"]
        ]
        memory.add_history(history_messages)
        final_messages = memory.messages()

        mcp_result = await mcp_service.run(
            db=db,
            agent=agent,
            user_message=user_message,
            model_id=model_id,
            current_user_id=current_user_id,
        )
        for event_item in mcp_result.events:
            yield {"event": event_item["event"], "data": json.dumps(event_item["data"], ensure_ascii=False)}
            time.sleep(event_item.get("sleep", 0.1))
        if mcp_result.tool_result_prompt:
            memory.add_tool_result(mcp_result.tool_result_prompt)
            final_messages = memory.messages()

        yield {"event": "reasoning", "data": '{"object": "chat.completion.reasoning", "status": "AI正在整合信息推理回答"}'}
        time.sleep(0.1)
        yield {"event": "info", "data": json.dumps({
            "object": "chat.completion.info",
            "sources": sources,
            "web_search_results": web_search_results
        }, ensure_ascii=False)}
        time.sleep(0.1)
        yield {"event": "answer", "data": '{"object": "chat.completion.answer", "status": "AI开始生成答案"}'}
        time.sleep(0.1)

        final_messages, has_file_content = await response_service.ensure_file_guidance(
            memory=memory,
            final_messages=final_messages,
            file_ids=file_ids,
            db=db,
            document_service=document_service,
            user_message=user_message,
        )

        model_payload = inference_service.build_stream_payload(final_messages, config)
        model_response = await inference_service.run_stream(db, model_id, model_payload)
        async for chunk in model_response:
            event_name, event_data, delta_content = inference_service.normalize_stream_chunk(chunk)
            if delta_content:
                response_content += delta_content
            yield {"event": event_name, "data": event_data}
            if event_name in ["tool_calls", "tool_call_result"]:
                time.sleep(0.1)

        response_time = int((time.time() - start_time) * 1000)
        share_token_id = None
        if access_type == "share" and share_token:
            share_token_obj = db.query(AgentShareToken).filter(AgentShareToken.token == share_token).first()
            if share_token_obj:
                share_token_id = share_token_obj.id

        extra_data = response_service.build_extra_data(
            response_time=response_time,
            used_tokens=used_tokens,
            sources=sources,
            web_search_results=web_search_results,
            has_file_content=has_file_content,
        )
        try:
            agent_utils.create_chat_history(
                db=db,
                agent_id=agent_id,
                session_id=session_id,
                user_id=current_user_id,
                user_message=user_message,
                agent_response=response_content,
                tokens_used=used_tokens,
                response_time=response_time,
                extra_data=extra_data,
                access_type=access_type,
                api_key_id=api_key_id,
                share_token_id=share_token_id,
                model_id=model_id,
            )
        except Exception as err:
            print(f"记录聊天历史失败: {err}")
            traceback.print_exc()

        yield {"event": "status", "data": '{"object": "chat.completion.status", "status": "回答完成"}'}
        time.sleep(0.1)
        yield {"event": "done", "data": "[DONE]"}
        time.sleep(0.1)
    except Exception as e:
        print(f"生成流式响应时出错: {e}")
        traceback.print_exc()
        yield {"event": "error", "data": json.dumps({"error": f"生成响应时出错: {str(e)}"})}
        time.sleep(0.1)
        yield {"event": "done", "data": "[DONE]"}
        time.sleep(0.1)

@router.get("/", response_model=AgentListResponse)
async def get_agents(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    name: Optional[str] = None,
    type: Optional[str] = None,
    status: Optional[str] = None,
    current_user = Depends(get_current_active_user),
):
    """
    获取智能体列表
    """
    skip = (page - 1) * limit
    agents = agent_utils.get_agents(
        db, 
        skip=skip, 
        limit=limit, 
        name=name, 
        type=type, 
        status=status
    )
    
    # 将数据库对象转换为响应模型
    agent_dicts = [agent.to_dict() for agent in agents]
    
    # 获取总数
    total = agent_utils.count_agents(db, name=name, type=type, status=status)
    
    return {
        "total": total,
        "items": agent_dicts
    }

@router.get("/types")
async def get_agent_types(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取智能体类型
    """
    # 更丰富的类型定义
    agent_types = [
        {
            "id": 1, 
            "name": "问答助手", 
            "value": "qa_bot", 
            "description": "基于知识库的问答智能体，专注于精确回答"
        },
        {
            "id": 2, 
            "name": "对话助手", 
            "value": "chat", 
            "description": "通用对话智能体，适合开放式对话场景"
        },
        {
            "id": 3, 
            "name": "知识库助手", 
            "value": "knowledge_assistant", 
            "description": "基于知识库的智能体，提供详细的知识解答"
        },
        {
            "id": 4, 
            "name": "图谱问答", 
            "value": "graph_qa", 
            "description": "基于知识图谱的智能体，善于处理结构化信息"
        },
        {
            "id": 5, 
            "name": "混合增强", 
            "value": "rag", 
            "description": "同时使用知识库和知识图谱的高级智能体"
        }
    ]
    return agent_types

@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent_in: AgentCreate = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    创建智能体
    """
    # 使用当前用户的用户名作为创建者（兼容旧版本）
    creator = current_user.username if current_user else None
    
    # 获取当前用户ID
    user_id = current_user.id if current_user else None
    
    # 创建智能体
    agent = agent_utils.create_agent(
        db=db, 
        agent_in=agent_in, 
        creator=creator,
        user_id=user_id  # 传递用户ID
    )
    
    return agent.to_dict()

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent_detail(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取智能体详情
    """
    agent = agent_utils.get_agent(db=db, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在"
        )
    
    return agent.to_dict()

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    agent_in: AgentUpdate = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    更新智能体
    """
    agent = agent_utils.get_agent(db=db, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在"
        )
    
    updated_agent = agent_utils.update_agent(
        db=db, 
        agent=agent, 
        agent_in=agent_in
    )
    
    return updated_agent.to_dict()

@router.post("/{agent_id}/avatar", response_model=AgentResponse)
async def update_agent_avatar(
    agent_id: str,
    avatar_data: Dict[str, str] = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    更新智能体头像
    
    接收base64编码的头像图片并保存
    """
    # 获取智能体
    agent = agent_utils.get_agent(db=db, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在"
        )
    
    # 验证请求体中包含avatar字段
    if "avatar" not in avatar_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请求缺少avatar字段"
        )
    
    # 创建更新请求对象
    avatar_update = AgentUpdate(avatar=avatar_data["avatar"])
    
    # 更新智能体
    updated_agent = agent_utils.update_agent(
        db=db, 
        agent=agent, 
        agent_in=avatar_update
    )
    
    return updated_agent.to_dict()

@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    删除智能体
    """
    agent = agent_utils.get_agent(db=db, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在"
        )
    
    agent_utils.delete_agent(db=db, agent_id=agent_id)

@router.post("/{agent_id}/chat", response_model=AgentChatResponse)
async def chat_with_agent(
    agent_id: str,
    chat_request: AgentChatRequest = Body(...),
    db: Session = Depends(get_db),
    # current_user: Any = Depends(get_optional_current_user),  # 移除JWT认证依赖
    token: Optional[str] = Query(None, description="分享令牌，用于免登录窗口访问"),
    user_id: Optional[str] = Query("000000", description="用户ID，默认为000000表示游客"),
):
    """
    与智能体对话
    """
    logger.debug(f"chat_with_agent接收到的请求: agent_id={agent_id}, user_id={user_id}")
    logger.debug(f"file_ids字段值: {chat_request.file_ids}")
    agent, resolved_agent_id, is_share_access = _resolve_agent_access(db, agent_id, token)
    access_type = "share" if is_share_access else "user"
    stream_generator = _chat_orchestration_generator(
        db=db,
        agent_id=resolved_agent_id,
        chat_request=chat_request,
        current_user_id=user_id or "000000",
        access_type=access_type,
        share_token=token if is_share_access else None,
    )
    return _build_streaming_response(chat_request, stream_generator)
        
        
@router.post("/{agent_id}/generate-share-token", response_model=Dict[str, str])
async def generate_agent_share_token(
    agent_id: str,
    name: Optional[str] = Query(None, description="Token名称/描述"),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    为智能体生成分享令牌
    """
    # 检查智能体是否存在
    agent = agent_utils.get_agent(db, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在"
        )
    
    try:
        # 生成分享令牌
        token_id, token = agent_utils.generate_share_token(db, agent_id, name)
        # 返回结果
        return {"id": token_id, "share_token": token}
    except Exception as e:
        print(f"生成分享令牌失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成分享令牌失败: {str(e)}"
        )


@router.post("/{agent_id}/generate-api-key", response_model=Dict[str, str])
async def generate_agent_api_key(
    agent_id: str,
    name: Optional[str] = Query(None, description="密钥名称/描述"),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    为智能体生成API密钥
    """
    # 检查智能体是否存在
    agent = agent_utils.get_agent(db, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在"
        )
    
    try:
        # 生成API密钥
        key_id, api_key = agent_utils.generate_api_key(db, agent_id, name)
        # 返回结果
        return {"id": key_id, "api_key": api_key}
    except Exception as e:
        print(f"生成API密钥失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成API密钥失败: {str(e)}"
        )


@router.get("/{agent_id}/api-keys", response_model=Dict[str, Any])
async def get_agent_api_keys(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    获取智能体的所有API密钥
    """
    # 检查智能体是否存在
    agent = agent_utils.get_agent(db, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在"
        )
    
    # 获取API密钥列表
    api_keys = agent_utils.get_agent_api_keys(db, agent_id)
    
    # 转换为响应格式
    return {
        "enabled": agent.api_enabled,
        "items": [api_key.to_dict() for api_key in api_keys]
    }


@router.get("/{agent_id}/share-tokens", response_model=Dict[str, Any])
async def get_agent_share_tokens(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    获取智能体的所有分享Token
    """
    # 检查智能体是否存在
    agent = agent_utils.get_agent(db, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在"
        )
    
    # 获取分享Token列表
    share_tokens = agent_utils.get_agent_share_tokens(db, agent_id)
    
    # 转换为响应格式
    return {
        "enabled": agent.share_enabled,
        "items": [share_token.to_dict() for share_token in share_tokens]
    }


@router.delete("/{agent_id}/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_api_key(
    agent_id: str,
    key_id: str,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    删除智能体的API密钥
    """
    # 检查智能体是否存在
    agent = agent_utils.get_agent(db, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在"
        )
    
    # 删除API密钥
    success = agent_utils.delete_api_key(db, key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API密钥不存在"
        )


@router.delete("/{agent_id}/share-tokens/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_share_token(
    agent_id: str,
    token_id: str,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    删除智能体的分享Token
    """
    # 检查智能体是否存在
    agent = agent_utils.get_agent(db, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在"
        )
    
    # 删除分享Token
    success = agent_utils.delete_share_token(db, token_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="分享Token不存在"
        )


@router.post("/{agent_id}/toggle-share", response_model=Dict[str, bool])
async def toggle_agent_share(
    agent_id: str,
    enabled: bool = Query(..., description="是否启用分享"),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    切换智能体分享状态
    """
    success = agent_utils.toggle_share_status(db, agent_id, enabled)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在"
        )
    
    return {"success": True, "share_enabled": enabled}


@router.post("/{agent_id}/toggle-api", response_model=Dict[str, bool])
async def toggle_agent_api(
    agent_id: str,
    enabled: bool = Query(..., description="是否启用API访问"),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    切换智能体API访问状态
    """
    success = agent_utils.toggle_api_status(db, agent_id, enabled)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在"
        )
    
    return {"success": True, "api_enabled": enabled}


@router.get("/share/{token}", response_model=Dict[str, Any])
async def get_agent_by_share_token(
    token: str,
    db: Session = Depends(get_db),
):
    """
    通过分享令牌获取智能体公开信息
    """
    agent = agent_utils.get_agent_by_share_token(db, token)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在或分享链接无效"
        )
    
    return {
        "agent_id": agent.id,
        "name": agent.name,
        "description": agent.description,
        "avatar": agent.avatar,
        "welcome_message": agent.welcome_message,
        "type": agent.type
    }


@router.post("/chat-with-api-key")
async def chat_with_agent_api(
    chat_request: AgentChatRequest = Body(...),
    api_key: str = Header(..., description="智能体API密钥"),
    db: Session = Depends(get_db),
    user_id: Optional[str] = Query("000000", description="用户ID，默认为000000表示游客"),
):
    """
    使用API密钥与智能体对话
    """
    result = agent_utils.get_agent_by_api_key(db, api_key)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API密钥无效或已禁用"
        )

    agent, api_key_id = result
    if not agent.api_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="该智能体未启用API访问"
        )

    stream_generator = _chat_orchestration_generator(
        db=db,
        agent_id=agent.id,
        chat_request=chat_request,
        current_user_id=user_id or "000000",
        access_type="api",
        api_key_id=api_key_id,
    )
    return _build_streaming_response(chat_request, stream_generator)


@router.get("/{agent_id}/logs", response_model=Dict[str, Any])
async def get_agent_chat_logs(
    agent_id: str,
    skip: int = 0,
    limit: int = 20,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    获取智能体会话列表
    """
    # 检查智能体是否存在
    agent = agent_utils.get_agent(db, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在"
        )
    
    # 查询所有会话ID
    from sqlalchemy import func
    session_query = db.query(
        AgentChatHistory.session_id,
        func.max(AgentChatHistory.created_at).label("last_message"),
        func.count(AgentChatHistory.id).label("message_count"),
        func.min(AgentChatHistory.created_at).label("first_message"),  # 增加返回每个session里第一条消息
        func.min(AgentChatHistory.user_message).label("first_user_message"),  # 增加返回每个session里第一条消息的user_message
        func.min(AgentChatHistory.type).label("type")  # 增加返回每个session里第一条消息的user_message
    ).filter(
        AgentChatHistory.agent_id == agent_id
    ).group_by(
        AgentChatHistory.session_id
    ).order_by(
        func.max(AgentChatHistory.created_at).desc()
    ).all()
    
    sessions = [
        {
            "sessionId": session[0],
            "lastMessage": format_datetime(session[1]),
            "messageCount": session[2],
            "firstMessage": format_datetime(session[3]),  # 增加返回每个session里第一条消息
            "firstUserMessage": session[4],  # 增加返回每个session里第一条消息的user_message
            "type":session[5]
        }
        for session in session_query
    ]
    
    return {
        "total": len(sessions),
        "items": sessions
    }

@router.get("/{agent_id}/share-chat/{share_id}")
async def get_agent_share_info(
    agent_id: str,
    share_id: str,
    db: Session = Depends(get_db),
):
    """
    通过分享ID获取智能体信息
    """
    agent = agent_utils.get_agent_by_share_token(db, share_id)
    # print(agent)
    if not agent or agent.id != agent_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在或分享链接无效"
        )
    
    return {
        "agent_id": agent.id,
        "name": agent.name,
        "description": agent.description,
        "avatar": agent.avatar,
        "welcome_message": agent.welcome_message,
        "type": agent.type,
        "agent_info":agent.to_dict()
    }


@router.post("/{agent_id}/share-chat/{share_id}/chat")
async def share_chat_with_agent(
    agent_id: str,
    share_id: str,
    request_data: dict = Body(...),
    db: Session = Depends(get_db),
):
    """
    通过分享链接与智能体对话
    """
    try:
        # 验证分享令牌
        agent = agent_utils.get_agent_by_share_token(db, share_id)
        if not agent or agent.id != agent_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="智能体不存在或分享链接无效"
            )
        
        # 转换请求格式
        message = request_data.get("message", "")
        history = request_data.get("history", [])
        session_id = request_data.get("session_id", f"share_{share_id}_{int(time.time())}")
        file_ids = request_data.get("fileIds", [])  # 获取文件ID列表
        
        print(f"分享页面聊天请求: message={message}, fileIds={file_ids}")
        
        # 构建符合 AgentChatRequest 格式的消息列表
        messages = []
        
        # 添加历史消息
        # if history:
        #     messages.extend(history)
        
        # 添加当前消息
        if message:
            messages.append(AgentChatMessage(role="user", content=message))
            # messages = AgentChatMessage(role="user", content=message)
        # 创建聊天请求对象
        chat_request_data = {
            "messages": messages,
            "stream": True,  # 启用流式响应
            "session_id": session_id,  # 创建会话ID
            "config": {},  # 可选的配置参数
            "type": "web",
            "fileIds": file_ids  # 使用fileIds字段名
        }
        print(f"创建聊天请求的原始数据: {chat_request_data}")
        chat_request = AgentChatRequest.model_validate(chat_request_data)
        
        print(f"创建的聊天请求对象: {chat_request.model_dump()}")
        
        # 调用现有的聊天处理逻辑
        return await chat_with_agent(
            agent_id=agent_id,
            chat_request=chat_request,
            db=db,
            token=share_id
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        print(f"分享聊天处理失败: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"聊天处理失败: {str(e)}"
        )   
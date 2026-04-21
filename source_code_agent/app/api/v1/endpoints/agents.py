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
    # 添加调试输出
    logger.debug(f"chat_with_agent接收到的请求: agent_id={agent_id}, user_id={user_id}")
    logger.debug(f"file_ids字段值: {chat_request.file_ids}")
    
    current_user_id = user_id  # 直接使用传入的user_id
    agent, agent_id, is_share_access = _resolve_agent_access(db, agent_id, token)
    
    # 创建异步生成器用于流式响应
    async def response_generator():
        try:
            # 声明要使用的外部变量
            nonlocal agent
            nonlocal current_user_id  # 确保可以访问current_user_id
            
            # 发送初始状态：开始处理
            yield {"event": "status", "data": json.dumps({"object": "chat.completion.status", "status": "开始处理请求"}, ensure_ascii=False)}
            time.sleep(0.1)
    
            # 提取请求参数
            messages = chat_request.messages
            user_message = messages[-1].content if messages and messages[-1].role == "user" else ""
            stream = chat_request.stream
            session_id = chat_request.session_id or f"session_{int(time.time())}"
            config_override = chat_request.config or {}
            file_ids = chat_request.file_ids or []  # 获取文件ID列表
            logger.debug(f"从chat_request提取的file_ids: {file_ids}")
            
            # 重新从数据库获取agent对象，确保它绑定到当前会话
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
            
            # 获取模型
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
            
            # 合并配置
            agent_config = agent.config or {}
            config = {**agent_config, **config_override}
            
            # 记录开始时间
            start_time = time.time()
            
            # 初始化变量
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
            processed_file_contents = []  # 存储处理后的文件内容
            has_file_content = False  # 标记是否有文件内容
            
            # 处理上传的文件
            if file_ids and len(file_ids) > 0:
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

                processed_file_contents = file_context_result.formatted_contexts
                file_system_context = document_service.build_system_context(processed_file_contents)
                if file_system_context:
                    logger.info(f"成功处理文件内容，总长度: {len(file_system_context)}")
                    memory.prepend_context(file_system_context)
                    final_messages = memory.messages()
                    has_file_content = True
                    print(f"添加文件内容到对话上下文: {file_system_context[:100]}...")
                else:
                    print("没有成功处理任何文件内容")
            
            # 发送思考状态
            yield {"event": "think", "data": json.dumps({"object": "chat.completion.think", "status": "AI开始思考该如何回答您的问题"}, ensure_ascii=False)}
            time.sleep(0.1)
            # 如果有系统提示词，添加到消息列表
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
            
            # 添加历史消息和当前用户消息
            history_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
                if msg.role in ["user", "assistant", "system"]
            ]
            memory.add_history(history_messages)
            final_messages = memory.messages()
            
            # 打印完整的消息列表，用于调试
            print("发送给模型的完整消息列表:")
            for i, msg in enumerate(final_messages):
                print(f"消息 {i+1} - 角色: {msg['role']}, 内容: {msg['content'][:100]}...")
            
            mcp_result = await mcp_service.run(
                db=db,
                agent=agent,
                user_message=user_message,
                model_id=model_id,
                current_user_id=current_user_id,
            )
            for event_item in mcp_result.events:
                yield {
                    "event": event_item["event"],
                    "data": json.dumps(event_item["data"], ensure_ascii=False),
                }
                time.sleep(event_item.get("sleep", 0.1))
            if mcp_result.tool_result_prompt:
                memory.add_tool_result(mcp_result.tool_result_prompt)
                final_messages = memory.messages()
            
            # 调用大模型
            inference_start = time.time()
            
            # 发送思考完成状态
            yield {"event": "reasoning", "data": '{"object": "chat.completion.reasoning", "status": "AI正在整合信息推理回答"}'}
            time.sleep(0.1)
            
            # 第一个事件：发送搜索结果和引用源
            search_results_and_sources = {
                "object": "chat.completion.info",
                "sources": sources,
                "web_search_results": web_search_results
            }
            yield {"event": "info", "data": json.dumps(search_results_and_sources,ensure_ascii=False)}
            time.sleep(0.1)
            
            
            # 开始生成答案事件
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
            
            # 调用大模型生成回答
            model_payload = inference_service.build_stream_payload(final_messages, config)
            model_response = await inference_service.run_stream(db, model_id, model_payload)
            response_content = ""
            # 处理流式响应
            async for chunk in model_response:
                event_name, event_data, delta_content = inference_service.normalize_stream_chunk(chunk)
                if delta_content:
                    response_content += delta_content
                yield {"event": event_name, "data": event_data}
                if event_name in ["tool_calls", "tool_call_result"]:
                    time.sleep(0.1)
            # 流处理完成
            end_time = time.time()
            response_time = int((end_time - start_time) * 1000)  # 计算响应时间，毫秒为单位
            
            # 记录聊天历史
            try:
                # 确定用户ID和访问类型
                # current_user_id = None
                access_type = "user"  # 默认为用户访问
                api_key_id = None
                share_token_id = None
                
                # 如果是分享链接访问
                if is_share_access:
                    access_type = "share"
                    if token:
                        # 查找对应的share_token记录
                        share_token_obj = db.query(AgentShareToken).filter(AgentShareToken.token == token).first()
                        if share_token_obj:
                            share_token_id = share_token_obj.id
                
                # # 如果是登录用户
                # elif current_user:
                #     try:
                #         # 尝试获取当前用户ID
                #         from app.api.deps import get_current_user_optional
                #         user_obj = await get_current_user_optional(db, token)
                #         if user_obj:
                #             current_user_id = user_obj.id
                #     except Exception as e:
                #         print(f"获取当前用户信息失败: {e}")
                        
                extra_data = response_service.build_extra_data(
                    response_time=response_time,
                    used_tokens=used_tokens,
                    sources=sources,
                    web_search_results=web_search_results,
                    has_file_content=has_file_content,
                )
                
                # 创建聊天历史记录，记录下使用的模型ID
                try:
                    agent_utils.create_chat_history(
                        db=db,
                        agent_id=agent_id,
                        session_id=session_id,
                        user_id=current_user_id,  # 使用正确的current_user_id变量
                        user_message=user_message,
                        agent_response=response_content,
                        tokens_used=used_tokens,
                        response_time=response_time,
                        extra_data=extra_data,
                        access_type=access_type,
                        api_key_id=api_key_id,
                        share_token_id=share_token_id,
                        model_id=model_id  # 添加模型ID
                    )
                except Exception as e:
                    print(f"记录聊天历史失败: {e}")
                    import traceback  # 确保在异常处理块内也能访问traceback
                    traceback.print_exc()
                
                yield {"event": "status", "data": '{"object": "chat.completion.status", "status": "回答完成"}'}
                time.sleep(0.1)
                yield {"event": "done", "data": "[DONE]"}
                time.sleep(0.1)
            
            except Exception as e:
                print(f"生成流式响应时出错: {e}")
                import traceback  # 确保在异常处理块内也能访问traceback
                traceback.print_exc()
                yield {"event": "error", "data": json.dumps({"error": f"生成响应时出错: {str(e)}"})}
                time.sleep(0.1)
                yield {"event": "done", "data": "[DONE]"}
                time.sleep(0.1)
        
        except Exception as e:
            print(f"生成流式响应时出错: {e}")
            import traceback  # 确保在异常处理块内也能访问traceback
            traceback.print_exc()
            yield {"event": "error", "data": json.dumps({"error": f"生成响应时出错: {str(e)}"})}
            time.sleep(0.1)
            yield {"event": "done", "data": "[DONE]"}
            time.sleep(0.1)
    
    if chat_request.stream:
        return EventSourceResponse(
            response_generator(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="当前仅支持流式请求"
    )
        
        
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
    # 验证API密钥
    result = agent_utils.get_agent_by_api_key(db, api_key)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API密钥无效或已禁用"
        )
    
    agent, api_key_id = result
    
    # 验证智能体API访问是否启用
    if not agent.api_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="该智能体未启用API访问"
        )
    
    # 创建异步生成器用于流式响应
    async def response_generator():
        try:
            # 发送初始状态：开始处理
            yield {"event": "status", "data": json.dumps({"object": "chat.completion.status", "status": "开始处理请求"}, ensure_ascii=False)}
            time.sleep(0.1)
    
            # 提取请求参数
            messages = chat_request.messages
            user_message = messages[-1].content if messages and messages[-1].role == "user" else ""
            stream = chat_request.stream
            session_id = chat_request.session_id or f"session_{int(time.time())}"
            config_override = chat_request.config or {}
            
            if not user_message:
                yield {"event": "error", "data": json.dumps({"error": "请求中缺少用户消息"}, ensure_ascii=False)}
                time.sleep(0.1)
                yield {"event": "done", "data": "[DONE]"}
                time.sleep(0.1)
                return
            
            # 获取模型
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
            
            # 合并配置
            agent_config = agent.config or {}
            config = {**agent_config, **config_override}
            
            # 记录开始时间
            start_time = time.time()
            
            # 初始化变量
            response_content = ""
            used_tokens = 0
            sources = []
            web_search_results = []
            final_messages = []
            
            # 处理上传的文件
            file_ids = chat_request.file_ids or []
            has_file_content = False
            processed_file_contents = []
            
            if file_ids and len(file_ids) > 0:
                yield {"event": "status", "data": json.dumps({"object": "chat.completion.status", "status": "正在处理上传文件"}, ensure_ascii=False)}
                time.sleep(0.1)
                
                # 导入文件模型和处理函数
                from app.models.file import File as FileModel
                from app.utils.file_processor import extract_text_from_file_path
                
                # 处理每个文件
                for file_id in file_ids:
                    try:
                        # 从数据库获取文件记录
                        file = db.query(FileModel).filter(FileModel.id == file_id).first()
                        if not file:
                            yield {"event": "file_processing", "data": json.dumps({"status": f"文件不存在: {file_id}"}, ensure_ascii=False)}
                            time.sleep(0.1)
                            continue
                        
                        # 检查文件状态
                        if file.status != "processed":
                            if file.status == "processing":
                                yield {"event": "file_processing", "data": json.dumps({"status": f"文件 {file.original_filename} 正在处理中，请稍后再试"}, ensure_ascii=False)}
                            else:
                                yield {"event": "file_processing", "data": json.dumps({"status": f"文件 {file.original_filename} 未处理完成，状态: {file.status}"}, ensure_ascii=False)}
                            time.sleep(0.1)
                            continue
                        
                        # 获取文件内容
                        file_content = ""
                        if file.text_content:
                            # 如果数据库中已有提取的文本内容，直接使用
                            file_content = file.text_content
                        else:
                            # 尝试从MinIO获取文件内容
                            from app.core.minio_client import get_file_stream
                            
                            try:
                                # 从路径中提取bucket和object_name
                                if file.path and '/' in file.path:
                                    bucket, object_name = file.path.split('/', 1)
                                    
                                    # 获取文件流
                                    response = get_file_stream(bucket, object_name)
                                    if response:
                                        # 创建临时文件
                                        import tempfile
                                        import os
                                        
                                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.file_type}") as temp_file:
                                            temp_file.write(response.read())
                                            temp_file_path = temp_file.name
                                        
                                        try:
                                            # 提取文本
                                            file_content = await extract_text_from_file_path(temp_file_path)
                                            if not file_content or file_content.startswith("提取文件文本内容时出错"):
                                                # 如果提取失败，尝试简单读取文件内容
                                                try:
                                                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                                                        file_content = f.read()
                                                except UnicodeDecodeError:
                                                    try:
                                                        with open(temp_file_path, 'r', encoding='latin-1') as f:
                                                            file_content = f.read()
                                                    except Exception as read_err:
                                                        print(f"读取文件内容失败: {read_err}")
                                                        yield {"event": "file_processing", "data": json.dumps({"status": f"读取文件 {file.original_filename} 内容失败"}, ensure_ascii=False)}
                                                        time.sleep(0.1)
                                        finally:
                                            # 删除临时文件
                                            try:
                                                os.unlink(temp_file_path)
                                            except:
                                                pass
                            except Exception as e:
                                print(f"获取文件内容失败: {e}")
                                yield {"event": "file_processing", "data": json.dumps({"status": f"获取文件 {file.original_filename} 内容失败: {str(e)}"}, ensure_ascii=False)}
                                time.sleep(0.1)
                                continue
                        
                        # 如果成功获取到文件内容
                        if file_content:
                            has_file_content = True
                            processed_file_contents.append({
                                "file_id": file_id,
                                "file_name": file.original_filename,
                                "content": file_content
                            })
                            yield {"event": "file_processing", "data": json.dumps({"status": f"文件 {file.original_filename} 处理完成"}, ensure_ascii=False)}
                            time.sleep(0.1)
                    except Exception as e:
                        print(f"处理文件失败: {e}")
                        yield {"event": "file_processing", "data": json.dumps({"status": f"处理文件失败: {str(e)}"}, ensure_ascii=False)}
                        time.sleep(0.1)
            
            # 设置访问类型
            access_type = "api"
            
            # 准备模型输入
            yield {"event": "status", "data": json.dumps({"object": "chat.completion.status", "status": "正在准备模型输入"}, ensure_ascii=False)}
            time.sleep(0.1)
            
            # 构建系统提示词
            system_prompt = agent.system_prompt or "你是一个智能助手，请回答用户的问题。"
            
            # 如果有文件内容，添加到系统提示词中
            if processed_file_contents:
                file_content_text = "\n\n".join([
                    f"文件名: {file_info['file_name']}\n内容:\n{file_info['content']}"
                    for file_info in processed_file_contents
                ])
                
                system_prompt += f"\n\n用户上传了以下文件，请基于这些文件内容回答问题：\n\n{file_content_text}"
            
            # 构建消息列表（使用 ADT 保护上下文状态，避免裸列表在链路中被意外污染）
            memory = MemoryManager()
            memory.add_system_prompt(system_prompt)

            # 添加历史消息
            history_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
                if msg.role in ["user", "assistant", "system"]
            ]
            memory.add_history(history_messages)
            final_messages = memory.messages()
            
            # 准备模型参数
            model_params = {
                "messages": final_messages,
                "stream": True
            }
            
            # 添加配置参数
            if "temperature" in config:
                model_params["temperature"] = float(config["temperature"])
            if "max_tokens" in config:
                model_params["max_tokens"] = int(config["max_tokens"])
            if "top_p" in config:
                model_params["top_p"] = float(config["top_p"])
            if "frequency_penalty" in config:
                model_params["frequency_penalty"] = float(config["frequency_penalty"])
            
            # 调用模型
            yield {"event": "status", "data": json.dumps({"object": "chat.completion.status", "status": "正在调用模型生成回复"}, ensure_ascii=False)}
            time.sleep(0.1)
            
            # 执行模型推理
            async for chunk in agent_utils.execute_model_inference(db, model_id, model_params):
                if isinstance(chunk, dict):
                    # 处理流式响应
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        choice = chunk["choices"][0]
                        if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
                            content = choice["delta"]["content"]
                            response_content += content
                            yield {"event": "message_chunk", "data": json.dumps(chunk, ensure_ascii=False)}
                            time.sleep(0.01)
                    
                    # 处理使用的tokens
                    if "usage" in chunk and "total_tokens" in chunk["usage"]:
                        used_tokens = chunk["usage"]["total_tokens"]
            
            # 计算响应时间
            response_time = int((time.time() - start_time) * 1000)  # 毫秒
            
            # 准备额外数据
            extra_data = {
                "tokens_used": used_tokens
            }
            
            # 添加知识库、网络搜索和资源引用信息（如果有）
            if sources:
                extra_data["sources"] = sources
            if web_search_results:
                extra_data["web_results"] = web_search_results
            if has_file_content:
                extra_data["has_file_content"] = True
            
            # 创建聊天历史记录，记录下使用的模型ID
            try:
                agent_utils.create_chat_history(
                    db=db,
                    agent_id=agent.id,
                    session_id=session_id,
                    user_id=user_id,  # 使用传入的user_id
                    user_message=user_message,
                    agent_response=response_content,
                    tokens_used=used_tokens,
                    response_time=response_time,
                    extra_data=extra_data,
                    access_type=access_type,
                    api_key_id=api_key_id,
                    model_id=model_id  # 添加模型ID
                )
            except Exception as e:
                print(f"记录聊天历史失败: {e}")
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
    
    try:
        # 如果是流式请求，直接返回流式响应
        if chat_request.stream:
            return EventSourceResponse(
                response_generator(),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
        
    except Exception as e:
        print(f"聊天处理失败: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"聊天处理失败: {str(e)}"
        )


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
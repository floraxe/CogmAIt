from fastapi import APIRouter, Depends, HTTPException, Query, status, Body, Header, Request
from fastapi.responses import StreamingResponse, Response, JSONResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, AsyncIterable, AsyncGenerator
import time
import json
import traceback  # 确保导入 traceback 模块
import asyncio
from sse_starlette.sse import EventSourceResponse
from app.utils.embedding import EmbeddingManager
from app.utils.model import execute_model_inference
from app.utils.knowledge import get_knowledge
from app.utils.web_search import search_web, get_web_search_client
from app.db.session import get_db
from app.utils.deps import get_current_active_user, get_optional_current_user
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
from app.utils.llm_knowledge_extractor import LLMKnowledgeExtractor
from app.utils.neo4j_utils import get_neo4j_service
from app.utils.config import get_neo4j_config
from app.utils import format_datetime  # 添加这行导入
from app.models.agent import Agent, AgentChatHistory, AgentShareToken
import uuid
from pydantic import ValidationError
from datetime import datetime

router = APIRouter()

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
    print(f"chat_with_agent接收到的请求: agent_id={agent_id}, user_id={user_id}")
    print(f"file_ids字段值: {chat_request.file_ids}")
    
    # 验证访问权限
    agent = None
    is_share_access = False
    current_user_id = user_id  # 直接使用传入的user_id
    
    # 如果有token，验证是否是有效的分享令牌
    if token:
        agent = agent_utils.get_agent_by_share_token(db, token)
        if agent:
            is_share_access = True
            agent_id = agent.id  # 确保使用正确的agent_id
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="分享令牌无效或已禁用"
            )
    
    # 如果还没获取agent，则通过agent_id获取
    if not agent:
        agent = agent_utils.get_agent(db=db, agent_id=agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="智能体不存在"
            )
    
    # 确保在继续之前agent变量已经被正确初始化
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="初始化智能体失败"
        )
    
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
            print(f"从chat_request提取的file_ids: {file_ids}")
            
            # 重新从数据库获取agent对象，确保它绑定到当前会话
            agent = agent_utils.get_agent(db, agent_id)
            if not agent:
                yield {"event": "error", "data": json.dumps({"error": "智能体不存在"}, ensure_ascii=False)}
                time.sleep(0.1)
                yield {"event": "done", "data": "[DONE]"}
                time.sleep(0.1)
                return
                
            # 获取关联的MCP服务
            mcp_services = []
            if hasattr(agent, "mcp_services") and agent.mcp_services:
                for service in agent.mcp_services:
                    mcp_services.append(service)
            
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
            processed_file_contents = []  # 存储处理后的文件内容
            has_file_content = False  # 标记是否有文件内容
            
            # 处理上传的文件
            if file_ids and len(file_ids) > 0:
                print(f"开始处理文件，文件ID列表: {file_ids}")
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
                        print(f"处理文件ID: {file_id}, 文件存在: {file is not None}")
                        if not file:
                            yield {"event": "file_processing", "data": json.dumps({"status": f"文件不存在: {file_id}"}, ensure_ascii=False)}
                            time.sleep(0.1)
                            continue
                        
                        # 检查文件状态
                        print(f"文件状态: {file.status}, 文件名: {file.original_filename}, 文件类型: {file.file_type}")
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
                            print(f"从数据库获取文件内容，长度: {len(file_content)}")
                        else:
                            # 尝试从MinIO获取文件内容
                            from app.core.minio_client import get_file_stream, RAW_BUCKET
                            
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
                                            
                                            # 删除临时文件
                                            if os.path.exists(temp_file_path):
                                                os.unlink(temp_file_path)
                                        except Exception as e:
                                            yield {"event": "file_processing", "data": json.dumps({"status": f"处理文件 {file.original_filename} 时出错: {str(e)}"}, ensure_ascii=False)}
                                            time.sleep(0.1)
                                            # 确保临时文件被删除
                                            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                                                os.unlink(temp_file_path)
                            except Exception as e:
                                yield {"event": "file_processing", "data": json.dumps({"status": f"获取文件 {file.original_filename} 内容时出错: {str(e)}"}, ensure_ascii=False)}
                                time.sleep(0.1)
                        
                        if file_content:
                            # 为长文件添加摘要和省略指示
                            max_content_length = 3000  # 最大内容长度限制
                            if len(file_content) > max_content_length:
                                truncated_content = file_content[:max_content_length] + f"\n\n[内容过长，已截断。原文共 {len(file_content)} 字符]"
                                file_content = truncated_content
                            
                            # 格式化文件内容，添加文件信息
                            formatted_content = f"--- 文件: {file.original_filename} ({file.file_type}) ---\n\n{file_content}\n\n"
                            processed_file_contents.append(formatted_content)
                            
                            yield {"event": "file_processing", "data": json.dumps({"status": f"已处理文件: {file.original_filename}"}, ensure_ascii=False)}
                            time.sleep(0.1)
                    except Exception as e:
                        yield {"event": "file_processing", "data": json.dumps({"status": f"处理文件ID {file_id} 时出错: {str(e)}"}, ensure_ascii=False)}
                        time.sleep(0.1)
                
                # 如果成功处理了文件，将文件内容添加到对话中
                if processed_file_contents:
                    combined_content = "\n".join(processed_file_contents)
                    print(f"成功处理文件内容，总长度: {len(combined_content)}")
                    
                    # 添加系统消息，包含文件内容，放在消息列表的最前面
                    file_context_message = {
                        "role": "system", 
                        "content": f"以下是用户上传的文件内容，这是非常重要的上下文信息，请务必仔细阅读并在回答问题时参考这些内容。如果用户询问关于文件内容的问题，请直接基于这些内容回答：\n\n{combined_content}"
                    }
                    # 将文件内容消息放在最前面，确保模型优先考虑
                    final_messages.insert(0, file_context_message)
                    
                    # 标记有文件内容
                    has_file_content = True
                    
                    # 添加日志，便于调试
                    print(f"添加文件内容到对话上下文: {combined_content[:100]}...")
                else:
                    print("没有成功处理任何文件内容")
            
            # 发送思考状态
            yield {"event": "think", "data": json.dumps({"object": "chat.completion.think", "status": "AI开始思考该如何回答您的问题"}, ensure_ascii=False)}
            time.sleep(0.1)
            # 如果有系统提示词，添加到消息列表
            if agent.system_prompt:
                final_messages.append({"role": "system", "content": agent.system_prompt})
            
            # 处理网络搜索 - 如果启用了网络搜索
            if agent.enable_web_search:
                # 发送网络搜索状态
                yield {"event": "web_search", "data": json.dumps({"object": "chat.completion.web_search", "status": "正在联网搜索最新信息"}, ensure_ascii=False)}
                time.sleep(0.5)
                try:
                    print(f"执行网络搜索: {user_message}")
                    
                    # 调用网络搜索API
                    search_results = await search_web(user_message)
                    
                    if search_results.get("results"):
                        # 将搜索结果保存，稍后添加到响应中
                        web_search_results = search_results.get("results", [])
                        
                        # 格式化搜索结果
                        web_search_client = get_web_search_client()
                        web_search_context = web_search_client.format_search_results(search_results)
                        
                        # 搜索结果信息来源
                        for i, result in enumerate(search_results.get("results", []), 1):
                            sources.append({
                                "content": result.get("content", ""),
                                "score": 1.0,  # 网络搜索结果默认高分
                                "source_file": result.get("title", "网络搜索结果"),
                                "url": result.get("url", ""),
                                "type": "web_search"  # 标记为网络搜索类型的来源
                            })
                        
                        # 添加网络搜索上下文
                        final_messages.append({"role": "system", "content": web_search_context})
                        print("添加了网络搜索结果到上下文")
                        
                        # 发送网络搜索结果状态
                        yield {"event": "web_search_complete", "data": json.dumps({"object": "chat.completion.web_search_complete", "status": "已找到相关信息", "results_count": len(web_search_results), "webList": web_search_results },ensure_ascii=False)}
                        time.sleep(0.5)
                    else:
                        print("网络搜索未返回结果")
                        yield {"event": "web_search_complete", "data": '{"object": "chat.completion.web_search_complete", "status": "未找到相关网络信息", "results_count": 0, "webList": []}'}
                        time.sleep(0.1)
                except Exception as e:
                    print(f"网络搜索失败: {str(e)}")
                    traceback.print_exc()
                    yield {"event": "web_search_complete", "data": f'{{"object": "chat.completion.web_search_complete", "status": "网络搜索过程中发生错误", "error": "{str(e)}", "webList": []}}'}
                    time.sleep(0.1)
            
            # 获取相关知识条目（如果需要）
            if agent.knowledge_bases:
                # 发送知识库检索状态
                yield {"event": "knowledge_search", "data": '{"object": "chat.completion.knowledge_search", "status": "正在检索知识库相关内容"}'}
                time.sleep(0.1)
                
                # 提取配置中的相似度阈值和召回条目数
                similarity_threshold = config.get("similarity_threshold", 0.7)
                top_k = config.get("top_k", 5)
                
                # 从知识库中检索相关信息
                retrieval_results = []
                kb_processed = 0
                total_kb = len(agent.knowledge_bases)
                
                for kb in agent.knowledge_bases:
                    kb_processed += 1
                    # 发送知识库检索进度
                    yield {"event": "knowledge_progress", "data": '{"object": "chat.completion.knowledge_progress", "status": "正在检索知识库", "progress": {kb_processed}/{total_kb}}'}
                    time.sleep(0.1)
                
                    knowledge = get_knowledge(db, kb.id)
                    if not knowledge or not knowledge.embedding_model:
                        continue
                
                    # 生成查询向量
                    yield {"event": "embedding", "data": '{"object": "chat.completion.embedding", "status": "正在计算语义向量"}'}
                    time.sleep(0.1)
                    
                    query_embedding_result = await execute_model_inference(
                        db,
                        knowledge.embedding_model,
                        {
                            "input": [user_message],
                            "model_type": "embedding"
                        }
                    )
                
                    if "error" in query_embedding_result:
                        print(f"生成查询向量失败: {query_embedding_result['error']}")
                        yield {"event": "embedding_error", "data": '{"object": "chat.completion.embedding_error", "status": "向量计算失败", "error": {query_embedding_result["error"]}}'}
                        time.sleep(0.1)
                        continue
                
                    # 获取查询向量
                    query_embedding = query_embedding_result.get("embeddings", [])[0]
                
                    # 在向量库中搜索
                    vector_store = EmbeddingManager.get_vector_store()
                    if not vector_store or vector_store.client is None:
                        print("向量存储服务不可用")
                        yield {"event": "vector_store_error", "data": '{"object": "chat.completion.vector_store_error", "status": "向量存储服务不可用"}'}
                        time.sleep(0.1)
                        continue
                
                    # 执行检索
                    yield {"event": "vector_search", "data": '{"object": "chat.completion.vector_search", "status": "正在检索相似文档"}'}
                    time.sleep(0.1)
                    
                    results = vector_store.search_similar(
                        knowledge_id=kb.id,
                        query_vector=query_embedding,
                        limit=top_k,
                        filter_expr=None
                    )
                    # 处理结果，补充文件信息
                    result_count = 0
                    for hit in results:
                        # 获取对应的文件信息
                        from app.utils.knowledge import get_knowledge_file
                        file_id = hit.get("file_id", "")
                        file_info = get_knowledge_file(db, file_id)
                        file_name = file_info.original_filename if file_info else "未知文件"
                        
                        # 计算得分，确保分数在0到1之间
                        score = hit.get("score", 0)
                        if score < 0:
                            score = 0
                        elif score > 1:
                            score = 1
                        
                        # 只添加符合相似度阈值的结果
                        if score >= similarity_threshold:
                            result_count += 1
                            retrieval_results.append({
                                "content": hit.get("text", "").replace(" ","").replace("\n","").replace("\\n","")[:512]+"...",
                                "score": score,
                                "source_file": file_name.replace(" ","").replace("\n","").replace("\\n",""),
                                "file_id": file_id,
                                "knowledge_id": kb.id,
                                "knowledge_name": knowledge.name.replace(" ","").replace("\n","").replace("\\n",""),
                                "chunk_id": hit.get("chunk_index", 0),
                                "type": "document"  # 标记为文档类型的来源
                            })
                    
                    # 发送检索结果状态
                    
                yield {"event": "vector_search_complete", "data": '{"object": "chat.completion.vector_search_complete", "status": "知识库检索完成，找到{result_count}条相关内容", "results_count": {result_count}, "ragList": []}'}
                time.sleep(0.5)
            
            # 构建图谱可视化数据
            graphList = {
                "nodes": [],
                "links": []
            }
            # 处理知识图谱相关逻辑
            graph_source_index = len(sources) + 1  # 知识图谱来源的起始索引
            if agent.graphs:
                # 发送知识图谱查询状态
                yield {"event": "graph_search", "data": '{"object": "chat.completion.graph_search", "status": "正在查询知识图谱"}'}
                time.sleep(0.1)
                
                try:
                    # 初始化知识图谱服务
                    neo4j_config = get_neo4j_config()
                    yield {"event": "graph_connecting", "data": '{"object": "chat.completion.graph_connecting", "status": "正在连接知识图谱数据库"}'}
                    time.sleep(0.1)
                    
                    neo4j_service = get_neo4j_service(
                        uri=neo4j_config.get("uri"),
                        username=neo4j_config.get("username"),
                        password=neo4j_config.get("password"),
                        database=neo4j_config.get("database"),
                        force_new=True
                    )
                    
                    # 检查连接是否成功
                    if not neo4j_service or not neo4j_service.driver:
                        print("Neo4j服务初始化失败，请检查连接配置")
                        yield {"event": "graph_connecting_error", "data": '{"object": "chat.completion.graph_connection_error", "status": "Neo4j服务初始化失败"}'}
                        time.sleep(0.1)
                        raise Exception("Neo4j服务未成功初始化")
                    
                    # 测试连接
                    if not neo4j_service.is_connected():
                        print("Neo4j连接测试失败，请检查连接配置")
                        yield {"event": "graph_connection", "data": '{"object": "chat.completion.graph_connection_error", "status": "Neo4j连接测试失败"}'}
                        time.sleep(0.1)
                        raise Exception("Neo4j连接测试失败")
                    else:
                        print("Neo4j连接成功")
                        yield {"event": "graph_connected", "data": '{"object": "chat.completion.graph_connected", "status": "知识图谱数据库连接成功"}'}
                        time.sleep(0.1)
                    
                    extractor = LLMKnowledgeExtractor(db=db)
                    
                    # 使用LLM从用户问题中提取实体和关系
                    graph_context = ""
                    graph_visualizations = []
                    
                    # 遍历每个关联的知识图谱
                    graph_processed = 0
                    total_graphs = len(agent.graphs)
                    
                    for gb in agent.graphs:
                        graph_processed += 1
                        # 发送图谱查询进度
                        # yield {"event": "graph_progress", "data": '{"object": "chat.completion.graph_progress", "status": "正在分析知识图谱", "progress": {graph_processed}/{total_graphs}}'}
                        # time.sleep(0.1)
                        
                        graph = agent_utils.get_graph(db, gb.id)
                        if not graph or not graph.neo4j_subgraph:
                            continue
                        
                        # 获取知识图谱schema
                        schema = None
                        try:
                            # 获取图谱的schema定义
                            from app.utils.graph import get_graph_schema
                            schema = get_graph_schema(db, graph.id)
                            yield {"event": "graph_schema", "data": '{"object": "chat.completion.graph_schema", "status": "已获取知识图谱结构定义"}'}
                            time.sleep(0.1)
                        except Exception as e:
                            print(f"获取图谱schema失败: {e}")
                            yield {"event": "graph_schema_error", "data": '{"object": "chat.completion.graph_schema_error", "status": "获取图谱结构定义失败", "error": {str(e)}}'}
                            time.sleep(0.1)
                        
                        # 使用大模型分析查询，生成Cypher查询语句
                        entity_extraction_prompt = f"""
                        请分析以下用户问题，并提取相关实体和属性进行知识图谱查询。

                        用户问题:
                        "{user_message}"
                        
                        当前知识图谱Schema包含以下实体和关系类型:
                        {json.dumps(schema, ensure_ascii=False) if schema else '未定义schema'}
                        
                        请严格按照以下JSON格式回复，仅从用户问题"{user_message}"中根据Schema要求进行知识抽取，不要添加任何额外内容，不要抽取schema中的内容:
                        {{
                          "entities": [
                            {{"name": "实体名称1", "type": "实体类型1"}},
                            {{"name": "实体名称2", "type": "实体类型2"}}
                          ],
                          "relations": ["关系类型1", "关系类型2"],
                          "cypher": "MATCH (n) WHERE n.name CONTAINS '实体名称' OR LOWER(n.name) CONTAINS LOWER('实体名称') RETURN n LIMIT 15"
                        }}

                        注意:
                        1. 不要指定具体节点标签，使用 MATCH (n) 而不是 MATCH (n:具体类型)
                        2. 使用宽松的匹配条件，如 WHERE n.name CONTAINS '名称' 或 LOWER(n.name) CONTAINS LOWER('关键词')
                        3. 确保查询简单有效，但要尽可能多地返回相关实体
                        4. 不要在查询中使用省略号(...)，不要省略任何条件
                        5. 仅返回JSON格式，不要添加文本或markdown格式
                        """
                        
                        # 使用大模型进行分析
                        yield {"event": "graph_analysis", "data": '{"object": "chat.completion.graph_analysis", "status": "正在分析问题与知识图谱的关联"}'}
                        time.sleep(0.1)
                        
                        extraction_result = await execute_model_inference(
                        db, 
                        model_id, 
                        {
                            "messages": [
                                {"role": "system", "content": "你是一个专门分析用户问题并生成Neo4j Cypher查询的助手。"},
                                {"role": "user", "content": entity_extraction_prompt}
                            ],
                            "model_type": "chat"
                        }
                        )
                        
                        yield {"event": "graph_query_generated", "data": '{"object": "chat.completion.graph_query_generated", "status": "已生成知识图谱查询语句"}'}
                        time.sleep(0.1)
                        
                        # 执行Cypher查询并获取结果
                        try:
                            # 解析大模型返回的结果
                            graph_search_result = []
                            cypher_query = None
                            
                            try:
                                # 处理大模型返回的内容
                                if isinstance(extraction_result, dict) and 'choices' in extraction_result:
                                    # 如果返回结果已经是一个字典
                                    message_content = ""
                                    if 'choices' in extraction_result and extraction_result['choices']:
                                        first_choice = extraction_result['choices'][0]
                                        if 'message' in first_choice and 'content' in first_choice['message']:
                                            message_content = first_choice['message']['content']
                                    
                                    # 从消息内容中提取JSON
                                    import re
                                    json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', message_content)
                                    if json_match:
                                        json_str = json_match.group(1)
                                        extraction_json = json.loads(json_str)
                                    else:
                                        # 尝试直接解析消息内容（可能直接是JSON）
                                        extraction_json = json.loads(message_content)
                                elif isinstance(extraction_result, str):
                                    # 传统字符串解析方式
                                    extraction_json = json.loads(extraction_result)
                                
                                # 从解析后的JSON提取Cypher查询
                                if "cypher" in extraction_json:
                                    cypher_query = extraction_json.get("cypher")
                                    
                                    # 记录实体和关系信息用于调试
                                    print(f"提取到的实体: {extraction_json.get('entities', [])}")
                                    print(f"提取到的关系: {extraction_json.get('relations', [])}")
                                    print(f"生成的Cypher查询: {cypher_query}")
                                    
                                    yield {"event": "graph_extraction", "data": '{"object": "chat.completion.graph_extraction", "status": "已分析用户问题中的实体和关系"}'}
                                    time.sleep(0.1)
                                
                            except Exception as json_error:
                                # 如果无法解析为JSON，尝试在文本中提取cypher查询
                                print(f"解析抽取结果为JSON失败: {json_error}")
                                print(f"原始结果: {extraction_result}")
                                
                                # 确定要解析的实际文本内容
                                text_to_parse = ""
                                if isinstance(extraction_result, dict) and 'choices' in extraction_result and extraction_result['choices']:
                                    first_choice = extraction_result['choices'][0]
                                    if 'message' in first_choice and 'content' in first_choice['message']:
                                        text_to_parse = first_choice['message']['content']
                                elif isinstance(extraction_result, str):
                                    text_to_parse = extraction_result
                                
                                # 尝试从文本中提取cypher查询
                                import re
                                # 尝试多种模式匹配
                                cypher_patterns = [
                                    r'```cypher\s*(MATCH[\s\S]+?)\s*```',
                                    r'```\s*(MATCH[\s\S]+?)\s*```',
                                    r'"cypher"\s*:\s*"(MATCH[\s\S]+?)(?:"|$)',
                                    r'(MATCH\s*\([^)]+\)[\s\S]+?RETURN[^;]+)',
                                ]
                                
                                for pattern in cypher_patterns:
                                    cypher_matches = re.findall(pattern, text_to_parse, re.IGNORECASE)
                                    if cypher_matches:
                                        cypher_query = cypher_matches[0].strip()
                                        print(f"从文本中提取到的Cypher查询: {cypher_query}")
                                        break
                                
                                if not cypher_query:
                                    # 获取有效实体名称进行模糊查询
                                    entity_name = user_message.strip().replace("'", "")
                                    # 创建更丰富的默认查询，获取与实体相关的所有节点、关系及属性
                                    cypher_query = f"""
                                    // 首先找到与关键词匹配的所有节点(不限类型)
                                    MATCH (start)
                                    WHERE start.graph_id = '{subgraph_id}' AND 
                                          (LOWER(start.name) CONTAINS LOWER('{entity_name}') OR 
                                           LOWER(start.title) CONTAINS LOWER('{entity_name}') OR
                                           LOWER(start.description) CONTAINS LOWER('{entity_name}') OR
                                           LOWER(start.content) CONTAINS LOWER('{entity_name}'))
                                    
                                    // 分三步UNION查询返回所有相关节点和关系
                                    // 1. 返回匹配的起始节点
                                    RETURN DISTINCT start AS n, null AS r
                                    
                                    // 2. 返回与起始节点直接相连的一级节点和关系
                                    UNION
                                    MATCH (start)
                                    WHERE start.graph_id = '{subgraph_id}' AND 
                                          (LOWER(start.name) CONTAINS LOWER('{entity_name}') OR 
                                           LOWER(start.title) CONTAINS LOWER('{entity_name}') OR
                                           LOWER(start.description) CONTAINS LOWER('{entity_name}') OR
                                           LOWER(start.content) CONTAINS LOWER('{entity_name}'))
                                    MATCH path = (start)-[r1]-(n1)
                                    WHERE n1.graph_id = '{subgraph_id}'
                                    RETURN DISTINCT n1 AS n, r1 AS r
                                    
                                    // 3. 返回与一级节点相连的二级节点和关系
                                    UNION
                                    MATCH (start)
                                    WHERE start.graph_id = '{subgraph_id}' AND 
                                          (LOWER(start.name) CONTAINS LOWER('{entity_name}') OR 
                                           LOWER(start.title) CONTAINS LOWER('{entity_name}') OR
                                           LOWER(start.description) CONTAINS LOWER('{entity_name}') OR
                                           LOWER(start.content) CONTAINS LOWER('{entity_name}'))
                                    MATCH path = (start)-[r1]-(n1)-[r2]-(n2)
                                    WHERE n1 <> start AND n1.graph_id = '{subgraph_id}' AND n2.graph_id = '{subgraph_id}'
                                    RETURN DISTINCT n2 AS n, r2 AS r
                                    
                                    LIMIT 200
                                    """
                                    print(f"生成高级知识图谱查询: {cypher_query}")
                                
                                yield {"event": "graph_extraction_text", "data": '{"object": "chat.completion.graph_extraction_text", "status": "已从回答中提取Cypher查询"}'}
                            
                            # 如果成功获取了cypher查询，执行查询前进行优化
                            if cypher_query:
                                # 移除所有节点标签约束，使查询更宽松
                                cypher_query = re.sub(r'MATCH\s*\(\w+:\w+\)', 'MATCH (n)', cypher_query)
                                
                                # 如果查询中没有WHERE子句但包含实体名称，添加模糊匹配条件
                                if "WHERE" not in cypher_query.upper() and user_message.strip():
                                    entity_name = user_message.strip()
                                    return_pos = cypher_query.upper().find("RETURN")
                                    if return_pos > 0:
                                        cypher_query = cypher_query[:return_pos] + f" WHERE n.name CONTAINS '{entity_name}' " + cypher_query[return_pos:]
                                
                                # 获取Neo4j配置
                                neo4j_config = get_neo4j_config()
                                neo4j_service = get_neo4j_service(
                                    uri=neo4j_config["uri"],
                                    username=neo4j_config["username"],
                                    password=neo4j_config["password"],
                                    database=neo4j_config["database"],
                                    force_new=True
                                )
                                
                                # 确保子图ID符合Neo4j命名规范
                                subgraph_id = graph.neo4j_subgraph.lower().replace(" ", "_").replace("-", "_")
                                
                                # 清理和修复Cypher查询
                                cypher_query = cypher_query.replace("...", "").strip()
                                if cypher_query.endswith('"'):
                                    cypher_query = cypher_query[:-1]
                                
                                # 确保单引号被正确处理
                                cypher_query = cypher_query.replace('"', "'")
                                # 修复可能的双重单引号问题
                                cypher_query = cypher_query.replace("''", "'")
                                
                                # 确保图谱ID过滤条件
                                try:
                                    # 确保子图ID符合Neo4j命名规范
                                    subgraph_id = graph.neo4j_subgraph.lower().replace(" ", "_").replace("-", "_")
                                    
                                    # 简化查询处理逻辑
                                    if "WHERE" in cypher_query.upper():
                                        # 在已有WHERE条件中添加图谱ID条件
                                        where_pos = cypher_query.upper().find("WHERE")
                                        return_pos = cypher_query.upper().find("RETURN", where_pos)
                                        
                                        if return_pos != -1:
                                            # 在现有WHERE和RETURN之间插入图谱ID条件
                                            where_part = cypher_query[where_pos:return_pos]
                                            if "graph_id" not in where_part.lower():
                                                # 只有在没有graph_id条件时才添加
                                                new_where = f"WHERE n.graph_id = '{subgraph_id}' AND " + where_part[5:].strip()
                                                cypher_query = cypher_query[:where_pos] + new_where + cypher_query[return_pos:]
                                        else:
                                            # 找不到RETURN，在WHERE后直接添加条件
                                            if "graph_id" not in cypher_query.lower():
                                                cypher_query = cypher_query.replace("WHERE", f"WHERE n.graph_id = '{subgraph_id}' AND ")
                                    else:
                                        # 查找RETURN位置，在前面添加WHERE子句
                                        return_pos = cypher_query.upper().find("RETURN")
                                        if return_pos > 0:
                                            cypher_query = cypher_query[:return_pos] + f" WHERE n.graph_id = '{subgraph_id}' " + cypher_query[return_pos:]
                                        else:
                                            # 在查询末尾添加WHERE子句
                                            cypher_query += f" WHERE n.graph_id = '{subgraph_id}'"
                                except Exception as filter_error:
                                    print(f"添加图谱ID过滤条件失败: {filter_error}")
                                    traceback.print_exc()
                                    # 如果处理出错，使用简单的查询
                                    entity_name = user_message.strip().replace("'", "")
                                    cypher_query = f"MATCH (n) WHERE n.graph_id = '{subgraph_id}' AND n.name CONTAINS '{entity_name}' RETURN n LIMIT 10"
                                
                                print(f"执行的Cypher查询: {cypher_query}")
                                yield {"event": "graph_search", "data": '{"object": "chat.completion.graph_search", "status": "正在查询知识图谱..."}'}
                                time.sleep(0.5)
                                # 定义查询实体名称，确保在整个处理过程中可用
                                query_entity_name = user_message.strip().replace("'", "")
                                
                                # 执行查询
                                try:
                                    with neo4j_service.driver.session(database=neo4j_service.database) as session:
                                        result = session.run(cypher_query)
                                        records = list(result)
                                        
                                        # 处理查询结果
                                        if records:
                                            print(f"Neo4j查询返回 {len(records)} 条结果")
                                            
                                            # 处理结果转换为标准格式
                                            processed_records = []
                                            try:
                                                for record in records:
                                                    processed_record = {}
                                                    # 处理节点和关系，转换为可序列化格式
                                                    for key, value in record.items():
                                                        if value is None:
                                                            processed_record[key] = None
                                                            continue
                                                            
                                                        if hasattr(value, 'id') and hasattr(value, 'labels'):
                                                            # 是节点
                                                            # 使用正确的属性访问方式 - 兼容不同版本的Neo4j驱动
                                                            if hasattr(value, "properties"):
                                                                node_props = {k: v for k, v in dict(value.properties).items()}
                                                            elif hasattr(value, "_properties"):
                                                                node_props = {k: v for k, v in dict(value._properties).items()}
                                                            else:
                                                                # 如果无法直接获取属性，尝试使用字典方式访问
                                                                node_props = {}
                                                                try:
                                                                    # 遍历可能的属性
                                                                    for key in dir(value):
                                                                        if not key.startswith('_') and not callable(getattr(value, key)):
                                                                            node_props[key] = getattr(value, key)
                                                                except:
                                                                    # 如果失败，使用空字典
                                                                    pass
                                                            
                                                            # 确保所有属性值都可序列化
                                                            for prop_key, prop_value in list(node_props.items()):
                                                                if isinstance(prop_value, (dict, list)):
                                                                    try:
                                                                        # 尝试JSON序列化检查
                                                                        json.dumps(prop_value)
                                                                    except:
                                                                        # 如果不可序列化，转为字符串
                                                                        node_props[prop_key] = str(prop_value)
                                                                elif not isinstance(prop_value, (str, int, float, bool, type(None))):
                                                                    # 非基本类型转为字符串
                                                                    node_props[prop_key] = str(prop_value)
                                                            
                                                            node_id = str(value.id)
                                                            node_labels = list(value.labels) if hasattr(value, 'labels') else ['Entity']
                                                            processed_record[key] = {
                                                                'id': node_id,
                                                                'labels': node_labels,
                                                                'properties': node_props
                                                            }
                                                        elif hasattr(value, 'type') and hasattr(value, 'start_node') and hasattr(value, 'end_node'):
                                                            # 是关系
                                                            rel_type = value.type
                                                            start_id = str(value.start_node.id)
                                                            end_id = str(value.end_node.id)
                                                            rel_props = {k: v for k, v in dict(value.properties).items()} if hasattr(value, 'properties') else {}
                                                            processed_record[key] = {
                                                                'type': rel_type,
                                                                'start': start_id,
                                                                'end': end_id,
                                                                'properties': rel_props
                                                            }
                                                        else:
                                                            # 其他类型
                                                            processed_record[key] = str(value)
                                                    
                                                    processed_records.append(processed_record)
                                                
                                                print(f"处理后的结果: {processed_records}...")
                                                
                                                # 构建图形化数据
                                                graph_search_result = processed_records
                                                
                                                # 跟踪已处理的节点和边，避免重复
                                                processed_nodes = {}
                                                processed_links = {}
                                                
                                                
                                                # 处理所有节点和关系
                                                for record in processed_records:
                                                    # 处理节点
                                                    if 'n' in record and record['n'] is not None:
                                                        node_data = record['n']
                                                        node_id = node_data['id']
                                                        
                                                        if node_id not in processed_nodes:
                                                            properties = node_data['properties']
                                                            # 确保name属性存在
                                                            node_name = properties.get('name', properties.get('title', f'节点{node_id}'))
                                                            if not node_name or node_name == f'节点{node_id}':
                                                                # 尝试寻找其他可用作名称的属性
                                                                for possible_name in ['label', 'text', 'value', 'description']:
                                                                    if possible_name in properties and properties[possible_name]:
                                                                        node_name = str(properties[possible_name])
                                                                        break
                                                            
                                                            node_type = node_data['labels'][0] if node_data['labels'] else 'Entity'
                                                            
                                                            graphList["nodes"].append({
                                                                "id": node_id,
                                                                "name": node_name,
                                                                "symbolSize": 50,
                                                                "category": node_type,
                                                                "properties": properties
                                                            })
                                                            processed_nodes[node_id] = True
                                                    
                                                    # 处理关系
                                                    if 'r' in record and record['r'] is not None:
                                                        rel_data = record['r']
                                                        link_id = f"{rel_data['start']}_{rel_data['type']}_{rel_data['end']}"
                                                        
                                                        if link_id not in processed_links:
                                                            graphList["links"].append({
                                                                "source": rel_data['start'],
                                                                "target": rel_data['end'],
                                                                "value": rel_data['type'],
                                                                "properties": rel_data.get('properties', {})
                                                            })
                                                            processed_links[link_id] = True
                                                
                                                # 构建RAG知识结果
                                                ragList = []
                                                
                                                # 首先添加中心节点
                                                if len(graphList["nodes"]) > 0:
                                                    # 查找与搜索词最匹配的节点作为中心节点
                                                    center_nodes = [node for node in graphList["nodes"] 
                                                                if query_entity_name and query_entity_name.lower() in (node.get("name", "").lower() or "")]
                                                    
                                                    if center_nodes:
                                                        center_node = center_nodes[0]
                                                        # 标记为中心节点 - 调整大小和样式
                                                        center_node["symbolSize"] = 70
                                                        center_node["itemStyle"] = {"color": "#f06292"}
                                                        
                                                        # 添加到RAG结果
                                                        properties = center_node["properties"]
                                                        content_parts = []
                                                        
                                                        # 收集有意义的属性信息
                                                        for k, v in properties.items():
                                                            if k not in ["graph_id", "id", "created_at"] and v:
                                                                content_parts.append(f"{k}: {v}")
                                                        
                                                        if content_parts:
                                                            node_content = "\n".join(content_parts)
                                                            ragList.append({
                                                                "knowledge_name": graph.name,
                                                                "source_file": f"{center_node['name']} ({center_node['category']})",
                                                                "content": node_content,
                                                                "score": 0.99,
                                                                "type": "graph"
                                                            })
                                                    else:
                                                        # 如果找不到完全匹配的节点，使用第一个节点作为中心
                                                        center_node = graphList["nodes"][0]
                                                        center_node["symbolSize"] = 70
                                                        center_node["itemStyle"] = {"color": "#f06292"}
                                                
                                                # 然后添加相关关系信息
                                                if len(graphList["links"]) > 0:
                                                    # 根据关系类型分组
                                                    relation_types = {}
                                                    for link in graphList["links"]:
                                                        rel_type = link["value"]
                                                        if rel_type not in relation_types:
                                                            relation_types[rel_type] = []
                                                        
                                                        # 查找源节点和目标节点
                                                        source_node = next((n for n in graphList["nodes"] if n["id"] == link["source"]), None)
                                                        target_node = next((n for n in graphList["nodes"] if n["id"] == link["target"]), None)
                                                        
                                                        if source_node and target_node:
                                                            relation_types[rel_type].append({
                                                                "source": source_node.get("name", "未知节点"),
                                                                "source_type": source_node.get("category", "未知类型"),
                                                                "target": target_node.get("name", "未知节点"),
                                                                "target_type": target_node.get("category", "未知类型"),
                                                                "properties": link.get("properties", {})
                                                            })
                                                    
                                                    # 为每种关系类型创建一个RAG条目
                                                    for rel_type, relations in relation_types.items():
                                                        # 按关系源节点和目标节点名称排序，使结果更有条理
                                                        try:
                                                            relations.sort(key=lambda x: (x.get("source", ""), x.get("target", "")))
                                                        except Exception as sort_error:
                                                            print(f"排序关系时出错: {sort_error}")
                                                            # 排序错误不影响继续处理
                                                        
                                                        content_parts = [f"关系类型: {rel_type}", ""]
                                                        
                                                        # 获取所有关系实例
                                                        for i, relation in enumerate(relations[:20]):  # 限制最多20个关系实例
                                                            source = relation["source"]
                                                            source_type = relation["source_type"]
                                                            target = relation["target"]
                                                            target_type = relation["target_type"]
                                                            properties = relation["properties"]
                                                            
                                                            content_parts.append(f"{i+1}. {source} ({source_type}) → {target} ({target_type})")
                                                            
                                                            # 添加关系属性
                                                            if properties and len(properties) > 0:
                                                                for k, v in properties.items():
                                                                    if k not in ["graph_id", "id"] and v:
                                                                        content_parts.append(f"   - {k}: {v}")
                                                                content_parts.append("")
                                                        
                                                        if len(relations) > 20:
                                                            content_parts.append(f"...还有 {len(relations) - 20} 个其他实例")
                                                        
                                                        # 确保content_parts不为空
                                                        if len(content_parts) > 2:  # 至少有标题和一个实例
                                                            node_content = "\n".join(content_parts)
                                                            ragList.append({
                                                                "knowledge_name": graph.name,
                                                                "source_file": f"关系: {rel_type}",
                                                                "content": node_content,
                                                                "score": 0.95,
                                                                "type": "graph"
                                                            })
                                                
                                                # 如果有图谱数据但没有RAG结果，添加一个基本的图谱信息
                                                if len(graphList["nodes"]) > 0 and not ragList:
                                                    content_parts = [
                                                        f"查询 '{query_entity_name}' 在知识图谱中找到:",
                                                        f"- {len(graphList['nodes'])} 个节点",
                                                        f"- {len(graphList['links'])} 条关系"
                                                    ]
                                                    
                                                    # 添加部分节点名称作为参考
                                                    node_names = [node.get("name", "未命名节点") for node in graphList["nodes"][:10]]
                                                    if node_names:
                                                        content_parts.append("")
                                                        content_parts.append("相关节点包括:")
                                                        for name in node_names:
                                                            content_parts.append(f"- {name}")
                                                    
                                                    ragList.append({
                                                        "knowledge_name": graph.name,
                                                        "source_file": "知识图谱概览",
                                                        "content": "\n".join(content_parts),
                                                        "score": 0.9,
                                                        "type": "graph"
                                                    })
                                            except Exception as process_error:
                                                print(f"处理Neo4j结果时出错: {process_error}")
                                                traceback.print_exc()
                                                # 出错时创建一个简单节点用于调试
                                                graphList = {
                                                    "nodes": [{
                                                        "id": "error_node",
                                                        "name": f"处理出错: {str(process_error)}",
                                                        "symbolSize": 50,
                                                        "category": "Error",
                                                        "properties": {"error": str(process_error)}
                                                    }],
                                                    "links": []
                                                }
                                                # 确保ragList初始化，避免未定义错误
                                                ragList = []
                                            
                                            # 构建RAG结果
                                            for item in graph_search_result[:5]:  # 限制返回的结果数量用于RAG
                                                # 提取知识内容
                                                content = ""
                                                source_file = ""
                                                knowledge_name = graph.name
                                                
                                                # 尝试提取不同格式的内容
                                                for key, value in item.items():
                                                    if isinstance(value, dict):
                                                        # 对于节点或关系对象
                                                        if "name" in value:
                                                            source_file = value.get("name", "")
                                                        if "description" in value:
                                                            content += value.get("description", "") + "\n"
                                                        elif "content" in value:
                                                            content += value.get("content", "") + "\n"
                                                        else:
                                                            # 使用所有属性作为内容
                                                            content += " ".join([f"{k}: {v}" for k, v in value.items() if k not in ["graph_id", "id", "created_at"]]) + "\n"
                                                    elif isinstance(value, str) and len(value) > 5:
                                                        content += value + "\n"
                                                
                                                # 添加到结果列表
                                                if content:
                                                    ragList.append({
                                                        "knowledge_name": knowledge_name,
                                                        "source_file": source_file or "知识图谱结果",
                                                        "content": content.strip(),
                                                        "score": 0.95,  # 固定相似度分数
                                                        "type": "document"
                                                    })
                                            
                                            # 发送知识图谱RAG结果
                                            if ragList:
                                                graphListData =  {"nodes": [{"id": n.id, "name": n.name, "symbolSize": n.symbolSize, "category": n.category, "properties": n.properties} for n in graphList["nodes"]],
                                                                    "links": [{"id": l.id, "source": l.source, "target": l.target, "value": l.value, "properties": l.properties} for l in graphList["links"]]
                                                                    }
                                                yield {"event": "graph_search_complete", "data": json.dumps({"object": "chat.completion.graph_search_complete", "status": "知识图谱搜索完成", "graphList":graphListData},ensure_ascii=False)}
                                                time.sleep(0.5)
                                            else:
                                                yield {"event": "graph_search_complete", "data": '{"object": "chat.completion.graph_search_complete", "status": "知识图谱中未找到相关信息", "graphList": {"nodes": [], "links": []}}'}
                                                time.sleep(0.5)
                                        else:
                                            print("Neo4j查询未返回结果")
                                            yield {"event": "graph_search_complete", "data": '{"object": "chat.completion.graph_search_complete", "status": "知识图谱中未找到相关信息", "graphList": {"nodes": [], "links": []}}'}
                                            time.sleep(0.1)
                                except Exception as query_error:
                                    print(f"执行Neo4j查询失败: {query_error}")
                                    traceback.print_exc()
                                    yield {"event": "graph_search_error", "data": '{"object": "chat.completion.graph_search_error", "status": "执行知识图谱查询失败", "error": {str(query_error)}}'}
                                    time.sleep(0.1)
                            else:
                                print("未能提取出有效的Cypher查询")
                                yield {"event": "graph_search_error", "data": '{"object": "chat.completion.graph_search_error", "status": "无法生成有效的知识图谱查询"}'}
                                time.sleep(0.1)
                        except Exception as e:
                            print(f"处理知识图谱查询失败: {e}")
                            traceback.print_exc()
                            yield {"event": "graph_search_error", "data": '{"object": "chat.completion.graph_search_error", "status": "知识图谱查询失败", "error": {str(e)}}'}
                            time.sleep(0.1)
                    
                    # 知识图谱查询完成通知
                    yield {"event": "graph_search_complete", "data": '{"object": "chat.completion.graph_search_complete", "status": "知识图谱查询完成", "graphList": {nodes: {id: n.id, name: n.name, symbolSize: n.symbolSize, category: n.category, properties: n.properties} for n in graphList["nodes"]}, links: {id: l.id, source: l.source, target: l.target, value: l.value, properties: l.properties} for l in graphList["links"]}}'}
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"处理知识图谱查询失败: {e}")
                    traceback.print_exc()
                    yield {"event": "graph_search_error", "data": '{"object": "chat.completion.graph_search_error", "status": "知识图谱查询失败", "error": {str(e)}}'}
                    time.sleep(0.1)
            
            # 添加历史消息和当前用户消息
            for msg in messages:
                final_messages.append({"role": msg.role, "content": msg.content})
            
            # 打印完整的消息列表，用于调试
            print("发送给模型的完整消息列表:")
            for i, msg in enumerate(final_messages):
                print(f"消息 {i+1} - 角色: {msg['role']}, 内容: {msg['content'][:100]}...")
            
            # 处理MCP服务
            if mcp_services and len(mcp_services) > 0:
                try:
                    # 通知前端正在处理MCP服务
                    yield {"event": "mcp_processing", "data": json.dumps({"object": "chat.completion.mcp_processing", "status": "正在处理MCP服务请求"}, ensure_ascii=False)}
                    time.sleep(0.1)
                    
                    # 导入MCP服务工具
                    from app.utils.mcp import call_mcp_service, analyze_mcp_service_needs
                    
                    # 获取MCP服务列表
                    mcp_service_list = []
                    if hasattr(agent, "mcp_services") and agent.mcp_services:
                        for service in agent.mcp_services:
                            mcp_service_list.append(service)
                    
                    if mcp_service_list:
                        # 分析用户消息，判断是否需要调用MCP服务
                        print(f"开始分析用户消息是否需要调用MCP服务: {user_message[:50]}...")
                        detection_result = await analyze_mcp_service_needs(
                            db=db,
                            model_id=model_id,
                            user_message=user_message,
                            available_services=mcp_service_list
                        )
                        
                        print(f"MCP服务需求分析结果: {json.dumps(detection_result, ensure_ascii=False)}")
                        
                        # 如果检测到需要调用MCP服务
                        if detection_result and detection_result.get("call_mcp", False):
                            try:
                                service_id = detection_result.get("service_id")
                                function_name = detection_result.get("function_name")
                                params = detection_result.get("params", {})
                                
                                if not service_id:
                                    error_msg = "MCP服务ID为空"
                                    print(error_msg)
                                    yield {"event": "mcp_error", "data": json.dumps({
                                        "object": "chat.completion.mcp_error",
                                        "error": error_msg
                                    }, ensure_ascii=False)}
                                    return
                                    
                                if not function_name:
                                    error_msg = "MCP函数名称为空"
                                    print(error_msg)
                                    yield {"event": "mcp_error", "data": json.dumps({
                                        "object": "chat.completion.mcp_error",
                                        "error": error_msg
                                    }, ensure_ascii=False)}
                                    return
                                
                                print(f"需要调用MCP服务: service_id={service_id}, function={function_name}, params={json.dumps(params, ensure_ascii=False)}")
                                
                                # 通知前端调用了哪个MCP服务
                                yield {"event": "mcp_call", "data": json.dumps({
                                    "object": "chat.completion.mcp_call",
                                    "service_id": service_id,
                                    "function_name": function_name,
                                    "params": params
                                }, ensure_ascii=False)}
                                time.sleep(0.1)
                                
                                # 从关联的服务中查找指定服务
                                service = next((s for s in mcp_service_list if s.id == service_id), None)
                                
                                if service:
                                    # 获取当前用户ID，如果存在的话
                                    # current_user_id = None
                                    # if token:
                                    #     # 如果是通过分享令牌访问，使用智能体所有者ID
                                    #     current_user_id = agent.created_by
                                    # else:
                                    #     # 尝试获取当前登录用户ID
                                    #     try:
                                    #         # from app.api.deps import get_current_user_optional
                                    #         current_user = await get_current_user_optional(db, token)
                                    #         if current_user:
                                    #             current_user_id = current_user.id
                                    #     except:
                                    #         # 如果无法获取当前用户，使用智能体所有者ID
                                    #         current_user_id = agent.created_by

                                    print(f"开始调用MCP服务: service_id={service_id}, function={function_name}, params={json.dumps(params, ensure_ascii=False)}")
                                    
                                    # 调用MCP服务
                                    mcp_result = await call_mcp_service(
                                        db=db,
                                        service_id=service_id,
                                        function_name=function_name,
                                        params=params,
                                        user_id=current_user_id
                                    )
                                    
                                    print(f"MCP服务调用结果: {json.dumps(mcp_result, ensure_ascii=False)}")
                                    
                                    # 通知前端MCP服务调用结果
                                    service_name = service.name if hasattr(service, 'name') else service.get('name', 'Unknown')
                                    yield {"event": "mcp_result", "data": json.dumps({
                                        "object": "chat.completion.mcp_result",
                                        "service": service_name,
                                        "function": function_name,
                                        "result": mcp_result
                                    }, ensure_ascii=False)}
                                    time.sleep(0.1)
                                    
                                    # 将MCP服务结果添加到消息中
                                    final_messages.append({
                                        "role": "system", 
                                        "content": f"以下是调用MCP服务 '{service_name}' 的函数 '{function_name}' 的结果，请使用这些结果回答用户的问题:\n\n```json\n{json.dumps(mcp_result, ensure_ascii=False, indent=2)}\n```"
                                    })
                                else:
                                    error_msg = f"请求的MCP服务ID {service_id} 不在当前智能体关联的服务列表中"
                                    print(error_msg)
                                    yield {"event": "mcp_error", "data": json.dumps({
                                        "object": "chat.completion.mcp_error",
                                        "error": error_msg
                                    }, ensure_ascii=False)}
                            except Exception as e:
                                error_msg = f"处理MCP服务调用时出错: {str(e)}"
                                print(error_msg)
                                import traceback
                                traceback.print_exc()
                                yield {"event": "mcp_error", "data": json.dumps({
                                    "object": "chat.completion.mcp_error",
                                    "error": error_msg
                                }, ensure_ascii=False)}
                        else:
                            reason = detection_result.get('reason', '无原因')
                            print(f"分析结果表明不需要调用MCP服务: {reason}")
                            # 如果是由于解析失败导致的，通知前端
                            if "无法解析" in reason or "解析错误" in reason or "解析失败" in reason or "解析JSON" in reason:
                                yield {"event": "mcp_error", "data": json.dumps({
                                    "object": "chat.completion.mcp_error",
                                    "error": f"解析MCP服务需求失败: {reason}"
                                }, ensure_ascii=False)}
                    else:
                        print(f"当前智能体没有关联的MCP服务")
                        yield {"event": "mcp_error", "data": json.dumps({
                            "object": "chat.completion.mcp_error",
                            "error": "当前智能体没有关联的MCP服务"
                        }, ensure_ascii=False)}
                except Exception as e:
                    print(f"处理MCP服务时出错: {e}")
                    traceback.print_exc()
                    # 通知前端MCP服务处理错误
                    yield {"event": "mcp_error", "data": json.dumps({
                        "object": "chat.completion.mcp_error",
                        "error": f"处理MCP服务时出错: {str(e)}"
                    }, ensure_ascii=False)}
            
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
            
            # 检查是否有文件内容相关的消息，如果有，添加额外的指导消息
            has_file_content = any("以下是用户上传的文件内容" in msg.get("content", "") for msg in final_messages if msg.get("role") == "system")
            print(f"是否有文件内容: {has_file_content}, 用户消息: {user_message}")
            print(f"最终消息列表: {[msg.get('role') for msg in final_messages]}")
            
            # 如果没有检测到文件内容但有文件ID，强制添加一个提示
            if not has_file_content and file_ids:
                print(f"未检测到文件内容但有文件ID，强制添加文件内容提示")
                # 从数据库中获取文件内容
                from app.models.file import File as FileModel
                
                file_contents = []
                for file_id in file_ids:
                    file = db.query(FileModel).filter(FileModel.id == file_id).first()
                    if file and file.text_content:
                        formatted_content = f"--- 文件: {file.original_filename} ({file.file_type}) ---\n\n{file.text_content}\n\n"
                        file_contents.append(formatted_content)
                        print(f"添加文件内容: {file.original_filename}, 长度: {len(file.text_content)}")
                
                if file_contents:
                    combined_content = "\n".join(file_contents)
                    file_context_message = {
                        "role": "system", 
                        "content": f"以下是用户上传的文件内容，这是非常重要的上下文信息，请务必仔细阅读并在回答问题时参考这些内容。如果用户询问关于文件内容的问题，请直接基于这些内容回答：\n\n{combined_content}"
                    }
                    final_messages.insert(0, file_context_message)
                    has_file_content = True
                    print(f"强制添加文件内容到对话上下文: {combined_content[:100]}...")
            
            if has_file_content and ("文档说的什么" in user_message or "文档说了什么" in user_message or "文件内容是什么" in user_message or "文件说了什么" in user_message or "文件说的什么" in user_message):
                # 添加额外的系统消息，指导模型如何回答
                guidance_message = {
                    "role": "system",
                    "content": "用户正在询问文件内容。请直接回答文件的内容是什么，不要回避或者说找不到相关信息。文件内容已经在之前的系统消息中提供。"
                }
                final_messages.append(guidance_message)
                print("添加了额外的指导消息，引导模型回答文件内容相关问题")
            
            # 调用大模型生成回答
            model_response = await execute_model_inference(
                db,
                model_id,
                {
                    "messages": final_messages,
                    "stream": True,
                    **config
                }
            )
            response_content = ""
            # 处理流式响应
            async for chunk in model_response:
                # 检查chunk的类型，如果是工具调用，使用特定的事件类型
                if isinstance(chunk, dict) and chunk.get("choices"):
                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    
                    if delta.get("tool_calls"):
                        # 工具调用事件
                        yield {"event": "tool_calls", "data":chunk}
                        time.sleep(0.1)
                    elif choice.get("finish_reason") == "tool_calls":
                        # 工具调用结果事件
                        yield {"event": "tool_call_result", "data":chunk}
                        time.sleep(0.1)
                    else:
                        # 普通消息块
                        yield {"event": "message_chunk", "data":chunk}
                else:
                    # 默认消息块
                    try:
                        response_content += json.loads(chunk)["choices"][0]["delta"]["content"]
                    except Exception as e:
                        print("response_content += json.loads(chunk)::",response_content,"\n\n:::",chunk)
                        pass
                    # print(response_content)
                    yield {"event": "message_chunk", "data":chunk}
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
                        
                # 构建额外数据
                extra_data = {
                    "response_time_ms": response_time,
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
        import traceback  # 确保在异常处理块内也能访问traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"聊天处理失败: {str(e)}"
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
                            from app.core.minio_client import get_file_stream, RAW_BUCKET
                            
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
            
            # 构建消息列表
            final_messages = [{"role": "system", "content": system_prompt}]
            
            # 添加历史消息
            for msg in messages:
                if msg.role in ["user", "assistant", "system"]:
                    final_messages.append({"role": msg.role, "content": msg.content})
            
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
    from sqlalchemy import distinct, func
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
# @router.post("/{agent_id}/share-chat/{share_id}/chat")
# async def share_chat_with_agent(
#     agent_id: str,
#     share_id: str,
#     request_data: dict = Body(...),
#     db: Session = Depends(get_db),
#     user_id: Optional[str] = Query("000000", description="用户ID，默认为000000表示游客"),
# ):
#     """
#     通过分享链接与智能体对话
#     """
#     print("share_chat_with_agent:::", request_data)
#     # 验证分享链接是否有效
#     share_token = db.query(AgentShareToken).filter(
#         AgentShareToken.token == share_id,
#         AgentShareToken.is_active == True
#     ).first()
    
#     if not share_token:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="分享链接不存在或已失效"
#         )
    
#     # 验证智能体是否存在
#     agent = db.query(Agent).filter(Agent.id == agent_id).first()
#     if not agent:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="智能体不存在"
#         )
    
#     # 验证分享功能是否启用
#     if not agent.share_enabled:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="该智能体未启用分享功能"
#         )
    
#     # 获取请求数据
#     message = request_data.get("message", "")
#     session_id = request_data.get("session_id", f"share_{int(time.time())}")
    
#     if not message:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="消息内容不能为空"
#         )
    
#     try:
#         # 获取模型
#         model_id = agent.model_id
#         if not model_id:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="该智能体未关联模型"
#             )
        
#         model = agent_utils.get_model(db, model_id)
#         if not model:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"模型不存在: {model_id}"
#             )
        
#         # 记录开始时间
#         start_time = time.time()
        
#         # 构建系统提示词
#         system_prompt = agent.system_prompt or "你是一个智能助手，请回答用户的问题。"
        
#         # 构建消息列表
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": message}
#         ]
#         # 准备模型参数
#         model_params = {
#             "messages": messages,
#             "stream": False
#         }
        
#         # 添加配置参数
#         config = agent.config or {}
#         if "temperature" in config:
#             model_params["temperature"] = float(config["temperature"])
#         if "max_tokens" in config:
#             model_params["max_tokens"] = int(config["max_tokens"])
#         if "top_p" in config:
#             model_params["top_p"] = float(config["top_p"])
#         if "frequency_penalty" in config:
#             model_params["frequency_penalty"] = float(config["frequency_penalty"])
        
#         # 调用模型
#         return await agent_utils.execute_model_inference(db, model_id, model_params)
#     except Exception as e:
#         print(f"分享聊天处理失败: {e}")
#         traceback.print_exc()
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"处理失败: {str(e)}"
#         )
#         # print("response:::", response)
# #         # 提取响应内容
# #         response_content = ""
# #         used_tokens = 0
        
# #         if isinstance(response, dict):
# #             if "choices" in response and len(response["choices"]) > 0:
# #                 choice = response["choices"][0]
# #                 if "message" in choice and "content" in choice["message"]:
# #                     response_content = choice["message"]["content"]
            
# #             if "usage" in response and "total_tokens" in response["usage"]:
# #                 used_tokens = response["usage"]["total_tokens"]
        
# #         # 计算响应时间
# #         response_time = int((time.time() - start_time) * 1000)  # 毫秒
        
# #         # 准备额外数据
# #         extra_data = {
# #             "tokens_used": used_tokens
# #         }
        
# #         # 创建聊天历史记录
# #         try:
# #             agent_utils.create_chat_history(
# #                 db=db,
# #                 agent_id=agent_id,
# #                 session_id=session_id,
# #                 user_id=user_id,  # 使用传入的user_id
# #                 user_message=message,
# #                 agent_response=response_content,
# #                 tokens_used=used_tokens,
# #                 response_time=response_time,
# #                 extra_data=extra_data,
# #                 access_type="share",
# #                 share_token_id=share_id,
# #                 model_id=model_id  # 添加模型ID
# #             )
# #         except Exception as e:
# #             print(f"记录聊天历史失败: {e}")
# #             traceback.print_exc()
        
# #         # 更新分享Token的使用次数和最后使用时间
# #         share_token.usage_count += 1
# #         share_token.last_used_at = datetime.now()
# #         db.commit()
        
# #         # 返回响应
# #         return {
# #             "message": response_content,
# #             "tokens_used": used_tokens,
# #             "response_time_ms": response_time
# #         }
        
# #     except Exception as e:
# #         print(f"分享聊天处理失败: {e}")
# #         traceback.print_exc()
# #         raise HTTPException(
# #             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
# #             detail=f"处理失败: {str(e)}"
# #         )
        
        
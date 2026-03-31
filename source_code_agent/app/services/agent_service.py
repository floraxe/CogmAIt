from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
import uuid
import json
import asyncio

from ..models.agent import Agent, AgentType
from ..models.model import Model
from ..models.knowledge import Knowledge
from ..models.graph import KnowledgeGraph
from ..db import db
from ..providers import model_provider_factory
from ..services import knowledge_service, graph_service


async def get_agents(user_id: str) -> List[Dict[str, Any]]:
    """获取用户的所有智能体"""
    agents = await db.agents.find({"user_id": user_id}).to_list(length=100)
    return [agent.dict() for agent in agents]


async def get_agent_types() -> List[Dict[str, Any]]:
    """获取所有支持的智能体类型"""
    return [
        {
            "id": "text",
            "name": "文本智能体",
            "description": "基础的文本交互智能体",
            "icon": "chat",
            "requires": ["model_id"]
        },
        {
            "id": "knowledge",
            "name": "知识库智能体",
            "description": "基于知识库的智能体",
            "icon": "database",
            "requires": ["model_id", "knowledge_id"]
        },
        {
            "id": "graph",
            "name": "知识图谱智能体",
            "description": "基于知识图谱的智能体",
            "icon": "share-alt",
            "requires": ["model_id", "graph_id"]
        },
        {
            "id": "hybrid",
            "name": "混合智能体",
            "description": "同时使用知识库和知识图谱的智能体",
            "icon": "appstore",
            "requires": ["model_id", "knowledge_id", "graph_id"]
        }
    ]


async def create_agent(
    user_id: str,
    name: str,
    description: str,
    agent_type: str,
    model_id: Optional[str] = None,
    knowledge_id: Optional[str] = None,
    graph_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """创建新的智能体"""
    # 验证必要参数
    if not name or not agent_type:
        raise ValueError("智能体名称和类型不能为空")
    
    # 验证agent_type是否有效
    valid_types = [t["id"] for t in await get_agent_types()]
    if agent_type not in valid_types:
        raise ValueError(f"不支持的智能体类型: {agent_type}")
    
    # 根据智能体类型验证必要参数
    if agent_type in ["text", "knowledge", "graph", "hybrid"] and not model_id:
        raise ValueError("模型ID不能为空")
    
    if agent_type in ["knowledge", "hybrid"] and not knowledge_id:
        raise ValueError("知识库ID不能为空")
    
    if agent_type in ["graph", "hybrid"] and not graph_id:
        raise ValueError("知识图谱ID不能为空")
    
    # 验证模型是否存在
    if model_id:
        model = await db.models.find_one({"_id": model_id, "user_id": user_id})
        if not model:
            raise ValueError(f"找不到指定的模型: {model_id}")
    
    # 验证知识库是否存在
    if knowledge_id:
        knowledge = await db.knowledge.find_one({"_id": knowledge_id, "user_id": user_id})
        if not knowledge:
            raise ValueError(f"找不到指定的知识库: {knowledge_id}")
    
    # 验证知识图谱是否存在
    if graph_id:
        graph = await db.knowledge_graphs.find_one({"_id": graph_id, "user_id": user_id})
        if not graph:
            raise ValueError(f"找不到指定的知识图谱: {graph_id}")
    
    # 创建智能体
    agent_id = str(uuid.uuid4().hex)
    now = datetime.utcnow()
    
    agent = Agent(
        id=agent_id,
        user_id=user_id,
        name=name,
        description=description,
        type=agent_type,
        model_id=model_id,
        knowledge_id=knowledge_id,
        graph_id=graph_id,
        system_prompt=system_prompt or "",
        config=config or {},
        created_at=now,
        updated_at=now
    )
    
    await db.agents.insert_one(agent.dict())
    return agent.dict()


async def update_agent(
    agent_id: str,
    user_id: str,
    update_data: Dict[str, Any]
) -> Dict[str, Any]:
    """更新智能体信息"""
    # 验证智能体是否存在
    agent = await db.agents.find_one({"_id": agent_id, "user_id": user_id})
    if not agent:
        raise ValueError(f"找不到指定的智能体: {agent_id}")
    
    agent_dict = agent.dict()
    
    # 获取需要更新的字段
    update_fields = {}
    for key, value in update_data.items():
        if key in agent_dict and key not in ["id", "user_id", "created_at"]:
            update_fields[key] = value
    
    if not update_fields:
        return agent_dict  # 没有需要更新的字段
    
    # 更新最后修改时间
    update_fields["updated_at"] = datetime.utcnow()
    
    # 更新智能体
    await db.agents.update_one(
        {"_id": agent_id, "user_id": user_id},
        {"$set": update_fields}
    )
    
    # 返回更新后的智能体
    updated_agent = await db.agents.find_one({"_id": agent_id, "user_id": user_id})
    return updated_agent.dict()


async def delete_agent(agent_id: str, user_id: str) -> None:
    """删除智能体"""
    # 验证智能体是否存在
    agent = await db.agents.find_one({"_id": agent_id, "user_id": user_id})
    if not agent:
        raise ValueError(f"找不到指定的智能体: {agent_id}")
    
    # 删除智能体
    await db.agents.delete_one({"_id": agent_id, "user_id": user_id})


async def _prepare_agent_context(agent: Agent, user_id: str) -> str:
    """准备智能体的上下文信息"""
    context = ""
    
    # 如果使用知识库，获取知识库内容
    if agent.knowledge_id:
        knowledge_content = await knowledge_service.get_knowledge_content(
            agent.knowledge_id, 
            user_id
        )
        if knowledge_content:
            context += f"知识库内容:\n{knowledge_content}\n\n"
    
    # 如果使用知识图谱，获取图谱内容
    if agent.graph_id:
        graph_content = await graph_service.get_graph_content(
            agent.graph_id, 
            user_id
        )
        if graph_content:
            context += f"知识图谱内容:\n{graph_content}\n\n"
    
    return context


async def chat_with_agent(
    agent_id: str, 
    user_id: str, 
    messages: List[Dict[str, str]],
    stream: bool = False
) -> Any:
    """与智能体对话"""
    # 获取智能体信息
    agent_doc = await db.agents.find_one({"_id": agent_id, "user_id": user_id})
    if not agent_doc:
        raise ValueError(f"找不到指定的智能体: {agent_id}")
    
    agent = Agent(**agent_doc)
    
    # 获取模型信息
    model_doc = await db.models.find_one({"_id": agent.model_id, "user_id": user_id})
    if not model_doc:
        raise ValueError(f"找不到智能体使用的模型: {agent.model_id}")
    
    model = Model(**model_doc)
    
    # 实例化模型提供者
    provider = model_provider_factory.create_provider(model.provider)
    
    # 准备系统提示词
    system_prompt = agent.system_prompt or "你是一个有帮助的AI助手。"
    
    # 根据智能体类型准备上下文
    context = await _prepare_agent_context(agent, user_id)
    if context:
        system_prompt = f"{system_prompt}\n\n{context}"
    
    # 准备完整的消息列表
    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages.extend(messages)
    
    # 调用模型API进行对话
    try:
        # 流式输出
        if stream:
            return provider.chat_completion(
                model_id=model.model_id,
                messages=full_messages,
                api_key=model.api_key,
                base_url=model.base_url,
                stream=True,
                **agent.config
            )
        
        # 非流式输出
        response = await provider.chat_completion(
            model_id=model.model_id,
            messages=full_messages,
            api_key=model.api_key,
            base_url=model.base_url,
            stream=False,
            **agent.config
        )
        
        return response
    except Exception as e:
        raise ValueError(f"与模型对话失败: {str(e)}")


async def test_agent(
    agent_id: str, 
    user_id: str, 
    messages: List[Dict[str, str]],
    stream: bool = False
) -> Any:
    """测试智能体对话（不保存对话历史）"""
    # 调用相同的对话逻辑，但可以添加一些测试专用的处理
    return await chat_with_agent(agent_id, user_id, messages, stream=stream) 
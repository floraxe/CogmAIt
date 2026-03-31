from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import uuid
import secrets
import string

from sqlalchemy.orm import Session
from sqlalchemy import or_, and_

from app.models.agent import Agent, AgentChatHistory, AgentApiKey, AgentShareToken
from app.schemas.agent import AgentCreate, AgentUpdate
from app.models.model import Model
from app.utils.graph import get_graph


def get_model(db: Session, model_id: str) -> Optional[Model]:
    """
    通过ID获取模型
    
    参数:
        db (Session): 数据库会话
        model_id (str): 模型ID
    
    返回:
        Optional[Model]: 模型对象或None
    """
    return db.query(Model).filter(Model.id == model_id).first()


def get_agent(db: Session, agent_id: str) -> Optional[Agent]:
    """
    通过ID获取智能体

    参数:
        db (Session): 数据库会话
        agent_id (str): 智能体ID

    返回:
        Optional[Agent]: 智能体对象或None
    """
    return db.query(Agent).filter(Agent.id == agent_id).first()


def get_agents(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    name: Optional[str] = None,
    type: Optional[str] = None,
    status: Optional[str] = None
) -> List[Agent]:
    """
    获取智能体列表，支持过滤

    参数:
        db (Session): 数据库会话
        skip (int): 跳过的记录数
        limit (int): 限制返回的记录数
        name (Optional[str]): 按名称过滤
        type (Optional[str]): 按类型过滤
        status (Optional[str]): 按状态过滤

    返回:
        List[Agent]: 智能体列表
    """
    query = db.query(Agent)

    # 应用过滤条件
    if name:
        query = query.filter(Agent.name.ilike(f"%{name}%"))
    if type:
        query = query.filter(Agent.type == type)
    if status:
        query = query.filter(Agent.status == status)

    # 应用分页
    return query.order_by(Agent.created_at.desc()).offset(skip).limit(limit).all()


def count_agents(
    db: Session,
    name: Optional[str] = None,
    type: Optional[str] = None,
    status: Optional[str] = None
) -> int:
    """
    计算符合条件的智能体总数

    参数:
        db (Session): 数据库会话
        name (Optional[str]): 按名称过滤
        type (Optional[str]): 按类型过滤
        status (Optional[str]): 按状态过滤

    返回:
        int: 符合条件的智能体总数
    """
    query = db.query(Agent)

    # 应用过滤条件
    if name:
        query = query.filter(Agent.name.ilike(f"%{name}%"))
    if type:
        query = query.filter(Agent.type == type)
    if status:
        query = query.filter(Agent.status == status)

    return query.count()


def create_agent(db: Session, agent_in: AgentCreate, creator: Optional[str] = None, user_id: Optional[str] = None) -> Agent:
    """
    创建新智能体

    参数:
        db (Session): 数据库会话
        agent_in (AgentCreate): 智能体创建模式
        creator (Optional[str]): 创建者名称（兼容旧版本）
        user_id (Optional[str]): 用户ID，外键关联到User表

    返回:
        Agent: 创建的智能体
    """
    # 创建智能体对象的属性字典
    agent_attrs = {
        "id": str(uuid.uuid4().hex),
        "name": agent_in.name,
        "type": agent_in.type,
        "description": agent_in.description,
        "system_prompt": agent_in.system_prompt,
        "welcome_message": agent_in.welcome_message,
        "config": agent_in.config,
        "creator": creator,
        "user_id": user_id,  # 添加用户ID
        "status": "active"
    }
    
    # 只有当 model_id 不为 None 时才添加到属性字典
    if agent_in.model_id is not None:
        agent_attrs["model_id"] = agent_in.model_id
    
    # 创建智能体对象
    db_agent = Agent(**agent_attrs)

    # 添加到数据库
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)

    # 处理知识库关联
    if agent_in.knowledge_ids:
        from app.utils.knowledge import get_knowledge
        for knowledge_id in agent_in.knowledge_ids:
            kb = get_knowledge(db, knowledge_id)
            if kb:
                db_agent.knowledge_bases.append(kb)

    # 处理知识图谱关联
    if agent_in.graph_ids:
        from app.utils.graph import get_graph
        for graph_id in agent_in.graph_ids:
            graph = get_graph(db, graph_id)
            if graph:
                db_agent.graphs.append(graph)

    # 处理MCP服务关联
    if agent_in.mcp_service_ids:
        from app.utils.mcp import get_mcp_service
        for service_id in agent_in.mcp_service_ids:
            service = get_mcp_service(db, service_id)
            if service:
                db_agent.mcp_services.append(service)

    # 保存关联
    if agent_in.knowledge_ids or agent_in.graph_ids or agent_in.mcp_service_ids:
        db.add(db_agent)
        db.commit()
        db.refresh(db_agent)

    return db_agent


def update_agent(db: Session, agent: Agent, agent_in: AgentUpdate) -> Agent:
    """
    更新智能体信息

    参数:
        db (Session): 数据库会话
        agent (Agent): 要更新的智能体
        agent_in (AgentUpdate): 智能体更新模式

    返回:
        Agent: 更新后的智能体
    """
    # 获取更新数据
    update_data = agent_in.dict(exclude_unset=True)

    # 分离特殊处理的字段
    knowledge_ids = update_data.pop("knowledge_ids", None)
    graph_ids = update_data.pop("graph_ids", None)
    mcp_service_ids = update_data.pop("mcp_service_ids", None)

    # 更新基本字段
    for field, value in update_data.items():
        if hasattr(agent, field) and value is not None:
            setattr(agent, field, value)

    # 更新时间戳
    agent.updated_at = datetime.utcnow()

    # 保存基本信息
    db.add(agent)
    db.commit()
    db.refresh(agent)

    # 处理知识库关联
    if knowledge_ids is not None:
        from app.utils.knowledge import get_knowledge
        # 清除现有关联
        agent.knowledge_bases = []
        # 添加新关联
        for knowledge_id in knowledge_ids:
            kb = get_knowledge(db, knowledge_id)
            if kb:
                agent.knowledge_bases.append(kb)

    # 处理知识图谱关联
    if graph_ids is not None:
        from app.utils.graph import get_graph
        # 清除现有关联
        agent.graphs = []
        # 添加新关联
        for graph_id in graph_ids:
            graph = get_graph(db, graph_id)
            if graph:
                agent.graphs.append(graph)
                
    # 处理MCP服务关联
    if mcp_service_ids is not None:
        from app.utils.mcp import get_mcp_service
        # 清除现有关联
        agent.mcp_services = []
        # 添加新关联
        for service_id in mcp_service_ids:
            service = get_mcp_service(db, service_id)
            if service:
                agent.mcp_services.append(service)

    # 保存关联更新
    if knowledge_ids is not None or graph_ids is not None or mcp_service_ids is not None:
        db.add(agent)
        db.commit()
        db.refresh(agent)

    return agent


def delete_agent(db: Session, agent_id: str) -> None:
    """
    删除智能体

    参数:
        db (Session): 数据库会话
        agent_id (str): 智能体ID
    """
    agent = get_agent(db, agent_id)
    if agent:
        db.delete(agent)
        db.commit()

# 聊天历史相关函数可以在后续实现 

def create_chat_history(
    db: Session,
    agent_id: str,
    session_id: str,
    user_message: str,
    agent_response: str,
    user_id: Optional[str] = None,
    tokens_used: int = 0,
    response_time: int = 0,
    extra_data: Optional[Dict[str, Any]] = None,
    access_type: str = "user",
    api_key_id: Optional[str] = None,
    share_token_id: Optional[str] = None,
    type: Optional[str] = None,
    model_id: Optional[str] = None,  # 添加model_id参数
) -> AgentChatHistory:
    """
    创建智能体聊天历史记录
    
    参数:
        db (Session): 数据库会话
        agent_id (str): 智能体ID
        session_id (str): 会话ID
        user_message (str): 用户消息
        agent_response (str): 智能体响应
        user_id (Optional[str], optional): 用户ID. Defaults to None.
        tokens_used (int, optional): 使用的令牌数. Defaults to 0.
        response_time (int, optional): 响应时间（毫秒）. Defaults to 0.
        extra_data (Optional[Dict[str, Any]], optional): 额外数据. Defaults to None.
        access_type (str, optional): 访问类型. Defaults to "user".
        api_key_id (Optional[str], optional): API密钥ID. Defaults to None.
        share_token_id (Optional[str], optional): 分享Token ID. Defaults to None.
        type (Optional[str], optional): 会话类型. Defaults to None.
        model_id (Optional[str], optional): 生成回复使用的模型ID. Defaults to None.
        
    返回:
        AgentChatHistory: 创建的聊天历史记录
    """
    print("user_id:::222222222:", user_id)
    db_history = AgentChatHistory(
        id=str(uuid.uuid4().hex),
        agent_id=agent_id,
        session_id=session_id,
        user_id=user_id,
        user_message=user_message,
        agent_response=agent_response,
        tokens_used=tokens_used,
        response_time=response_time,
        extra_data=extra_data,
        access_type=access_type,
        api_key_id=api_key_id,
        share_token_id=share_token_id,
        type=type,
        model_id=model_id  # 添加model_id字段
    )
    
    db.add(db_history)
    db.commit()
    db.refresh(db_history)
    
    return db_history

def get_chat_history(
    db: Session,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100
) -> List[AgentChatHistory]:
    """
    获取聊天历史记录
    
    参数:
        db (Session): 数据库会话
        agent_id (Optional[str]): 按智能体ID过滤
        session_id (Optional[str]): 按会话ID过滤
        user_id (Optional[str]): 按用户ID过滤
        skip (int): 跳过的记录数
        limit (int): 限制返回的记录数
        
    返回:
        List[AgentChatHistory]: 聊天历史记录列表
    """
    query = db.query(AgentChatHistory)
    
    if agent_id:
        query = query.filter(AgentChatHistory.agent_id == agent_id)
    if session_id:
        query = query.filter(AgentChatHistory.session_id == session_id)
    if user_id:
        query = query.filter(AgentChatHistory.user_id == user_id)
    
    return query.order_by(AgentChatHistory.created_at.desc()).offset(skip).limit(limit).all()

def count_chat_history(
    db: Session,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> int:
    """
    计算聊天历史记录数量
    
    参数:
        db (Session): 数据库会话
        agent_id (Optional[str]): 按智能体ID过滤
        session_id (Optional[str]): 按会话ID过滤
        user_id (Optional[str]): 按用户ID过滤
        
    返回:
        int: 记录数量
    """
    query = db.query(AgentChatHistory)
    
    if agent_id:
        query = query.filter(AgentChatHistory.agent_id == agent_id)
    if session_id:
        query = query.filter(AgentChatHistory.session_id == session_id)
    if user_id:
        query = query.filter(AgentChatHistory.user_id == user_id)
    
    return query.count()

async def execute_model_inference(
    db: Session, 
    model_id: str, 
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    执行模型推理
    
    参数:
        db (Session): 数据库会话
        model_id (str): 模型ID
        payload (Dict[str, Any]): 请求负载
    
    返回:
        Dict[str, Any]: 模型推理结果
    """
    from app.utils.model import execute_model_inference as model_inference
    
    return await model_inference(db, model_id, payload)

def get_graph(db: Session, graph_id: str):
    """
    通过ID获取知识图谱
    
    参数:
        db (Session): 数据库会话
        graph_id (str): 知识图谱ID
    
    返回:
        Optional[Graph]: 知识图谱对象或None
    """
    from app.utils.graph import get_graph as get_graph_util
    return get_graph_util(db, graph_id) 

def generate_random_token(length=48):
    """
    生成随机令牌
    
    参数:
        length (int): 令牌长度
        
    返回:
        str: 生成的随机令牌
    """
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def generate_share_token(db: Session, agent_id: str, name: Optional[str] = None) -> Tuple[str, str]:
    """
    为智能体生成分享令牌
    
    参数:
        db (Session): 数据库会话
        agent_id (str): 智能体ID
        name (Optional[str]): Token名称/描述
        
    返回:
        Tuple[str, str]: 令牌ID和值
    """
    # 获取智能体
    agent = get_agent(db, agent_id)
    if not agent:
        raise ValueError("智能体不存在")
        
    # 生成随机令牌
    token = generate_random_token(32)
    
    # 创建新的ShareToken记录
    share_token = AgentShareToken(
        id=str(uuid.uuid4().hex),
        agent_id=agent_id,
        token=token,
        name=name or f"分享链接 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    
    # 保存到数据库
    db.add(share_token)
    
    # 确保启用分享
    if not agent.share_enabled:
        agent.share_enabled = True
    
    # 保存更改
    db.commit()
    db.refresh(share_token)
    
    return share_token.id, token


def generate_api_key(db: Session, agent_id: str, name: Optional[str] = None) -> Tuple[str, str]:
    """
    为智能体生成API密钥
    
    参数:
        db (Session): 数据库会话
        agent_id (str): 智能体ID
        name (Optional[str]): 密钥名称/描述
        
    返回:
        Tuple[str, str]: 密钥ID和值
    """
    # 获取智能体
    agent = get_agent(db, agent_id)
    if not agent:
        raise ValueError("智能体不存在")
        
    # 生成随机密钥
    api_key = generate_random_token(48)
    
    # 创建新的ApiKey记录
    api_key_obj = AgentApiKey(
        id=str(uuid.uuid4().hex),
        agent_id=agent_id,
        key=api_key,
        name=name or f"API密钥 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    
    # 保存到数据库
    db.add(api_key_obj)
    
    # 确保启用API访问
    if not agent.api_enabled:
        agent.api_enabled = True
    
    # 保存更改
    db.commit()
    db.refresh(api_key_obj)
    
    return api_key_obj.id, api_key


def delete_api_key(db: Session, key_id: str) -> bool:
    """
    删除API密钥
    
    参数:
        db (Session): 数据库会话
        key_id (str): API密钥ID
    
    返回:
        bool: 是否成功删除
    """
    # 查找API密钥记录
    api_key = db.query(AgentApiKey).filter(AgentApiKey.id == key_id).first()
    if not api_key:
        return False
    
    # 删除API密钥
    db.delete(api_key)
    db.commit()
    
    return True


def delete_share_token(db: Session, token_id: str) -> bool:
    """
    删除分享Token
    
    参数:
        db (Session): 数据库会话
        token_id (str): 分享TokenID
    
    返回:
        bool: 是否成功删除
    """
    # 查找分享Token记录
    share_token = db.query(AgentShareToken).filter(AgentShareToken.id == token_id).first()
    if not share_token:
        return False
    
    # 删除分享Token
    db.delete(share_token)
    db.commit()
    
    return True


def toggle_share_status(db: Session, agent_id: str, enabled: bool) -> bool:
    """
    切换智能体分享状态
    
    参数:
        db (Session): 数据库会话
        agent_id (str): 智能体ID
        enabled (bool): 是否启用分享
        
    返回:
        bool: 是否成功更新
    """
    # 获取智能体
    agent = get_agent(db, agent_id)
    if not agent:
        return False
        
    # 更新分享状态
    agent.share_enabled = enabled
    
    # 保存更改
    db.commit()
    db.refresh(agent)
    
    return True


def toggle_api_status(db: Session, agent_id: str, enabled: bool) -> bool:
    """
    切换智能体API访问状态
    
    参数:
        db (Session): 数据库会话
        agent_id (str): 智能体ID
        enabled (bool): 是否启用API访问
        
    返回:
        bool: 是否成功更新
    """
    # 获取智能体
    agent = get_agent(db, agent_id)
    if not agent:
        return False
        
    # 更新API访问状态
    agent.api_enabled = enabled
    
    # 保存更改
    db.commit()
    db.refresh(agent)
    
    return True
    
    
def get_agent_by_share_token(db: Session, token: str) -> Optional[Agent]:
    """
    通过分享令牌获取智能体
    
    参数:
        db (Session): 数据库会话
        token (str): 分享令牌
        
    返回:
        Optional[Agent]: 智能体对象或None
    """
    # 先查找新版的share_token记录
    share_token = db.query(AgentShareToken).join(Agent, AgentShareToken.agent_id == Agent.id).filter(AgentShareToken.token == token).first()
    # if share_token:
        # 更新使用次数和最后使用时间
    share_token.usage_count += 1
    share_token.last_used_at = datetime.now()
    db.commit()
        
        # 返回关联的智能体
    return share_token.agent
    
    # 兼容旧版本：查找具有该分享令牌的智能体
    # return db.query(Agent).filter(
    #     Agent.share_token == token,
    #     Agent.share_enabled == True
    # ).first()


def get_agent_by_api_key(db: Session, api_key: str) -> Optional[Tuple[Agent, str]]:
    """
    通过API密钥获取智能体
    
    参数:
        db (Session): 数据库会话
        api_key (str): API密钥
        
    返回:
        Optional[Tuple[Agent, str]]: 智能体对象和API密钥ID的元组，或None
    """
    # 先查找新版的api_key记录
    api_key_obj = db.query(AgentApiKey).filter(AgentApiKey.key == api_key).first()
    if api_key_obj:
        # 检查是否激活
        if not api_key_obj.is_active:
            return None
            
        # 更新使用次数和最后使用时间
        api_key_obj.usage_count += 1
        api_key_obj.last_used_at = datetime.now()
        db.commit()
        
        # 返回关联的智能体和密钥ID
        return api_key_obj.agent, api_key_obj.id
    
    # 兼容旧版本：查找具有该API密钥的智能体
    agent = db.query(Agent).filter(
        Agent.api_key == api_key,
        Agent.api_enabled == True
    ).first() 
    
    if agent:
        return agent, "legacy"
    
    return None


def get_agent_api_keys(db: Session, agent_id: str) -> List[AgentApiKey]:
    """
    获取智能体的所有API密钥
    
    参数:
        db (Session): 数据库会话
        agent_id (str): 智能体ID
    
    返回:
        List[AgentApiKey]: API密钥列表
    """
    return db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).all()


def get_agent_share_tokens(db: Session, agent_id: str) -> List[AgentShareToken]:
    """
    获取智能体的所有分享Token
    
    参数:
        db (Session): 数据库会话
        agent_id (str): 智能体ID
    
    返回:
        List[AgentShareToken]: 分享Token列表
    """
    return db.query(AgentShareToken).filter(AgentShareToken.agent_id == agent_id).all() 
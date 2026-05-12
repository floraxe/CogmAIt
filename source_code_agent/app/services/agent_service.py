"""
智能体持久化与令牌、会话聚合等业务逻辑。

路由层与编排层应调用本模块，避免在 app.utils 中堆叠业务实现。
"""
from __future__ import annotations

import secrets
import string
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from app.models.agent import Agent, AgentApiKey, AgentChatHistory, AgentShareToken
from app.models.model import Model
from app.schemas.agent import AgentCreate, AgentUpdate

if TYPE_CHECKING:
    from app.models.user import User

def get_model(db: Session, model_id: str) -> Optional[Model]:
    return db.query(Model).filter(Model.id == model_id).first()


def get_agent(db: Session, agent_id: str) -> Optional[Agent]:
    return db.query(Agent).filter(Agent.id == agent_id).first()


def get_agents(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    name: Optional[str] = None,
    type: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Agent]:
    query = db.query(Agent)
    if name:
        query = query.filter(Agent.name.ilike(f"%{name}%"))
    if type:
        query = query.filter(Agent.type == type)
    if status:
        query = query.filter(Agent.status == status)
    return query.order_by(Agent.created_at.desc()).offset(skip).limit(limit).all()


def count_agents(
    db: Session,
    name: Optional[str] = None,
    type: Optional[str] = None,
    status: Optional[str] = None,
) -> int:
    query = db.query(Agent)
    if name:
        query = query.filter(Agent.name.ilike(f"%{name}%"))
    if type:
        query = query.filter(Agent.type == type)
    if status:
        query = query.filter(Agent.status == status)
    return query.count()


def create_agent(
    db: Session,
    agent_in: AgentCreate,
    *,
    owner: Optional["User"] = None,
) -> Agent:
    """创建智能体；归属信息仅来自 owner（user_id / creator / created_by 一致）。"""
    uid = str(owner.id) if owner else None
    uname = owner.username if owner else None
    agent_attrs: Dict[str, Any] = {
        "id": str(uuid.uuid4().hex),
        "name": agent_in.name,
        "type": agent_in.type,
        "description": agent_in.description,
        "system_prompt": agent_in.system_prompt,
        "welcome_message": agent_in.welcome_message,
        "config": agent_in.config,
        "creator": uname,
        "created_by": uid,
        "user_id": uid,
        "status": "active",
    }
    if agent_in.model_id is not None:
        agent_attrs["model_id"] = agent_in.model_id

    db_agent = Agent(**agent_attrs)
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)

    if agent_in.knowledge_ids:
        from app.utils.knowledge import get_knowledge

        for knowledge_id in agent_in.knowledge_ids:
            kb = get_knowledge(db, knowledge_id)
            if kb:
                db_agent.knowledge_bases.append(kb)

    if agent_in.graph_ids:
        from app.utils.graph import get_graph

        for graph_id in agent_in.graph_ids:
            graph = get_graph(db, graph_id)
            if graph:
                db_agent.graphs.append(graph)

    if agent_in.mcp_service_ids:
        from app.utils.mcp import get_mcp_service

        for service_id in agent_in.mcp_service_ids:
            service = get_mcp_service(db, service_id)
            if service:
                db_agent.mcp_services.append(service)

    if agent_in.knowledge_ids or agent_in.graph_ids or agent_in.mcp_service_ids:
        db.add(db_agent)
        db.commit()
        db.refresh(db_agent)

    return db_agent


def update_agent(db: Session, agent: Agent, agent_in: AgentUpdate) -> Agent:
    update_data = agent_in.dict(exclude_unset=True)
    knowledge_ids = update_data.pop("knowledge_ids", None)
    graph_ids = update_data.pop("graph_ids", None)
    mcp_service_ids = update_data.pop("mcp_service_ids", None)

    for field, value in update_data.items():
        if hasattr(agent, field) and value is not None:
            setattr(agent, field, value)

    agent.updated_at = datetime.utcnow()
    db.add(agent)
    db.commit()
    db.refresh(agent)

    if knowledge_ids is not None:
        from app.utils.knowledge import get_knowledge

        agent.knowledge_bases = []
        for knowledge_id in knowledge_ids:
            kb = get_knowledge(db, knowledge_id)
            if kb:
                agent.knowledge_bases.append(kb)

    if graph_ids is not None:
        from app.utils.graph import get_graph

        agent.graphs = []
        for graph_id in graph_ids:
            graph = get_graph(db, graph_id)
            if graph:
                agent.graphs.append(graph)

    if mcp_service_ids is not None:
        from app.utils.mcp import get_mcp_service

        agent.mcp_services = []
        for service_id in mcp_service_ids:
            service = get_mcp_service(db, service_id)
            if service:
                agent.mcp_services.append(service)

    if knowledge_ids is not None or graph_ids is not None or mcp_service_ids is not None:
        db.add(agent)
        db.commit()
        db.refresh(agent)

    return agent


def delete_agent(db: Session, agent_id: str) -> None:
    agent = get_agent(db, agent_id)
    if agent:
        db.delete(agent)
        db.commit()


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
    model_id: Optional[str] = None,
) -> AgentChatHistory:
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
        model_id=model_id,
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
    limit: int = 100,
) -> List[AgentChatHistory]:
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
    user_id: Optional[str] = None,
) -> int:
    query = db.query(AgentChatHistory)
    if agent_id:
        query = query.filter(AgentChatHistory.agent_id == agent_id)
    if session_id:
        query = query.filter(AgentChatHistory.session_id == session_id)
    if user_id:
        query = query.filter(AgentChatHistory.user_id == user_id)
    return query.count()


def get_chat_sessions(
    db: Session,
    agent_id: str,
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
) -> Tuple[List[Dict[str, Any]], int]:
    """按会话聚合；返回 (当前页摘要列表, 符合条件的会话总数)。"""
    filters = [AgentChatHistory.agent_id == agent_id]
    if session_id:
        filters.append(AgentChatHistory.session_id == session_id)
    if user_id:
        filters.append(AgentChatHistory.user_id == user_id)
    combined = and_(*filters)

    total = (
        db.query(func.count(func.distinct(AgentChatHistory.session_id)))
        .filter(combined)
        .scalar()
    ) or 0

    rows = (
        db.query(
            AgentChatHistory.session_id,
            func.max(AgentChatHistory.created_at).label("last_message"),
            func.count(AgentChatHistory.id).label("message_count"),
            func.min(AgentChatHistory.created_at).label("first_message"),
            func.min(AgentChatHistory.user_message).label("first_user_message"),
            func.min(AgentChatHistory.type).label("type"),
        )
        .filter(combined)
        .group_by(AgentChatHistory.session_id)
        .order_by(func.max(AgentChatHistory.created_at).desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    items = [
        {
            "sessionId": row.session_id,
            "lastMessage": row.last_message,
            "messageCount": row.message_count,
            "firstMessage": row.first_message,
            "firstUserMessage": row.first_user_message,
            "type": row.type,
        }
        for row in rows
    ]
    return items, int(total)


async def execute_model_inference(
    db: Session,
    model_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    from app.utils.model import execute_model_inference as model_inference

    return await model_inference(db, model_id, payload)


def get_graph(db: Session, graph_id: str):
    from app.utils.graph import get_graph as get_graph_util

    return get_graph_util(db, graph_id)


def generate_random_token(length: int = 48) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_share_token(db: Session, agent_id: str, name: Optional[str] = None) -> Tuple[str, str]:
    agent = get_agent(db, agent_id)
    if not agent:
        raise ValueError("智能体不存在")

    token = generate_random_token(32)
    share_token = AgentShareToken(
        id=str(uuid.uuid4().hex),
        agent_id=agent_id,
        token=token,
        name=name or f"分享链接 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    )
    db.add(share_token)
    if not agent.share_enabled:
        agent.share_enabled = True
    db.commit()
    db.refresh(share_token)
    return share_token.id, token


def generate_api_key(db: Session, agent_id: str, name: Optional[str] = None) -> Tuple[str, str]:
    agent = get_agent(db, agent_id)
    if not agent:
        raise ValueError("智能体不存在")

    api_key = generate_random_token(48)
    api_key_obj = AgentApiKey(
        id=str(uuid.uuid4().hex),
        agent_id=agent_id,
        key=api_key,
        name=name or f"API密钥 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    )
    db.add(api_key_obj)
    if not agent.api_enabled:
        agent.api_enabled = True
    db.commit()
    db.refresh(api_key_obj)
    return api_key_obj.id, api_key


def delete_api_key(db: Session, agent_id: str, key_id: str) -> bool:
    api_key = (
        db.query(AgentApiKey)
        .filter(AgentApiKey.id == key_id, AgentApiKey.agent_id == agent_id)
        .first()
    )
    if not api_key:
        return False
    db.delete(api_key)
    db.commit()
    return True


def delete_share_token(db: Session, agent_id: str, token_id: str) -> bool:
    share_token = (
        db.query(AgentShareToken)
        .filter(AgentShareToken.id == token_id, AgentShareToken.agent_id == agent_id)
        .first()
    )
    if not share_token:
        return False
    db.delete(share_token)
    db.commit()
    return True


def toggle_share_status(db: Session, agent_id: str, enabled: bool) -> bool:
    agent = get_agent(db, agent_id)
    if not agent:
        return False
    agent.share_enabled = enabled
    db.commit()
    db.refresh(agent)
    return True


def toggle_api_status(db: Session, agent_id: str, enabled: bool) -> bool:
    agent = get_agent(db, agent_id)
    if not agent:
        return False
    agent.api_enabled = enabled
    db.commit()
    db.refresh(agent)
    return True


def get_agent_by_share_token(db: Session, token: str) -> Optional[Agent]:
    share_token = (
        db.query(AgentShareToken)
        .join(Agent, AgentShareToken.agent_id == Agent.id)
        .filter(AgentShareToken.token == token)
        .first()
    )
    if share_token:
        share_token.usage_count += 1
        share_token.last_used_at = datetime.now()
        db.commit()
        return share_token.agent

    return (
        db.query(Agent)
        .filter(
            Agent.share_token == token,
            Agent.share_enabled == True,
        )
        .first()
    )


def get_agent_by_api_key(db: Session, api_key: str) -> Optional[Tuple[Agent, str]]:
    api_key_obj = db.query(AgentApiKey).filter(AgentApiKey.key == api_key).first()
    if api_key_obj:
        if not api_key_obj.is_active:
            return None
        api_key_obj.usage_count += 1
        api_key_obj.last_used_at = datetime.now()
        db.commit()
        return api_key_obj.agent, api_key_obj.id

    agent = db.query(Agent).filter(
        Agent.api_key == api_key,
        Agent.api_enabled == True,
    ).first()
    if agent:
        return agent, "legacy"
    return None


def get_agent_api_keys(db: Session, agent_id: str) -> List[AgentApiKey]:
    return db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).all()


def get_agent_share_tokens(db: Session, agent_id: str) -> List[AgentShareToken]:
    return db.query(AgentShareToken).filter(AgentShareToken.agent_id == agent_id).all()

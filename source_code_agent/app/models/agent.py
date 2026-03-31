import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from sqlalchemy import Column, String, DateTime, JSON, Boolean, Integer, Float, ForeignKey, Text, Table
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.mysql import LONGTEXT
from app.db.base import Base, get_cn_datetime
from app.utils import format_datetime


# 智能体与知识库的多对多关联表
agent_knowledge_association = Table(
    'agent_knowledge_association',
    Base.metadata,
    Column('agent_id', String(36), ForeignKey('agents.id'), primary_key=True),
    Column('knowledge_id', String(36), ForeignKey('knowledge.id'), primary_key=True)
)

# 智能体与知识图谱的多对多关联表
agent_graph_association = Table(
    'agent_graph_association',
    Base.metadata,
    Column('agent_id', String(36), ForeignKey('agents.id'), primary_key=True),
    Column('graph_id', String(36), ForeignKey('graphs.id'), primary_key=True)
)

# 智能体与MCP服务的多对多关联表
agent_mcp_service_association = Table(
    'agent_mcp_service_association',
    Base.metadata,
    Column('agent_id', String(36), ForeignKey('agents.id'), primary_key=True),
    Column('service_id', String(36), ForeignKey('mcp_services.id'), primary_key=True)
)


class Agent(Base):
    """
    智能体数据库模型
    
    存储智能体基本信息，如名称、类型、关联的模型等
    """
    __tablename__ = "agents"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    name = Column(String(100), index=True, nullable=False)
    type = Column(String(50), index=True, nullable=False)  # qa_bot, knowledge_assistant, graph_qa, rag 等
    description = Column(String(500), nullable=True)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=True)
    config = Column(JSON, nullable=True)  # 存储智能体配置，如temperature, maxTokens等
    system_prompt = Column(Text, nullable=True)  # 系统提示词
    welcome_message = Column(Text, nullable=True)  # 开场白
    status = Column(String(20), default="active", index=True)  # active, inactive
    created_at = Column(DateTime, default=get_cn_datetime)
    updated_at = Column(DateTime, default=get_cn_datetime, onupdate=get_cn_datetime)
    creator = Column(String(100), nullable=True)  # 创建者名称（兼容旧版本）
    created_by = Column(String(36), nullable=True)  # 创建者ID（兼容旧版本）
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)  # 用户ID，外键关联到User表
    avatar = Column(LONGTEXT, nullable=True)  # 头像的base64编码
    enable_web_search = Column(Boolean, default=False)  # 是否启用网络搜索
    
    # 发布渠道相关字段 (兼容旧版本)
    share_token = Column(String(100), unique=True, index=True, nullable=True)  # 免登录窗口访问令牌
    share_enabled = Column(Boolean, default=False)  # 是否启用免登录窗口访问
    api_key = Column(String(100), unique=True, index=True, nullable=True)  # API访问的密钥
    api_enabled = Column(Boolean, default=False)  # 是否启用API访问
    
    # 关联
    model = relationship("Model")
    knowledge_bases = relationship("Knowledge", secondary=agent_knowledge_association)
    graphs = relationship("Graph", secondary=agent_graph_association)
    mcp_services = relationship("MCPService", secondary=agent_mcp_service_association)
    chat_history = relationship("AgentChatHistory", back_populates="agent", cascade="all, delete-orphan")
    user = relationship("User", back_populates="agents")  # 关联到创建该智能体的用户
    
    # 新增多个API密钥和分享Token的关联
    api_keys = relationship("AgentApiKey", back_populates="agent", cascade="all, delete-orphan")
    share_tokens = relationship("AgentShareToken", back_populates="agent", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """将智能体转换为字典表示形式"""
        # 获取API密钥和分享Token列表
        api_keys_list = []
        if self.api_keys:
            api_keys_list = [key.to_dict() for key in self.api_keys]
        
        share_tokens_list = []
        if self.share_tokens:
            share_tokens_list = [token.to_dict() for token in self.share_tokens]
            
        # 兼容旧版本
        if self.api_key and not api_keys_list:
            api_keys_list.append({
                "id": "legacy",
                "key": self.api_key,
                "created_at": format_datetime(self.created_at),
                "last_used_at": None,
                "usage_count": 0
            })
            
        if self.share_token and not share_tokens_list:
            share_tokens_list.append({
                "id": "legacy",
                "token": self.share_token,
                "created_at": format_datetime(self.created_at),
                "last_used_at": None,
                "usage_count": 0
            })
        
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "modelId": self.model_id,
            "config": self.config or {},
            "systemPrompt": self.system_prompt,
            "welcomeMessage": self.welcome_message,
            "knowledgeIds": [kb.id for kb in self.knowledge_bases] if self.knowledge_bases else [],
            "graphIds": [graph.id for graph in self.graphs] if self.graphs else [],
            "mcpServiceIds": [service.id for service in self.mcp_services] if self.mcp_services else [],
            "status": self.status,
            "created": format_datetime(self.created_at),
            "lastModified": format_datetime(self.updated_at),
            "creator": self.creator,
            "createdBy": self.created_by,
            "userId": self.user_id,
            "avatar": self.avatar,
            "enable_web_search": self.enable_web_search,
            "shareToken": self.share_token,  # 兼容旧版本
            "shareEnabled": self.share_enabled,
            "apiKey": self.api_key,  # 兼容旧版本
            "apiEnabled": self.api_enabled,
            "apiKeys": api_keys_list,  # 新版多API密钥
            "shareTokens": share_tokens_list  # 新版多分享Token
        }


class AgentChatHistory(Base):
    """
    智能体聊天历史数据库模型
    
    存储用户与智能体的聊天记录
    """
    __tablename__ = "agent_chat_history"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=False)
    type = Column(String(36),default="web")
    session_id = Column(String(36), index=True, nullable=False)  # 会话ID
    user_id = Column(String(36), index=True, nullable=True)  # 用户ID，可为空表示未登录用户
    user_message = Column(Text, nullable=False)  # 用户消息
    agent_response = Column(Text, nullable=False)  # 智能体响应
    tokens_used = Column(Integer, default=0)  # 使用的令牌数
    response_time = Column(Integer, default=0)  # 响应时间（毫秒）
    extra_data = Column(JSON, nullable=True)  # 额外元数据，如使用的知识库、引用的文档等
    created_at = Column(DateTime, default=get_cn_datetime)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=True)  # 用于记录生成回复时使用的模型ID
    
    # 访问方式相关字段
    access_type = Column(String(20), default="user")  # 访问类型: user, share, api
    api_key_id = Column(String(36), nullable=True)  # 使用的API密钥ID
    share_token_id = Column(String(36), nullable=True)  # 使用的分享Token ID
    
    # 关联
    agent = relationship("Agent", back_populates="chat_history")
    model = relationship("Model")  # 关联到模型表
    
    def to_dict(self) -> Dict[str, Any]:
        """将聊天记录转换为字典表示形式"""
        return {
            "id": self.id,
            "agentId": self.agent_id,
            "sessionId": self.session_id,
            "userId": self.user_id,
            "userMessage": self.user_message,
            "agentResponse": self.agent_response,
            "tokensUsed": self.tokens_used,
            "responseTime": self.response_time,
            "extra_data": self.extra_data or {},
            "created": format_datetime(self.created_at),
            "accessType": self.access_type,
            "apiKeyId": self.api_key_id,
            "shareTokenId": self.share_token_id,
            "modelId": self.model_id
        }


class AgentApiKey(Base):
    """
    智能体API密钥模型
    """
    __tablename__ = "agent_api_keys"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=False)
    key = Column(String(100), unique=True, index=True, nullable=False)  # API密钥
    name = Column(String(100), nullable=True)  # 密钥名称/描述
    created_at = Column(DateTime, default=get_cn_datetime)
    last_used_at = Column(DateTime, nullable=True)  # 最后使用时间
    usage_count = Column(Integer, default=0)  # 使用次数
    is_active = Column(Boolean, default=True)  # 是否激活
    
    # 关联
    agent = relationship("Agent", back_populates="api_keys")
    
    def to_dict(self) -> Dict[str, Any]:
        """将API密钥转换为字典表示形式"""
        return {
            "id": self.id,
            "agentId": self.agent_id,
            "key": self.key,
            "name": self.name,
            "created": format_datetime(self.created_at),
            "lastUsed": format_datetime(self.last_used_at) if self.last_used_at else None,
            "usageCount": self.usage_count,
            "isActive": self.is_active
        }


class AgentShareToken(Base):
    """
    智能体分享Token模型
    """
    __tablename__ = "agent_share_tokens"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=False)
    token = Column(String(100), unique=True, index=True, nullable=False)  # 分享Token
    name = Column(String(100), nullable=True)  # Token名称/描述
    created_at = Column(DateTime, default=get_cn_datetime)
    last_used_at = Column(DateTime, nullable=True)  # 最后使用时间
    usage_count = Column(Integer, default=0)  # 使用次数
    is_active = Column(Boolean, default=True)  # 是否激活
    
    # 关联
    agent = relationship("Agent", back_populates="share_tokens")
    
    def to_dict(self) -> Dict[str, Any]:
        """将分享Token转换为字典表示形式"""
        return {
            "id": self.id,
            "agentId": self.agent_id,
            "token": self.token,
            "name": self.name,
            "created": format_datetime(self.created_at),
            "lastUsed": format_datetime(self.last_used_at) if self.last_used_at else None,
            "usageCount": self.usage_count,
            "isActive": self.is_active
        } 
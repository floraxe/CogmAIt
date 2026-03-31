from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Union


class AgentBase(BaseModel):
    """智能体基础模式"""
    name: str
    type: str
    description: Optional[str] = None
    avatar: Optional[str] = None
    model_id: Optional[str] = None
    system_prompt: Optional[str] = None
    welcome_message: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    enable_web_search: Optional[bool] = False
    status: Optional[str] = "active"
    creator: Optional[str] = None
    
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=lambda field_name: ''.join(word.capitalize() if i else word for i, word in enumerate(field_name.split('_')))
    )


class AgentCreate(AgentBase):
    """创建智能体请求模式"""
    knowledge_ids: Optional[List[str]] = Field(default_factory=list, alias="knowledgeIds")
    graph_ids: Optional[List[str]] = Field(default_factory=list, alias="graphIds")
    mcp_service_ids: Optional[List[str]] = Field(default_factory=list, alias="mcpServiceIds")


class AgentUpdate(BaseModel):
    """更新智能体请求模式"""
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    avatar: Optional[str] = None
    model_id: Optional[str] = Field(default=None, alias="modelId")
    system_prompt: Optional[str] = Field(default=None, alias="systemPrompt")
    welcome_message: Optional[str] = Field(default=None, alias="welcomeMessage")
    config: Optional[Dict[str, Any]] = None
    enable_web_search: Optional[bool] = False
    status: Optional[str] = None
    knowledge_ids: Optional[List[str]] = Field(default=None, alias="knowledgeIds")
    graph_ids: Optional[List[str]] = Field(default=None, alias="graphIds")
    mcp_service_ids: Optional[List[str]] = Field(default=None, alias="mcpServiceIds")
    
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=lambda field_name: ''.join(word.capitalize() if i else word for i, word in enumerate(field_name.split('_')))
    )


class AgentStatistics(BaseModel):
    """智能体使用统计"""
    queries: int = 0
    tokens_used: int = 0
    avg_response_time: float = 0.0


class AgentResponse(AgentBase):
    """智能体响应模式"""
    id: str
    knowledge_ids: List[str] = Field(default_factory=list, alias="knowledgeIds")
    graph_ids: List[str] = Field(default_factory=list, alias="graphIds")
    mcp_service_ids: List[str] = Field(default_factory=list, alias="mcpServiceIds")
    status: str
    created: str = Field(alias="created_at")
    last_modified: Optional[str] = Field(default=None, alias="lastModified")
    statistics: Optional[AgentStatistics] = None
    avatar: Optional[str] = None
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        protected_namespaces=(),
        alias_generator=lambda field_name: ''.join(word.capitalize() if i else word for i, word in enumerate(field_name.split('_')))
    )


class AgentListResponse(BaseModel):
    """智能体列表响应"""
    total: int
    items: List[AgentResponse]


class AgentTypeInfo(BaseModel):
    """智能体类型信息"""
    id: int
    name: str
    value: str


class AgentChatMessage(BaseModel):
    """聊天消息模式"""
    role: str
    content: str


class AgentChatRequest(BaseModel):
    """智能体聊天请求"""
    messages: List[AgentChatMessage]
    stream: bool = False
    session_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    type: Optional[str] = None
    file_ids: Optional[List[str]] = Field(default=None, alias="fileIds")  # 添加文件ID列表字段


class AgentChatResponse(BaseModel):
    """智能体聊天响应"""
    message: AgentChatMessage
    tokens_used: int = 0
    response_time: int = 0  # 毫秒
    sources: Optional[List[Dict[str, Any]]] = None
    file_ids: Optional[List[str]] = Field(default=None, alias="fileIds")  # 添加文件ID列表字段


class AgentChatHistoryResponse(BaseModel):
    """聊天历史响应"""
    id: str
    agent_id: str = Field(alias="agentId")
    session_id: str = Field(alias="sessionId")
    user_message: str = Field(alias="userMessage")
    agent_response: str = Field(alias="agentResponse")
    tokens_used: int = Field(alias="tokensUsed")
    response_time: int = Field(alias="responseTime")
    extra_data: Dict[str, Any] = Field(default_factory=dict, alias="extraData")
    created: str = Field(alias="created_at")
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        protected_namespaces=(),
        alias_generator=lambda field_name: ''.join(word.capitalize() if i else word for i, word in enumerate(field_name.split('_')))
    )


class AgentChatHistoryListResponse(BaseModel):
    """聊天历史列表响应"""
    total: int
    items: List[AgentChatHistoryResponse] 
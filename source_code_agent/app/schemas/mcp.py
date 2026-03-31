from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel
from datetime import datetime

# MCP Service Schemas

class MCPServiceBase(BaseModel):
    name: str
    description: Optional[str] = None
    type: str
    provider: str
    icon: Optional[str] = None
    config_template: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    deployment_type: Optional[str] = "local"
    pricing: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None
    api_endpoints: Optional[List[Dict[str, Any]]] = None
    examples: Optional[List[Dict[str, Any]]] = None
    usage_docs: Optional[str] = None
    is_official: Optional[bool] = False
    owner_id: Optional[str] = None
    github_url: Optional[str] = None


class MCPServiceCreate(MCPServiceBase):
    # 添加创建特有字段
    id: Optional[str] = None  # 允许客户端指定ID，否则后端生成


class MCPServiceUpdate(BaseModel):
    # 所有字段都是可选的，用于更新
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    provider: Optional[str] = None
    icon: Optional[str] = None
    config_template: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    deployment_type: Optional[str] = None
    pricing: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None
    api_endpoints: Optional[List[Dict[str, Any]]] = None
    examples: Optional[List[Dict[str, Any]]] = None
    usage_docs: Optional[str] = None
    is_official: Optional[bool] = None
    owner_id: Optional[str] = None
    github_url: Optional[str] = None


class MCPServiceResponse(MCPServiceBase):
    id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True # Use from_attributes instead of orm_mode


class MCPServiceListResponse(BaseModel):
    total: int
    items: List[MCPServiceResponse]

# User Service Connection Schemas

class UserServiceConnectionBase(BaseModel):
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = "inactive" # Default status
    error_message: Optional[str] = None
    connection_type: Optional[str] = "api_key"
    usage_count: Optional[int] = 0


class UserServiceConnectionCreate(BaseModel):
    service_id: str
    config: Optional[Dict[str, Any]] = None
    connection_type: Optional[str] = "api_key"
    # user_id will be taken from the authenticated user


class UserServiceConnectionUpdate(UserServiceConnectionBase):
    # All fields are optional for update
    last_used_at: Optional[datetime] = None
    pass


class UserServiceConnectionResponse(UserServiceConnectionBase):
    id: int
    user_id: str
    service_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None

    class Config:
        from_attributes = True # Use from_attributes instead of orm_mode 
        
# Agent MCP Service Connection Schemas

class AgentServiceConnectionBase(BaseModel):
    service_id: str
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = "active"
    
class AgentServiceConnectionCreate(AgentServiceConnectionBase):
    agent_id: str
    
class AgentServiceConnectionUpdate(BaseModel):
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    
class AgentServiceConnectionResponse(AgentServiceConnectionBase):
    id: int
    agent_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
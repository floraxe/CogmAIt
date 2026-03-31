from datetime import datetime
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field, ConfigDict


class ModelBase(BaseModel):
    """模型基础模式，包含共享属性"""
    name: str
    provider: str
    type: str = Field(..., description="模型类型: chat, completion, embedding")
    description: Optional[str] = None
    base_url: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    icon: Optional[str] = Field(default=None, description="模型图标名称")
    tool_call_support: bool = Field(default=False, description="是否支持工具调用")
    function_call_support: bool = Field(default=False, description="是否支持函数调用")
    vision_support: bool = Field(default=False, description="是否支持图片识别")
    thinking_support: bool = Field(default=False, description="是否支持输出思考过程")
    mcp_support: bool = Field(default=False, description="是否支持MCP服务")
    default_prompt: Optional[str] = Field(default=None, description="默认提示词")
    max_context_length: Optional[int] = Field(default=4000, description="最大上下文长度")
    extra_body_params: Optional[Dict[str, Any]] = Field(default=None, description="额外调用参数")


class ModelCreate(ModelBase):
    """创建模型时使用的模式"""
    api_key: str
    

class ModelUpdate(BaseModel):
    """更新模型时使用的模式"""
    name: Optional[str] = None
    provider: Optional[str] = None
    type: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    icon: Optional[str] = None
    tool_call_support: Optional[bool] = None
    function_call_support: Optional[bool] = None
    vision_support: Optional[bool] = None
    thinking_support: Optional[bool] = None
    mcp_support: Optional[bool] = None
    default_prompt: Optional[str] = None
    max_context_length: Optional[int] = None
    extra_body_params: Optional[Dict[str, Any]] = None


class ModelResponse(ModelBase):
    """响应中返回的模型模式"""
    id: str
    status: str
    created: str
    updated: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class ModelListResponse(BaseModel):
    """模型列表响应"""
    total: int
    items: List[ModelResponse]


class ModelTestConnection(BaseModel):
    """模型连接测试请求"""
    model_id: str


class ModelTestResponse(BaseModel):
    """模型连接测试响应"""
    status: str
    message: str
    response: Optional[Dict[str, Any]] = None


class ProviderInfo(BaseModel):
    """提供商信息"""
    value: str
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    default_base_url: Optional[str] = None
    supported_types: List[str] = Field(default_factory=list, description="支持的模型类型")
    features: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(protected_namespaces=()) 
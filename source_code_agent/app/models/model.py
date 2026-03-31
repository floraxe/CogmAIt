import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, DateTime, JSON, Boolean, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship

from app.db.base import Base, get_cn_datetime
from app.utils import format_datetime


class Model(Base):
    """
    AI模型数据库模型
    
    存储模型相关信息，如名称、提供商、API密钥等
    """
    __tablename__ = "models"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    name = Column(String(100), index=True, nullable=False)
    provider = Column(String(50), index=True, nullable=False)
    type = Column(String(20), index=True, nullable=False)  # chat, completion, embedding
    api_key = Column(String(255), nullable=False)
    base_url = Column(String(255), nullable=True)
    description = Column(String(500), nullable=True)
    config = Column(JSON, nullable=True)  # 存储模型特定配置
    status = Column(String(20), default="active", index=True)  # active, inactive
    created_at = Column(DateTime, default=get_cn_datetime)
    updated_at = Column(DateTime, default=get_cn_datetime, onupdate=get_cn_datetime)
    icon = Column(String(1000), nullable=True)  # 模型图标，存储文件名或路径
    created_by = Column(String(36), nullable=True)  # 创建者ID（保留向后兼容）
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)  # 添加用户ID外键

    # 新增字段
    tool_call_support = Column(Boolean, default=False, nullable=False)
    function_call_support = Column(Boolean, default=False, nullable=False)
    vision_support = Column(Boolean, default=False, nullable=False)
    thinking_support = Column(Boolean, default=False, nullable=False)
    mcp_support = Column(Boolean, default=False, nullable=False)  # 添加MCP服务支持字段
    default_prompt = Column(Text, nullable=True)
    max_context_length = Column(Integer, default=4000, nullable=True)
    extra_body_params = Column(JSON, nullable=True)  # 存储额外的API调用参数
    
    # 关联关系
    datasources = relationship("DataSource", back_populates="model")
    user = relationship("User")  # 添加与User的关联
    
    def to_dict(self) -> Dict[str, Any]:
        """将模型转换为字典表示形式"""
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "type": self.type,
            "base_url": self.base_url,
            "description": self.description,
            "config": self.config,
            "status": self.status,
            "icon": self.icon,
            "created": format_datetime(self.created_at),
            "updated": format_datetime(self.updated_at),
            "created_by": self.created_by,
            "userId": self.user_id,  # 添加用户ID到返回字典
            # 新增字段
            "tool_call_support": self.tool_call_support,
            "function_call_support": self.function_call_support,
            "vision_support": self.vision_support,
            "thinking_support": self.thinking_support,
            "mcp_support": self.mcp_support,  # 添加MCP服务支持字段到返回数据中
            "default_prompt": self.default_prompt,
            "max_context_length": self.max_context_length,
            "extra_body_params": self.extra_body_params
        }
        
    @property
    def is_active(self) -> bool:
        """检查模型是否处于活动状态"""
        return self.status == "active" 
from sqlalchemy import Column, String, Integer, JSON, ForeignKey, Boolean, Text, DateTime
from sqlalchemy.orm import relationship
import datetime

from app.db.base import Base


class MCPService(Base):
    __tablename__ = "mcp_services"

    id = Column(String(255), primary_key=True, index=True)
    name = Column(String(255), index=True, nullable=False)
    description = Column(String(500), nullable=True)
    type = Column(String(50), index=True, nullable=False) # e.g., "llm", "embedding"
    provider = Column(String(255), index=True, nullable=False)
    icon = Column(String(255), nullable=True)
    config_template = Column(JSON, nullable=True) # JSON schema or template for user config
    
    # 新增字段
    tags = Column(JSON, nullable=True) # 服务标签
    deployment_type = Column(String(50), default="local") # local 或 hosted
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    pricing = Column(JSON, nullable=True) # 价格信息
    statistics = Column(JSON, nullable=True) # 统计信息
    api_endpoints = Column(JSON, nullable=True) # API端点信息
    examples = Column(JSON, nullable=True) # 示例代码
    usage_docs = Column(Text, nullable=True) # 使用文档
    is_official = Column(Boolean, default=False) # 是否为官方推荐服务
    owner_id = Column(String(255), nullable=True) # 服务所有者ID
    github_url = Column(String(255), nullable=True) # GitHub仓库URL

    connections = relationship("UserServiceConnection", back_populates="service")


class UserServiceConnection(Base):
    __tablename__ = "user_service_connections"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), ForeignKey("users.id"), nullable=False)
    service_id = Column(String(255), ForeignKey("mcp_services.id"), nullable=False)
    config = Column(JSON, nullable=True) # User provided configuration (e.g., API Key)
    status = Column(String(50), default="inactive") # e.g., "active", "inactive", "error"
    error_message = Column(String(500), nullable=True)
    
    # 新增字段
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0) # 使用次数
    connection_type = Column(String(50), default="api_key") # api_key 或 oauth

    user = relationship("User", back_populates="service_connections")
    service = relationship("MCPService", back_populates="connections") 
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from sqlalchemy import Column, String, DateTime, JSON, Boolean, Integer, ForeignKey, Text
from sqlalchemy.orm import relationship

from app.db.base import Base, get_cn_datetime
from app.utils import format_datetime


class User(Base):
    """
    用户数据库模型
    
    存储用户基本信息，如用户名、密码、角色等
    """
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)
    name = Column(String(100), index=True, nullable=True)
    email = Column(String(100), unique=True, index=True, nullable=True)
    phone = Column(String(20), unique=True, index=True, nullable=True)
    role = Column(String(20), default="user", index=True)  # admin, operator, user, guest
    department = Column(String(50), nullable=True)
    position = Column(String(50), nullable=True)
    avatar = Column(Text, nullable=True)
    status = Column(String(20), default="active", index=True)  # active, inactive
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=get_cn_datetime)
    updated_at = Column(DateTime, default=get_cn_datetime, onupdate=get_cn_datetime)
    
    # 关联关系
    agents = relationship("Agent", back_populates="user")
    datasources = relationship("DataSource", back_populates="user")
    service_connections = relationship("UserServiceConnection", back_populates="user")
    
    def to_dict(self) -> Dict[str, Any]:
        """将用户转换为字典表示形式（不包含敏感信息）"""
        return {
            "id": self.id,
            "username": self.username,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "role": self.role,
            "department": self.department,
            "position": self.position,
            "avatar": self.avatar,
            "status": self.status,
            "lastLogin": format_datetime(self.last_login),
            "created": format_datetime(self.created_at)
        }
    
    @property
    def is_active(self) -> bool:
        """检查用户是否处于活动状态"""
        return self.status == "active"
    

class Role(Base):
    """
    角色数据库模型
    
    存储系统角色定义
    """
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(50), unique=True, index=True, nullable=False)
    value = Column(String(50), unique=True, index=True, nullable=False)
    description = Column(String(255), nullable=True)
    permissions = Column(JSON, nullable=True)  # 存储角色权限
    created_at = Column(DateTime, default=get_cn_datetime)
    updated_at = Column(DateTime, default=get_cn_datetime, onupdate=get_cn_datetime)
    
    def to_dict(self) -> Dict[str, Any]:
        """将角色转换为字典表示形式"""
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "permissions": self.permissions or [],
            "created": format_datetime(self.created_at)
        } 
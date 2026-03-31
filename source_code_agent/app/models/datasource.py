import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, DateTime, JSON, Boolean, ForeignKey, Integer
from sqlalchemy.orm import relationship

from app.db.base import Base, get_cn_datetime
from app.utils import format_datetime


class DataSource(Base):
    """
    数据源数据库模型
    
    存储各类数据库连接信息，如MySQL、PostgreSQL、MongoDB等
    """
    __tablename__ = "datasources"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    name = Column(String(100), index=True, nullable=False)
    type = Column(String(50), index=True, nullable=False)  # mysql, postgresql, sqlserver, mongodb, etc.
    host = Column(String(255), nullable=False)
    port = Column(Integer, nullable=False)
    database = Column(String(255), nullable=False)
    username = Column(String(100), nullable=False)
    password = Column(String(255), nullable=False)
    extra_params = Column(JSON, nullable=True)  # 存储额外参数，如SSL设置等
    model_id = Column(String(36), ForeignKey("models.id"), nullable=True)  # 关联的AI模型ID
    created_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    remark = Column(String(500), nullable=True)  # 备注信息
    created_at = Column(DateTime, default=get_cn_datetime)
    updated_at = Column(DateTime, default=get_cn_datetime, onupdate=get_cn_datetime)
    
    # 关联关系
    model = relationship("Model", back_populates="datasources")
    user = relationship("User", back_populates="datasources")
    queries = relationship("DataSourceQuery", back_populates="datasource", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """将数据源转换为字典表示"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "password": self.password,  # 确保返回密码字符串
            "extra_params": self.extra_params,
            "model_id": self.model_id,
            "is_active": self.is_active,
            "remark": self.remark,
            "created_by": self.created_by,
            "created_at": format_datetime(self.created_at),
            "updated_at": format_datetime(self.updated_at)
        }


class DataSourceQuery(Base):
    """
    数据源查询记录
    
    存储用户执行的查询及结果文件信息
    """
    __tablename__ = "datasource_queries"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    datasource_id = Column(String(36), ForeignKey("datasources.id"), nullable=False)
    query_text = Column(String(4000), nullable=False)  # SQL查询语句
    result_file_id = Column(String(36), ForeignKey("files.id"), nullable=True)  # 关联到文件表的ID
    execution_time = Column(Integer, nullable=True)  # 执行时间(毫秒)
    rows_affected = Column(Integer, nullable=True)  # 影响行数
    status = Column(String(20), default="success")  # success, failed
    error_message = Column(String(500), nullable=True)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=get_cn_datetime)
    
    # 关联关系
    datasource = relationship("DataSource", back_populates="queries")
    result_file = relationship("File")
    user = relationship("User")
    
    def to_dict(self) -> Dict[str, Any]:
        """将查询记录转换为字典表示"""
        return {
            "id": self.id,
            "datasource_id": self.datasource_id,
            "query_text": self.query_text,
            "result_file_id": self.result_file_id,
            "execution_time": self.execution_time,
            "rows_affected": self.rows_affected,
            "status": self.status,
            "error_message": self.error_message,
            "created_by": self.created_by,
            "created_at": format_datetime(self.created_at)
        } 
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, DateTime, JSON, Boolean, Integer, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base, get_cn_datetime
from app.utils import format_datetime


class ExtractionTask(Base):
    """
    知识抽取任务数据库模型
    
    存储知识图谱抽取任务相关信息
    """
    __tablename__ = "extraction_tasks"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    graph_id = Column(String(36), ForeignKey("graphs.id", ondelete="CASCADE"), nullable=False)
    file_id = Column(String(36), ForeignKey("graph_files.id", ondelete="SET NULL"), nullable=True)
    original_file_id = Column(String(36), nullable=True)  # 关联的原始文件ID
    model_id = Column(String(36), nullable=False)
    status = Column(String(20), default="pending")  # pending, running, completed, failed, cancelled
    task_type = Column(String(30), nullable=True)  # 新增字段，用于区分任务类型：llm_extraction, neo4j_extraction等
    prompt = Column(Text, nullable=True)
    parameters = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    message = Column(Text, nullable=True)
    progress = Column(Float, default=0)
    entity_count = Column(Integer, default=0)
    relation_count = Column(Integer, default=0)
    retry_count = Column(Integer, default=0)  # 重试次数
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)  # 添加用户ID外键
    
    # 关联
    graph = relationship("Graph", back_populates="extraction_tasks")
    file = relationship("GraphFile", back_populates="extraction_tasks")
    user = relationship("User")  # 添加与User的关联
    
    def to_dict(self) -> Dict[str, Any]:
        """将抽取任务转换为字典表示形式"""
        return {
            "id": self.id,
            "graphId": self.graph_id,
            "fileId": self.file_id,
            "originalFileId": self.original_file_id,
            "modelId": self.model_id,
            "status": self.status,
            "taskType": self.task_type,  # 新增字段在to_dict方法中也需要返回
            "prompt": self.prompt,
            "parameters": self.parameters,
            "result": self.result,
            "message": self.message,
            "progress": self.progress,
            "entityCount": self.entity_count,
            "relationCount": self.relation_count,
            "retryCount": self.retry_count,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
            "userId": self.user_id  # 添加用户ID到返回字典
        } 
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, DateTime, JSON, Boolean, Integer, Float, ForeignKey, Text
from sqlalchemy.orm import relationship

from app.db.base import Base, get_cn_datetime
from app.utils import format_datetime


class GraphFile(Base):
    """
    知识图谱文件数据库模型
    
    存储上传到知识图谱的文件信息
    """
    __tablename__ = "graph_files"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    graph_id = Column(String(36), ForeignKey("graphs.id"), nullable=True)  # 允许为NULL，以便删除图谱时不影响文件
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=True)  # 原始文件名
    file_type = Column(String(50), nullable=False)  # pdf, doc, txt, csv, etc.
    file_size = Column(Integer, default=0)  # 以字节为单位
    path = Column(String(500), nullable=True)  # 存储路径
    status = Column(String(50), default="uploading")  # uploading, uploaded, processing, parsed, extracting, extracted, completed, failed
    error = Column(Text, nullable=True)  # 错误信息
    extra_data = Column(JSON, nullable=True)  # 文件元数据
    text_content = Column(Text, nullable=True)  # 提取的文本内容
    original_file_id = Column(String(36), nullable=True, index=True)  # 关联到文件管理系统的文件ID
    created_by =  Column(String(36), ForeignKey("users.id"), nullable=True)  # 创建者ID，外键关联到User表
    created_at = Column(DateTime, default=get_cn_datetime)
    updated_at = Column(DateTime, default=get_cn_datetime, onupdate=get_cn_datetime)
    
    # 关联
    graph = relationship("Graph", backref="files")
    extraction_tasks = relationship("ExtractionTask", back_populates="file", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """将文件转换为字典表示形式"""
        return {
            "id": self.id,
            "graphId": self.graph_id,
            "filename": self.filename,
            "originalFilename": self.original_filename,
            "fileType": self.file_type,
            "fileSize": self.file_size,
            "status": self.status,
            "error": self.error,
            "extraData": self.extra_data,
            "created": format_datetime(self.created_at),
            "updated": format_datetime(self.updated_at)
        } 
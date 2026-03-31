import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, DateTime, JSON, Boolean, Integer, Float, ForeignKey,Text, LargeBinary
from sqlalchemy.orm import relationship

from app.db.base import Base, get_cn_datetime
from app.utils import format_datetime


class Knowledge(Base):
    """
    知识库数据库模型
    
    存储知识库相关信息，如名称、描述、文件数量等
    """
    __tablename__ = "knowledge"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    name = Column(String(100), index=True, nullable=False)
    description = Column(Text, nullable=True)
    file_count = Column(Integer, default=0)
    total_size = Column(Integer, default=0)  # 以字节为单位
    vector_type = Column(String(20), default="faiss")  # faiss, milvus, pinecone, qdrant
    embedding_model = Column(String(50), default="openai")
    config = Column(JSON, nullable=True)  # 存储知识库特定配置
    status = Column(String(20), default="active", index=True)  # active, processing, inactive
    created_at = Column(DateTime, default=get_cn_datetime)
    updated_at = Column(DateTime, default=get_cn_datetime, onupdate=get_cn_datetime)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)  # 添加用户ID外键

    # 关联到知识库文件
    files = relationship("KnowledgeFile", back_populates="knowledge", cascade="all, delete-orphan")
    user = relationship("User")  # 添加与User的关联
    
    def to_dict(self) -> Dict[str, Any]:
        """将知识库转换为字典表示形式"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "fileCount": self.file_count,
            "totalSize": self.total_size,
            "vectorType": self.vector_type,
            "embeddingModel": self.embedding_model,
            "config": self.config,
            "status": self.status,
            "created": format_datetime(self.created_at),
            "updated": format_datetime(self.updated_at),
            "userId": self.user_id  # 添加用户ID到返回字典
        }
        
    @property
    def is_active(self) -> bool:
        """检查知识库是否处于活动状态"""
        return self.status == "active"


class KnowledgeFile(Base):
    """
    知识库文件数据库模型
    
    存储上传到知识库的文件信息
    """
    __tablename__ = "knowledge_files"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    knowledge_id = Column(String(36), ForeignKey("knowledge.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=True)  # 原始文件名
    file_type = Column(String(50), nullable=False)  # pdf, doc, txt, csv, etc.
    file_size = Column(Integer, default=0)  # 以字节为单位
    path = Column(String(500), nullable=True)  # 存储路径，与app.utils.knowledge中的字段保持一致
    chunk_count = Column(Integer, default=0)  # 分片数量
    vector_count = Column(Integer, default=0)  # 向量数量
    file_id = Column(String(36), ForeignKey("files.id"), nullable=False)
    status = Column(String(50), default="uploading")
    error = Column(Text, nullable=True)
    extra_data = Column(JSON, nullable=True)  # 文件元数据
    embedding_status = Column(String(50), default="pending")
    embedding_path = Column(String(500), nullable=True)  # 向量存储路径
    created_at = Column(DateTime, default=get_cn_datetime)
    updated_at = Column(DateTime, default=get_cn_datetime, onupdate=get_cn_datetime)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)  # 添加用户ID外键
    
    # 新增字段
    text_content = Column(Text, nullable=True)  # MEDIUMTEXT类型，用于存储提取的文本内容
    text_chunking_result = Column(JSON, nullable=True)  # 存储分段结果
    text_extraction_time = Column(DateTime, nullable=True)  # 文本提取时间
    chunking_time = Column(DateTime, nullable=True)  # 分段时间
    
    # 关联
    knowledge = relationship("Knowledge", back_populates="files")
    user = relationship("User")  # 添加与User的关联
    
    def to_dict(self) -> Dict[str, Any]:
        """将知识库文件转换为字典表示形式"""
        # 从extra_data中提取描述信息
        description = None
        if self.extra_data and isinstance(self.extra_data, dict):
            description = self.extra_data.get("description")
            
        return {
            "id": self.id,
            "knowledgeId": self.knowledge_id,
            "filename": self.filename,
            "originalFilename": self.original_filename,
            "fileType": self.file_type,
            "fileSize": self.file_size,
            "path": self.path,
            "chunkCount": self.chunk_count,
            "vectorCount": self.vector_count,
            "status": self.status,
            "error": self.error,
            "extra_data": self.extra_data,
            "description": description,  # 添加描述字段
            "embeddingStatus": self.embedding_status,
            "created": format_datetime(self.created_at),
            "updated": format_datetime(self.updated_at),
            "hasTextContent": self.text_content is not None,
            "textExtractionTime": format_datetime(self.text_extraction_time) if self.text_extraction_time else None,
            "chunkingTime": format_datetime(self.chunking_time) if self.chunking_time else None,
            "text_chunking_result": self.text_chunking_result if self.text_chunking_result else [],
            "process_status": self.get_process_status(),
            "userId": self.user_id  # 添加用户ID到返回字典
        }
        
    @property
    def is_active(self) -> bool:
        """检查文件是否处于活动状态"""
        return self.status == "active"

    def get_process_status(self) -> str:
        """
        获取文件处理状态的用户友好描述
        
        返回:
            str: 处理状态描述
        """
        if self.status == "failed" or self.embedding_status == "failed":
            return "处理失败"
        elif self.status == "uploading":
            return "上传中"
        elif self.status == "uploaded":
            return "等待处理"
        elif self.status == "parsing":
            return "解析中"
        elif self.status == "parsed":
            return "解析完成"
        elif self.status == "chunking":
            return "分段中"
        elif self.status == "chunked":
            return "分段完成"
        elif self.embedding_status == "processing":
            return "训练中"
        elif self.embedding_status == "completed":
            return "训练完成"
        elif self.status == "indexed":
            return "训练完成"
        else:
            return "处理中" 
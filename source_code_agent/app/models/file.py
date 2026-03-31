import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, DateTime, JSON, Boolean, Integer, Float, ForeignKey, Text
from sqlalchemy.orm import relationship

from app.db.base import Base, get_cn_datetime
from app.utils import format_datetime


class File(Base):
    """
    文件数据库模型
    
    存储上传的通用文件信息
    """
    __tablename__ = "files"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    filename = Column(String(255), nullable=False)                 # 存储的文件名(UUID)
    original_filename = Column(String(255), nullable=False)        # 原始文件名
    file_type = Column(String(50), nullable=False)                 # 文件类型，如pdf, doc, jpg等
    file_size = Column(Integer, default=0)                         # 文件大小(字节)
    path = Column(String(500), nullable=True)                      # 文件存储路径
    status = Column(String(50), default="uploaded")                # 状态: uploaded, processing, processed, failed
    error = Column(Text, nullable=True)                            # 错误信息
    description = Column(Text, nullable=True)                      # 文件描述
    extra_data = Column(JSON, nullable=True)                       # 附加数据
    text_content = Column(Text, nullable=True)                     # 提取的文本内容
    text_extraction_time = Column(DateTime, nullable=True)         # 文本提取时间
    created_at = Column(DateTime, default=get_cn_datetime)         # 创建时间
    updated_at = Column(DateTime, default=get_cn_datetime, onupdate=get_cn_datetime)  # 更新时间
    created_by = Column(String(100), nullable=True)                # 创建者用户名（保留向后兼容）
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)  # 添加用户ID外键
    
    # 新增字段 - 用于关联处理后的文件
    markdown_path = Column(String(500), nullable=True)             # Markdown文件路径
    visual_path = Column(String(500), nullable=True)               # 可视化文件路径
    images_folder = Column(String(500), nullable=True)             # 图片文件夹路径
    
    # 关联
    user = relationship("User")  # 添加与User的关联
    
    def to_dict(self) -> Dict[str, Any]:
        """将文件转换为字典表示形式"""
        # 构建返回的基本字典
        result = {
            "id": self.id,
            "filename": self.filename,
            "originalFilename": self.original_filename,
            "fileType": self.file_type,
            "fileSize": self.file_size,
            "path": self.path,
            "status": self.status,
            "error": self.error,
            "description": self.description,
            "textContent": self.text_content,
            "extraData": self.extra_data,
            "created": format_datetime(self.created_at),
            "updated": format_datetime(self.updated_at),
            "createdBy": self.created_by,
            "userId": self.user_id,  # 添加用户ID到返回字典
            "hasTextContent": self.text_content is not None,
            "textExtractionTime": format_datetime(self.text_extraction_time) if self.text_extraction_time else None,
        }
        
        # 添加处理后的文件路径
        if self.markdown_path:
            result["markdownPath"] = self.markdown_path
        
        if self.visual_path:
            result["visualPath"] = self.visual_path
        
        if self.images_folder:
            result["imagesFolder"] = self.images_folder
        
        return result
    
    def get_status_text(self) -> str:
        """获取状态的友好文本描述"""
        status_map = {
            "uploaded": "已上传",
            "processing": "处理中",
            "processed": "已处理",
            "failed": "处理失败"
        }
        return status_map.get(self.status, self.status) 
    
    def set_related_files(self, markdown_path=None, visual_path=None, images_folder=None):
        """设置与文件相关的处理后文件路径"""
        if markdown_path:
            self.markdown_path = markdown_path
        
        if visual_path:
            self.visual_path = visual_path
        
        if images_folder:
            self.images_folder = images_folder 
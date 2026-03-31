from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class FileBase(BaseModel):
    """文件基础模型"""
    original_filename: Optional[str] = None
    file_type: Optional[str] = None
    description: Optional[str] = None


class FileCreate(FileBase):
    """创建文件模型"""
    pass


class FileUpdate(BaseModel):
    """更新文件模型"""
    original_filename: Optional[str] = None
    description: Optional[str] = None


class FileResponse(BaseModel):
    """文件返回模型"""
    id: str
    filename: str
    originalFilename: Optional[str] = None
    fileType: str
    fileSize: int
    path: Optional[str] = None
    status: str
    error: Optional[str] = None
    extraData: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    created: str
    updated: str
    textContent: Optional[str] = None
    createdBy: Optional[str] = None
    hasTextContent: Optional[bool] = None
    textExtractionTime: Optional[str] = None
    
    # 新增关联文件路径字段
    markdownPath: Optional[str] = None
    visualPath: Optional[str] = None
    imagesFolder: Optional[str] = None
    
    # 文件URL字段，用于前端直接访问
    file_url: Optional[str] = None
    markdown_url: Optional[str] = None
    visual_url: Optional[str] = None
    
    class Config:
        from_attributes = True


class FileListResponse(BaseModel):
    """文件列表返回模型"""
    items: List[FileResponse]
    total: int


class ProcessFilesRequest(BaseModel):
    """处理文件请求模型"""
    file_ids: List[str] 
    vision_model_id: Optional[str] = Field(None, description="用于图像分析的多模态模型ID") 
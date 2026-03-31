from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class ChunkingMethod(str, Enum):
    """分段方法枚举"""
    CHARACTER = "character"  # 按字符分段
    TOKEN = "token"  # 按token分段
    SENTENCE = "sentence"  # 按句子分段
    PARAGRAPH = "paragraph"  # 按段落分段
    SEMANTIC = "semantic"  # 语义分段
    RECURSIVE = "recursive"  # 递归分段


class ChunkingConfig(BaseModel):
    method: str
    chunk_size: int
    chunk_overlap: int
    separator: Optional[str] = None
    separators: Optional[str] = None
    encoding_name: Optional[str] = None
    model_name: Optional[str] = None
    is_separator_regex: Optional[bool] = False
    keep_separator: Optional[bool] = False

# 添加请求模型
class FileProcessRequest(BaseModel):
    chunking_config: ChunkingConfig
    chunks: List[str]

class ChunkingConfig(BaseModel):
    """分段配置模型"""
    method: ChunkingMethod = ChunkingMethod.CHARACTER  # 分段方法
    chunk_size: int = 1000  # 分段大小
    chunk_overlap: int = 200  # 分段重叠
    separator: Optional[str] = None  # 分隔符，仅在某些分段方法中使用
    
    # RecursiveCharacterTextSplitter特有参数
    separators: Optional[List[str]] = None  # 分隔符列表，按优先级降序排列
    
    # TokenTextSplitter特有参数
    encoding_name: Optional[str] = "cl100k_base"  # 编码名称
    
    # SentenceTransformersTokenTextSplitter特有参数
    model_name: Optional[str] = "all-MiniLM-L6-v2"  # 模型名称
    
    # 其他可能的参数
    is_separator_regex: Optional[bool] = False  # 分隔符是否为正则表达式
    keep_separator: Optional[bool] = False  # 是否保留分隔符


class KnowledgeBase(BaseModel):
    """知识库基础模式"""
    name: str
    description: Optional[str] = None
    vector_type: str = ""
    embedding_model: str = ""
    chunking_config: Optional[ChunkingConfig] = Field(default_factory=lambda: ChunkingConfig())
    config: Optional[Dict[str, Any]] = None


class KnowledgeCreate(KnowledgeBase):
    """创建知识库请求模式"""
    pass


class KnowledgeUpdate(BaseModel):
    """更新知识库请求模式"""
    name: Optional[str] = None
    description: Optional[str] = None
    vector_type: Optional[str] = None
    embedding_model: Optional[str] = None
    chunking_config: Optional[ChunkingConfig] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None


class KnowledgeResponse(KnowledgeBase):
    """知识库响应模式"""
    id: str
    file_count: int
    total_size: int
    status: str
    created: str = Field(alias="created_at")
    updated: Optional[str] = Field(default=None, alias="updated_at")
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        protected_namespaces=(),
        alias_generator=lambda field_name: ''.join(
            x if i == 0 else x.capitalize() for i, x in enumerate(field_name.split('_'))
        )
    )


class KnowledgeListResponse(BaseModel):
    """知识库列表响应"""
    total: int
    items: List[KnowledgeResponse]


class KnowledgeFileBase(BaseModel):
    """知识库文件基础模式"""
    original_filename: str
    file_size: int
    file_type: str


class KnowledgeFileResponse(KnowledgeFileBase):
    """知识库文件响应模式"""
    id: str
    knowledge_id: str
    status: str
    chunk_count: int
    vector_count: int
    file_id: str
    filename: str
    created: str = Field(alias="created_at")
    updated: Optional[str] = Field(default=None, alias="updated_at")
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        protected_namespaces=(),
        alias_generator=lambda field_name: ''.join(
            x if i == 0 else x.capitalize() for i, x in enumerate(field_name.split('_'))
        )
    )


class KnowledgeFileListResponse(BaseModel):
    """知识库文件列表响应"""
    total: int
    items: List[KnowledgeFileResponse] 
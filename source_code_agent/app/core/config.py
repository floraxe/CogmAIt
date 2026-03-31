import secrets
import os
import json
from typing import Any, Dict, List, Optional, Union

from pydantic_settings import BaseSettings
from pydantic import field_validator, Field, AnyHttpUrl


class Settings(BaseSettings):
    # 基本设置
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "CogmAIt"
    
    # 安全设置（限制在72字节以内）
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32)[:72])
    # 60 分钟 * 24 小时 * 8 天 = 8 天
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    
    # CORS 设置
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080", "*"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            # 如果是一个字符串，尝试将其解析为JSON数组
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # 如果无法解析为JSON，则尝试以逗号分隔
                if v.startswith("[") and v.endswith("]"):
                    # 可能是格式不正确的JSON，尝试手动处理
                    v = v.strip("[]").strip()
                    if v:
                        return [i.strip().strip('"\'') for i in v.split(",")]
                    return []
                else:
                    # 普通的逗号分隔字符串
                    return [i.strip() for i in v.split(",")]
        
        # 如果已经是列表，直接返回
        if isinstance(v, list):
            return v
        
        # 默认返回空列表
        return []

    # 数据库设置
    DB_HOST: str = "localhost"
    DB_PORT: str = "3306"
    DB_USER: str = "root"
    DB_PASSWORD: str = "xkkxkkxkk"
    DB_NAME: str = "cogmait"
    DB_DATABASE: str = "cogmait"
    
    # 默认使用MySQL数据库
    DATABASE_URI: Optional[str] = None
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        """
        获取数据库URI
        """
        if self.DATABASE_URI:
            return self.DATABASE_URI
        
        # 默认使用MySQL
        return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    # 是否自动创建数据库表结构
    CREATE_TABLES: bool = True
    
    # 模型供应商设置
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # PDF OCR服务配置
    OCR_SPACE_API_KEY: Optional[str] = None
    
    # Pinecone向量数据库配置
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = "us-east-1"
    
    # 模型路径设置
    PROVIDERS_PACKAGE: str = "app.providers"
    
    # MinIO配置
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "xkk"
    MINIO_SECRET_KEY: str = "xkkxkkxkk"
    MINIO_SECURE: bool = False
    
    class Config:
        case_sensitive = True
        env_file = ".env"


# 创建设置实例
settings = Settings() 
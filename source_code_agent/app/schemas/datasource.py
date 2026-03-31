from datetime import datetime
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field


class DataSourceBase(BaseModel):
    """数据源基础模型"""
    name: str = Field(..., description="数据源名称")
    type: str = Field(..., description="数据源类型，如mysql、postgresql等")
    host: str = Field(..., description="数据库主机地址")
    port: int = Field(..., description="数据库端口")
    database: str = Field(..., description="数据库名称")
    username: str = Field(..., description="数据库用户名")
    remark: Optional[str] = Field(None, description="备注信息")
    model_id: Optional[str] = Field(None, description="关联的AI模型ID")
    extra_params: Optional[Dict[str, Any]] = Field(None, description="额外参数")


class DataSourceCreate(DataSourceBase):
    """创建数据源模型"""
    password: str = Field(..., description="数据库密码")


class DataSourceUpdate(BaseModel):
    """更新数据源模型"""
    name: Optional[str] = Field(None, description="数据源名称")
    type: Optional[str] = Field(None, description="数据源类型，如mysql、postgresql等")
    host: Optional[str] = Field(None, description="数据库主机地址")
    port: Optional[int] = Field(None, description="数据库端口")
    database: Optional[str] = Field(None, description="数据库名称")
    username: Optional[str] = Field(None, description="数据库用户名")
    password: Optional[str] = Field(None, description="数据库密码")
    remark: Optional[str] = Field(None, description="备注信息")
    model_id: Optional[str] = Field(None, description="关联的AI模型ID")
    is_active: Optional[bool] = Field(None, description="是否激活")
    extra_params: Optional[Dict[str, Any]] = Field(None, description="额外参数")


class DataSourceResponse(DataSourceBase):
    """数据源响应模型"""
    id: str = Field(..., description="数据源ID")
    password: str = Field(..., description="数据库密码")
    is_active: bool = Field(..., description="是否激活")
    created_by: Optional[str] = Field(None, description="创建者ID")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    
    class Config:
        from_attributes = True


class TestConnectionResponse(BaseModel):
    """测试连接响应模型"""
    success: bool = Field(..., description="是否成功")
    message: Optional[str] = Field(None, description="错误信息")


class ExecuteQueryRequest(BaseModel):
    """执行查询请求模型"""
    datasource_id: str = Field(..., description="数据源ID")
    query: str = Field(..., description="查询语句")
    save_result: bool = Field(False, description="是否保存结果")
    file_name: Optional[str] = Field(None, description="保存的文件名，仅在save_result为True时有效")
    file_description: Optional[str] = Field(None, description="文件描述，仅在save_result为True时有效")


class ExecuteQueryResponse(BaseModel):
    """执行查询响应模型"""
    success: bool = Field(..., description="是否成功")
    data: List[Dict[str, Any]] = Field([], description="查询结果")
    columns: List[str] = Field([], description="列名列表")
    total: int = Field(0, description="总行数")
    execution_time: int = Field(0, description="执行时间(毫秒)")
    error: Optional[str] = Field(None, description="错误信息")
    file_id: Optional[str] = Field(None, description="保存的文件ID，仅在save_result为True时返回")


class GenerateSQLRequest(BaseModel):
    """生成SQL请求模型"""
    datasource_id: str = Field(..., description="数据源ID")
    query: str = Field(..., description="自然语言查询")


class GenerateSQLResponse(BaseModel):
    """生成SQL响应模型"""
    success: bool = Field(..., description="是否成功")
    sql: Optional[str] = Field(None, description="生成的SQL查询")
    error: Optional[str] = Field(None, description="错误信息")
    query_id: Optional[str] = Field(None, description="生成的SQL查询记录ID")


class DatabaseStructureResponse(BaseModel):
    """数据库结构响应模型"""
    success: bool = Field(..., description="是否成功")
    structure: Dict[str, Any] = Field({}, description="数据库结构")
    error: Optional[str] = Field(None, description="错误信息") 
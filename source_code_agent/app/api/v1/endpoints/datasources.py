import csv
import io
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Body, Query
from sqlalchemy.orm import Session

from app.utils import deps
from app.db.session import get_db
from app.models.datasource import DataSource, DataSourceQuery
from app.models.model import Model
from app.models.file import File
from app.schemas.datasource import (
    DataSourceCreate, 
    DataSourceUpdate, 
    DataSourceResponse,
    TestConnectionResponse,
    ExecuteQueryRequest,
    ExecuteQueryResponse,
    GenerateSQLRequest,
    GenerateSQLResponse,
    DatabaseStructureResponse
)
from app.utils.db_connectors.factory import DBConnectorFactory
from app.utils.ai_sql_assistant import AISQLAssistant
from app.utils.db_connectors.manager import DBConnectorManager
from app.utils import format_datetime


router = APIRouter()


@router.get("/", response_model=List[DataSourceResponse])
async def get_datasources(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    name: Optional[str] = None,
    type: Optional[str] = None,
    current_user = Depends(deps.get_current_active_user)
):
    """
    获取数据源列表
    
    可按名称和类型筛选
    """
    query = db.query(DataSource)
    
    # 应用筛选条件
    if name:
        query = query.filter(DataSource.name.like(f"%{name}%"))
    if type:
        query = query.filter(DataSource.type == type)
    
    # 获取总数
    total = query.count()
    
    # 应用分页
    datasources = query.offset(skip).limit(limit).all()
    
    # 将每个数据源中的datetime转换为字符串
    for ds in datasources:
        ds.created_at = format_datetime(ds.created_at)
        ds.updated_at = format_datetime(ds.updated_at)
    
    return datasources


@router.post("/", response_model=DataSourceResponse)
async def create_datasource(
    *,
    db: Session = Depends(get_db),
    datasource_in: DataSourceCreate,
    current_user = Depends(deps.get_current_active_user)
):
    """创建数据源"""
    # 检查数据源名称是否已存在
    if db.query(DataSource).filter(DataSource.name == datasource_in.name).first():
        raise HTTPException(status_code=400, detail="数据源名称已存在")
    
    # 创建数据源记录
    datasource = DataSource(
        name=datasource_in.name,
        type=datasource_in.type,
        host=datasource_in.host,
        port=datasource_in.port,
        database=datasource_in.database,
        username=datasource_in.username,
        password=datasource_in.password,
        remark=datasource_in.remark,
        model_id=datasource_in.model_id,
        extra_params=datasource_in.extra_params,
        created_by=current_user.id
    )
    
    db.add(datasource)
    db.commit()
    db.refresh(datasource)
    
    # 将datetime转换为字符串
    datasource.created_at = format_datetime(datasource.created_at)
    datasource.updated_at = format_datetime(datasource.updated_at)
    
    return datasource


@router.get("/{datasource_id}", response_model=DataSourceResponse)
async def get_datasource(
    *,
    db: Session = Depends(get_db),
    datasource_id: str,
    current_user = Depends(deps.get_current_active_user)
):
    """获取数据源详情"""
    datasource = db.query(DataSource).filter(DataSource.id == datasource_id).first()
    if not datasource:
        raise HTTPException(status_code=404, detail="数据源不存在")
    
    # 将datetime转换为字符串
    datasource.created_at = format_datetime(datasource.created_at)
    datasource.updated_at = format_datetime(datasource.updated_at)
    
    # # 确保密码字段可见
    # password = datasource.password
    # if password is None:
    #     password = ""
    
    # 创建包含所有必要字段的响应数据
    response_data = {
        "id": datasource.id,
        "name": datasource.name,
        "type": datasource.type,
        "host": datasource.host,
        "port": datasource.port,
        "database": datasource.database,
        "username": datasource.username,
        "password": datasource.password,  # 确保是字符串
        "extra_params": datasource.extra_params,
        "is_active": datasource.is_active,
        "remark": datasource.remark,
        "model_id": datasource.model_id,
        "created_by": datasource.created_by,
        "created_at": datasource.created_at,
        "updated_at": datasource.updated_at
    }
    print(response_data)
    return response_data


@router.put("/{datasource_id}", response_model=DataSourceResponse)
async def update_datasource(
    *,
    db: Session = Depends(get_db),
    datasource_id: str,
    datasource_in: DataSourceUpdate,
    current_user = Depends(deps.get_current_active_user)
):
    """更新数据源"""
    datasource = db.query(DataSource).filter(DataSource.id == datasource_id).first()
    if not datasource:
        raise HTTPException(status_code=404, detail="数据源不存在")
    
    # 检查更新的名称是否与其他数据源冲突
    if datasource_in.name and datasource_in.name != datasource.name:
        if db.query(DataSource).filter(DataSource.name == datasource_in.name).first():
            raise HTTPException(status_code=400, detail="数据源名称已存在")
    
    # 更新数据源信息
    update_data = datasource_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(datasource, field, value)
    
    db.commit()
    db.refresh(datasource)
    
    # 将datetime转换为字符串
    datasource.created_at = format_datetime(datasource.created_at)
    datasource.updated_at = format_datetime(datasource.updated_at)
    
    return datasource


@router.delete("/{datasource_id}", response_model=Dict[str, Any])
async def delete_datasource(
    *,
    db: Session = Depends(get_db),
    datasource_id: str,
    current_user = Depends(deps.get_current_active_user)
):
    """删除数据源"""
    datasource = db.query(DataSource).filter(DataSource.id == datasource_id).first()
    if not datasource:
        raise HTTPException(status_code=404, detail="数据源不存在")
    
    db.delete(datasource)
    db.commit()
    
    return {"success": True}


@router.post("/test_connection", response_model=TestConnectionResponse)
async def test_connection(
    *,
    db: Session = Depends(get_db),
    datasource_id: str = Body(..., embed=True),
    current_user = Depends(deps.get_current_active_user)
):
    """测试数据库连接"""
    try:
        # 使用连接器管理器获取连接器和数据源信息
        connector, error = await DBConnectorManager.get_connector_by_id(datasource_id, db)
        
        if error:
            return TestConnectionResponse(success=False, message=error)
        
        # 测试连接
        success, error_msg = await connector.test_connection()
        
        # 确保安全断开连接
        try:
            await connector.disconnect()
        except Exception as e:
            print(f"断开连接时发生错误(可以忽略): {str(e)}")
        
        if success:
            return TestConnectionResponse(success=True)
        else:
            return TestConnectionResponse(success=False, message=error_msg or "连接测试失败")
    
    except Exception as e:
        print(f"测试连接异常: {str(e)}")
        return TestConnectionResponse(success=False, message=str(e))


@router.get("/{datasource_id}/structure", response_model=DatabaseStructureResponse)
async def get_database_structure(
    *,
    db: Session = Depends(get_db),
    datasource_id: str,
    current_user = Depends(deps.get_current_active_user)
):
    """获取数据库结构"""
    try:
        # 使用连接器管理器获取连接器
        connector, error = await DBConnectorManager.get_connector_by_id(datasource_id, db)
        
        if error:
            return DatabaseStructureResponse(success=False, error=error)
        
        # 获取数据库结构
        db_structure = await connector.get_db_structure()
        
        # 安全断开连接
        try:
            await connector.disconnect()
        except Exception as e:
            print(f"断开连接时发生错误(可以忽略): {str(e)}")
        
        return DatabaseStructureResponse(success=True, structure=db_structure)
    
    except Exception as e:
        print(f"获取数据库结构异常: {str(e)}")
        return DatabaseStructureResponse(success=False, error=str(e))


@router.post("/execute_query", response_model=ExecuteQueryResponse)
async def execute_query(
    *,
    db: Session = Depends(get_db),
    query_data: ExecuteQueryRequest,
    current_user = Depends(deps.get_current_active_user)
):
    """执行SQL查询"""
    datasource = db.query(DataSource).filter(DataSource.id == query_data.datasource_id).first()
    if not datasource:
        raise HTTPException(status_code=404, detail="数据源不存在")
    
    try:
        # 使用连接器管理器获取连接器
        connector, error = await DBConnectorManager.get_connector_by_id(query_data.datasource_id, db)
        
        if error:
            return ExecuteQueryResponse(success=False, error=error)
        
        # 执行查询
        rows, columns, affected_rows, execution_time = await connector.execute_query(query_data.query)
        
        # 安全断开连接
        try:
            await connector.disconnect()
        except Exception as e:
            print(f"断开连接时发生错误(可以忽略): {str(e)}")
        
        # 创建查询记录
        query_record = DataSourceQuery(
            datasource_id=datasource.id,
            query_text=query_data.query,
            execution_time=execution_time,
            rows_affected=affected_rows,
            status="success",
            created_by=current_user.id
        )
        
        # 如果需要保存结果为文件
        file_id = None
        if query_data.save_result and rows and columns:
            # 创建CSV内容
            output = io.StringIO()
            csv_writer = csv.writer(output)
            
            # 写入表头
            csv_writer.writerow(columns)
            
            # 写入数据行
            for row in rows:
                csv_writer.writerow([row[col] for col in columns])
            
            csv_content = output.getvalue()
            
            # 创建文件记录
            file_name = query_data.file_name or f"query_result_{uuid.uuid4().hex[:8]}.csv"
            if not file_name.endswith(".csv"):
                file_name += ".csv"
            
            file_record = File(
                filename=file_name,  # 修正字段名为filename
                original_filename=file_name,
                file_type="csv",
                file_size=len(csv_content),
                description=query_data.file_description or f"查询结果: {query_data.query[:100]}...",
                text_content=csv_content,
                status="processed",
                created_by=current_user.username,  # 修正为username
                user_id=current_user.id  # 添加用户ID
            )
            
            db.add(file_record)
            db.commit()
            db.refresh(file_record)
            
            # 更新查询记录
            query_record.result_file_id = file_record.id
            file_id = file_record.id
        
        # 保存查询记录
        db.add(query_record)
        db.commit()
        
        return ExecuteQueryResponse(
            success=True,
            data=rows,
            columns=columns,
            total=len(rows),
            execution_time=execution_time,
            file_id=file_id
        )
    
    except Exception as e:
        # 记录失败的查询
        error_msg = str(e)
        query_record = DataSourceQuery(
            datasource_id=datasource.id,
            query_text=query_data.query,
            status="failed",
            error_message=error_msg[:500],
            created_by=current_user.id
        )
        db.add(query_record)
        db.commit()
        
        return ExecuteQueryResponse(success=False, error=error_msg)


@router.post("/generate_sql", response_model=GenerateSQLResponse)
async def generate_sql(
    *,
    db: Session = Depends(get_db),
    gen_data: GenerateSQLRequest,
    current_user = Depends(deps.get_current_active_user)
):
    """使用AI模型生成SQL查询语句"""
    # 获取数据源信息
    datasource = db.query(DataSource).filter(DataSource.id == gen_data.datasource_id).first()
    if not datasource:
        raise HTTPException(status_code=404, detail="数据源不存在")
    
    # 检查是否关联了AI模型
    if not datasource.model_id:
        return GenerateSQLResponse(success=False, error="数据源未关联AI模型")
    
    # 获取AI模型
    model = db.query(Model).filter(Model.id == datasource.model_id).first()
    if not model:
        return GenerateSQLResponse(success=False, error="关联的AI模型不存在")
    
    # 检查模型状态
    if not model.is_active:
        return GenerateSQLResponse(success=False, error="AI模型未激活")
    
    # 检查查询参数是否有效
    if not gen_data.query or gen_data.query.strip() == "":
        return GenerateSQLResponse(success=False, error="查询内容不能为空")
    
    try:
        # 使用连接器管理器获取连接器
        connector, error = await DBConnectorManager.get_connector_by_id(gen_data.datasource_id, db)
        
        if error:
            return GenerateSQLResponse(success=False, error=error)
        
        # 获取数据库结构
        db_structure = await connector.get_db_structure()
        
        # 安全断开连接
        try:
            await connector.disconnect()
        except Exception as e:
            print(f"断开连接时发生错误(可以忽略): {str(e)}")
        
        # 创建SQL助手
        sql_assistant = AISQLAssistant(model)
        
        # 生成SQL，确保传递一个有效的查询字符串
        sql = await sql_assistant.generate_sql(
            datasource.type,
            db_structure,
            gen_data.query.strip(),
            db
        )
        print("sql:::", sql)
        # 创建SQL记录
        # query_record = DataSourceQuery(
        #     datasource_id=datasource.id,
        #     query_text=sql,
        #     status="generated",
        #     created_by=current_user.id
        # )
        
        # db.add(query_record)
        # db.commit()
        # db.refresh(query_record)
        
        return GenerateSQLResponse(success=True, sql=sql)
    
    except Exception as e:
        print(f"SQL生成异常: {str(e)}")
        return GenerateSQLResponse(success=False, error=f"模型调用失败: {str(e)}")


@router.get("/{datasource_id}/queries", response_model=List[Dict[str, Any]])
async def get_datasource_queries(
    *,
    db: Session = Depends(get_db),
    datasource_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    status: Optional[str] = None,
    current_user = Depends(deps.get_current_active_user)
):
    """获取数据源的查询记录"""
    # 验证数据源是否存在
    datasource = db.query(DataSource).filter(DataSource.id == datasource_id).first()
    if not datasource:
        raise HTTPException(status_code=404, detail="数据源不存在")
    
    # 构建查询
    query = db.query(DataSourceQuery).filter(DataSourceQuery.datasource_id == datasource_id)
    
    # 应用状态筛选
    if status:
        query = query.filter(DataSourceQuery.status == status)
    
    # 应用排序 - 按创建时间倒序
    query = query.order_by(DataSourceQuery.created_at.desc())
    
    # 获取总数
    total = query.count()
    
    # 应用分页
    queries = query.offset(skip).limit(limit).all()
    
    # 转换为dict，添加用户名
    result = []
    for q in queries:
        query_dict = q.to_dict()
        
        # 添加用户信息
        if q.user:
            query_dict["created_by_name"] = q.user.username
        else:
            query_dict["created_by_name"] = "未知用户"
            
        # 添加文件信息
        if q.result_file:
            query_dict["file_name"] = q.result_file.name
        
        result.append(query_dict)
    
    return result


@router.delete("/queries/{query_id}", response_model=Dict[str, Any])
async def delete_datasource_query(
    *,
    db: Session = Depends(get_db),
    query_id: str,
    current_user = Depends(deps.get_current_active_user)
):
    """删除查询记录"""
    query = db.query(DataSourceQuery).filter(DataSourceQuery.id == query_id).first()
    if not query:
        raise HTTPException(status_code=404, detail="查询记录不存在")
    
    # 可以增加一些权限检查，例如只允许删除自己创建的记录
    # if query.created_by != current_user.id:
    #     raise HTTPException(status_code=403, detail="无权删除此记录")
    
    db.delete(query)
    db.commit()
    
    return {"success": True, "message": "查询记录已删除"} 
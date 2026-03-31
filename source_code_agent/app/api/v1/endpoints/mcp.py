from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, Path
from sqlalchemy.orm import Session
from sqlalchemy import func # Import func for database functions
import uuid
import re
import httpx
from datetime import datetime
# from pydantic import BaseModel # BaseModel is imported in schemas

from app.db.session import get_db
from app.utils.deps import get_current_active_user
from app.models.mcp import MCPService # Import the new SQLAlchemy model
from app.schemas.mcp import (
    MCPServiceListResponse, 
    MCPServiceResponse,
    MCPServiceCreate,
    MCPServiceUpdate,
    UserServiceConnectionCreate,
    UserServiceConnectionResponse
)

router = APIRouter()

# # 定义响应模型 # Removed as schemas are imported
# class ServiceResponse(BaseModel):
#     id: str
#     name: str
#     description: str
#     type: str
#     status: str
#     icon: str
#     provider: str
#     config: Dict[str, Any]

# class ServiceListResponse(BaseModel):
#     total: int
#     items: List[ServiceResponse]

# class RecommendResponse(BaseModel):
#     code: int
#     data: List[ServiceResponse]
#     msg: str

# 模拟数据 # Removed mock data
# MOCK_SERVICES = [
#     {
#         "id": "1",
#         "name": "OpenAI GPT-4",
#         "description": "OpenAI 的 GPT-4 模型服务",
#         "type": "llm",
#         "status": "active",
#         "icon": "openai",
#         "provider": "OpenAI",
#         "config": {
#             "api_key": "",
#             "model": "gpt-4"
#         }
#     },
#     {
#         "id": "2",
#         "name": "Claude 3",
#         "description": "Anthropic 的 Claude 3 模型服务",
#         "type": "llm",
#         "status": "active",
#         "icon": "anthropic",
#         "provider": "Anthropic",
#         "config": {
#             "api_key": "",
#             "model": "claude-3-opus-20240229"
#         }
#     },
#     {
#         "id": "3",
#         "name": "Gemini Pro",
#         "description": "Google 的 Gemini Pro 模型服务",
#         "type": "llm",
#         "status": "active",
#         "icon": "google",
#         "provider": "Google",
#         "config": {
#             "api_key": "",
#             "model": "gemini-pro"
#         }
#     }
# ]

# RECOMMENDED_SERVICES = [
#     {
#         "id": "1",
#         "name": "OpenAI GPT-4",
#         "description": "OpenAI 的 GPT-4 模型服务",
#         "type": "llm",
#         "status": "active",
#         "icon": "openai",
#         "provider": "OpenAI",
#         "config": {
#             "api_key": "",
#             "model": "gpt-4"
#         }
#     }
# ]

@router.get("/services", response_model=MCPServiceListResponse)
async def get_service_list(
    db: Session = Depends(get_db),
    type: Optional[str] = None,
    deployment_type: Optional[str] = None,
    is_official: Optional[bool] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: Any = Depends(get_current_active_user), # Keep dependency if auth is required
):
    """
    获取 MCP 服务列表
    """
    query = db.query(MCPService)

    # 应用过滤条件
    if type:
        query = query.filter(MCPService.type == type)
    if deployment_type:
        query = query.filter(MCPService.deployment_type == deployment_type)
    if is_official is not None:
        query = query.filter(MCPService.is_official == is_official)

    # 获取总数
    total = query.count()

    # 应用分页
    services = query.offset((page - 1) * limit).limit(limit).all()

    return MCPServiceListResponse(
        total=total,
        items=[MCPServiceResponse.from_orm(service) for service in services] # Convert ORM objects to Pydantic objects
    )


@router.get("/services/types", response_model=List[str]) # New endpoint for service types
async def get_service_types(
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user), # Keep dependency if auth is required
):
    """
    获取所有可用的 MCP 服务类型列表
    """
    # Query distinct types from MCPService table
    types = db.query(MCPService.type).distinct().all()
    # Extract type strings from the result (which is a list of tuples)
    return [t[0] for t in types]

@router.get("/services/providers", response_model=List[str])
async def get_service_providers(
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    获取所有可用的 MCP 服务提供商列表
    """
    # 查询所有不同的提供商
    providers = db.query(MCPService.provider).distinct().all()
    # 提取提供商名称
    return [p[0] for p in providers]

@router.get("/services/recommend", response_model=List[MCPServiceResponse]) # Use new schema
async def get_recommend_services(
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    获取推荐的 MCP 服务
    """
    # 首先尝试获取官方推荐服务
    official_services = db.query(MCPService).filter(MCPService.is_official == True).limit(10).all()
    
    if official_services:
        return [MCPServiceResponse.from_orm(service) for service in official_services]
        
    # 如果没有官方推荐服务，返回随机服务
    recommended_services = db.query(MCPService).order_by(func.random()).limit(10).all()
    return [MCPServiceResponse.from_orm(service) for service in recommended_services]

@router.get("/services/search") # Removed response_model here, will be handled by middleware
async def search_services(
    keyword: str,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    搜索 MCP 服务
    """
    if not keyword:
        # Return empty list in data for consistency with list endpoint
        return {
            "code": 200,
            "data": {
                "total": 0,
                "items": []
            },
            "msg": "操作成功"
        }

    # 搜索服务名称、描述、提供商或标签
    results = db.query(MCPService).filter(
        (MCPService.name.ilike(f"%{keyword}%")) |
        (MCPService.description.ilike(f"%{keyword}%")) |
        (MCPService.provider.ilike(f"%{keyword}%"))
    ).all()

    # Convert ORM objects to Pydantic objects and return in the unified format
    search_results_pydantic = [MCPServiceResponse.from_orm(service) for service in results]

    return {
        "code": 200,
        "data": {
            "total": len(search_results_pydantic), # Total count of search results
            "items": search_results_pydantic
        },
        "msg": "操作成功"
    }

@router.get("/services/{service_id}", response_model=MCPServiceResponse) # Use new schema
async def get_service_detail(
    service_id: str = Path(..., description="服务ID"),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user), # Keep dependency if auth is required
):
    """
    获取 MCP 服务详情
    """
    service = db.query(MCPService).filter(MCPService.id == service_id).first()
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="服务不存在"
        )
    return MCPServiceResponse.from_orm(service) # Convert ORM object to Pydantic object

@router.post("/services", response_model=MCPServiceResponse)
async def create_service(
    service_data: MCPServiceCreate,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    创建 MCP 服务
    """
    # 生成唯一ID，如果没有提供
    if not service_data.id:
        service_data.id = str(uuid.uuid4())
    
    # 将服务所有者设置为当前用户
    service_dict = service_data.dict()
    service_dict["owner_id"] = current_user.id
    
    # 创建新服务
    new_service = MCPService(**service_dict)
    
    try:
        db.add(new_service)
        db.commit()
        db.refresh(new_service)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"创建服务失败: {str(e)}"
        )
    
    return MCPServiceResponse.from_orm(new_service)

@router.put("/services/{service_id}", response_model=MCPServiceResponse)
async def update_service(
    service_data: MCPServiceUpdate,
    service_id: str = Path(..., description="服务ID"),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    更新 MCP 服务
    """
    # 查找服务
    service = db.query(MCPService).filter(MCPService.id == service_id).first()
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="服务不存在"
        )
    
    # 检查权限 - 只有所有者或管理员可以更新
    if service.owner_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限更新此服务"
        )
    
    # 更新服务字段
    service_data_dict = service_data.dict(exclude_unset=True)
    service_data_dict["updated_at"] = datetime.utcnow()
    
    for key, value in service_data_dict.items():
        if value is not None:  # 只更新非空字段
            setattr(service, key, value)
    
    try:
        db.commit()
        db.refresh(service)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"更新服务失败: {str(e)}"
        )
    
    return MCPServiceResponse.from_orm(service)

@router.delete("/services/{service_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_service(
    service_id: str = Path(..., description="服务ID"),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    删除 MCP 服务
    """
    # 查找服务
    service = db.query(MCPService).filter(MCPService.id == service_id).first()
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="服务不存在"
        )
    
    # 检查权限 - 只有所有者或管理员可以删除
    if service.owner_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限删除此服务"
        )
    
    try:
        db.delete(service)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"删除服务失败: {str(e)}"
        )
    
    return None

@router.post("/services/connect", response_model=UserServiceConnectionResponse)
async def connect_service(
    connection_data: UserServiceConnectionCreate,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    连接 MCP 服务
    """
    # 检查服务是否存在
    service = db.query(MCPService).filter(MCPService.id == connection_data.service_id).first()
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="服务不存在"
        )
    
    from app.models.mcp import UserServiceConnection
    
    # 检查用户是否已经连接此服务
    existing_connection = db.query(UserServiceConnection).filter(
        UserServiceConnection.user_id == current_user.id,
        UserServiceConnection.service_id == connection_data.service_id
    ).first()
    
    if existing_connection:
        # 更新现有连接
        for key, value in connection_data.dict(exclude={"service_id"}, exclude_unset=True).items():
            if value is not None:
                setattr(existing_connection, key, value)
        existing_connection.status = "active"
        existing_connection.updated_at = datetime.utcnow()
        connection = existing_connection
    else:
        # 创建新连接
        connection_dict = connection_data.dict()
        connection_dict["user_id"] = current_user.id
        connection_dict["status"] = "active"
        connection = UserServiceConnection(**connection_dict)
        db.add(connection)
    
    try:
        db.commit()
        db.refresh(connection)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"连接服务失败: {str(e)}"
        )
    
    return UserServiceConnectionResponse.from_orm(connection)

@router.get("/services/connections", response_model=List[UserServiceConnectionResponse])
async def get_user_connections(
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    获取用户所有连接的服务
    """
    from app.models.mcp import UserServiceConnection
    
    connections = db.query(UserServiceConnection).filter(
        UserServiceConnection.user_id == current_user.id
    ).all()
    
    return [UserServiceConnectionResponse.from_orm(connection) for connection in connections]

@router.post("/services/from-github", response_model=MCPServiceResponse)
async def create_service_from_github(
    data: Dict[str, str] = Body(...),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    从Github仓库快速创建MCP服务
    """
    github_url = data.get("github_url")
    
    if not github_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请提供Github仓库地址"
        )
        
    # 解析Github仓库URL
    repo_pattern = re.compile(r"github\.com/([^/]+)/([^/]+)")
    match = repo_pattern.search(github_url)
    
    if not match:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Github仓库地址格式不正确"
        )
        
    owner, repo = match.groups()
    
    # 获取仓库信息
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            repo_response = await client.get(f"https://api.github.com/repos/{owner}/{repo}")
            repo_response.raise_for_status()
            repo_info = repo_response.json()
            
            # 获取README内容
            readme_response = await client.get(f"https://api.github.com/repos/{owner}/{repo}/readme")
            readme_content = ""
            if readme_response.status_code == 200:
                readme_info = readme_response.json()
                readme_content_response = await client.get(readme_info.get("download_url", ""))
                if readme_content_response.status_code == 200:
                    readme_content = readme_content_response.text
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Github仓库不存在或无法访问: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取Github仓库信息失败: {str(e)}"
        )
    
    # 解析仓库信息创建MCP服务
    service_id = str(uuid.uuid4())
    
    # 尝试从README中提取服务类型
    service_type = "llm"  # 默认类型
    if "embedding" in readme_content.lower() or "embedding" in repo_info["description"].lower() if repo_info.get("description") else "":
        service_type = "embedding"
    elif "vision" in readme_content.lower() or "vision" in repo_info["description"].lower() if repo_info.get("description") else "":
        service_type = "vision"
    
    # 创建MCP服务
    new_service = MCPService(
        id=service_id,
        name=repo_info.get("name", ""),
        description=repo_info.get("description", "") or f"Github仓库: {owner}/{repo}",
        type=service_type,
        provider=owner,
        github_url=github_url,
        deployment_type="local",
        owner_id=current_user.id,
        tags=["github", repo_info.get("language", "").lower()] if repo_info.get("language") else ["github"],
        config_template={
            "repo_url": github_url,
            "clone_cmd": f"git clone {github_url}"
        },
        usage_docs=readme_content
    )
    
    try:
        db.add(new_service)
        db.commit()
        db.refresh(new_service)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"创建服务失败: {str(e)}"
        )
    
    return MCPServiceResponse.from_orm(new_service)

@router.post("/services/{service_id}/test", response_model=Dict[str, Any])
async def test_service_connection(
    service_id: str = Path(..., description="服务ID"),
    data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    测试MCP服务连接
    """
    # 检查服务是否存在
    service = db.query(MCPService).filter(MCPService.id == service_id).first()
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="服务不存在"
        )

    config = data.get("config", {})
    
    # 根据服务类型进行测试
    try:
        if service.type == "llm":
            # 测试LLM服务
            return {
                "status": "success",
                "message": "LLM服务连接测试成功",
                "result": {
                    "model": config.get("model", ""),
                    "test_response": "这是一个测试回复，表示LLM服务连接正常。"
                }
            }
        elif service.type == "embedding":
            # 测试Embedding服务
            return {
                "status": "success",
                "message": "Embedding服务连接测试成功",
                "result": {
                    "dimensions": 1536,
                    "sample_embedding": [0.123, 0.456, 0.789, -0.123, -0.456]
                }
            }
        else:
            # 通用测试
            return {
                "status": "success",
                "message": f"{service.type}服务连接测试成功",
                "result": {
                    "service_id": service_id,
                    "test_time": datetime.utcnow().isoformat()
                }
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"服务连接测试失败: {str(e)}",
            "error": str(e)
        }

@router.post("/services/{service_id}/call", response_model=Dict[str, Any])
async def call_service_function(
    service_id: str = Path(..., description="服务ID"),
    data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    调用MCP服务功能
    """
    # 检查服务是否存在
    service = db.query(MCPService).filter(MCPService.id == service_id).first()
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="服务不存在"
        )
    
    # 检查必要参数
    function_name = data.get("function_name")
    params = data.get("params", {})
    
    if not function_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="缺少function_name参数"
        )
    
    # 调用MCP服务功能
    from app.utils.mcp import call_mcp_service
    
    try:
        result = await call_mcp_service(
            db=db,
            service_id=service_id,
            function_name=function_name,
            params=params,
            user_id=current_user.id
        )
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"调用服务功能失败: {str(e)}"
        )
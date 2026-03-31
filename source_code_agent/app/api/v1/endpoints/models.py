from typing import Any, List, Optional, Dict
import os

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.model import Model
from app.providers.manager import provider_manager
from app.schemas.model import (
    ModelCreate, 
    ModelUpdate, 
    ModelResponse, 
    ModelListResponse,
    ModelTestResponse,
    ProviderInfo
)
from app.utils.model import (
    create_model,
    get_model,
    get_models,
    update_model,
    delete_model,
    test_model_connection
)
from app.core.config import settings
from app.utils.deps import get_current_active_user

router = APIRouter()


@router.get("/", response_model=ModelListResponse)
async def read_models(
    db: Session = Depends(get_db),
    name: Optional[str] = None,
    provider: Optional[str] = None,
    type: Optional[str] = None,
    status: Optional[str] = None,
    vision_support: Optional[bool] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
) -> Any:
    """
    获取模型列表
    """
    skip = (page - 1) * limit
    
    # 构建查询
    query = db.query(Model)
    
    # 应用过滤条件
    if name:
        query = query.filter(Model.name.ilike(f"%{name}%"))
    if provider:
        query = query.filter(Model.provider == provider)
    if type:
        query = query.filter(Model.type == type)
    if status:
        query = query.filter(Model.status == status)
    if vision_support is not None:
        query = query.filter(Model.vision_support == vision_support)
    
    # 获取总数
    total = query.count()
    
    # 应用分页
    models = query.offset(skip).limit(limit).all()
    
    return {
        "total": total,
        "items": [model.to_dict() for model in models]
    }


@router.post("/", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_new_model(
    *,
    db: Session = Depends(get_db),
    model_in: ModelCreate,
    current_user: Any = Depends(get_current_active_user),
) -> Any:
    """
    创建新模型
    """
    try:
        # 检查提供商是否存在
        provider = provider_manager.get_provider(model_in.provider)
        
        # 传递当前用户ID
        model = create_model(db=db, model_in=model_in, user_id=current_user.id)
        return model.to_dict()
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"创建模型失败: {str(e)}"
        )


@router.get("/providers", response_model=List[ProviderInfo])
async def get_provider_list() -> Any:
    """
    获取所有可用的模型提供商
    """
    return provider_manager.get_all_providers()


@router.post("/providers/reload", status_code=status.HTTP_200_OK)
async def reload_providers() -> Any:
    """
    手动重新加载所有模型提供商
    
    此端点会强制重新扫描和加载提供商目录中的所有模块
    """
    try:
        provider_manager.reload_providers()
        
        return {
            "status": "success",
            "message": "提供商重新加载成功",
            "providers": provider_manager.get_all_providers()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重新加载提供商失败: {str(e)}"
        )


@router.get("/providers/modules", status_code=status.HTTP_200_OK)
async def get_provider_modules() -> Dict[str, Any]:
    """
    获取已加载的提供商模块详情
    
    返回所有已加载的模块和每个模块中的提供商类信息
    """
    try:
        module_details = provider_manager.get_module_details()
        return {
            "status": "success",
            **module_details
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取提供商模块详情失败: {str(e)}"
        )


@router.get("/providers/scan", status_code=status.HTTP_200_OK)
async def scan_provider_directory() -> Dict[str, Any]:
    """
    扫描提供商目录并返回文件系统状态
    
    用于调试文件系统监控功能
    """
    try:
        # 获取提供商目录路径
        import importlib
        provider_package = settings.PROVIDERS_PACKAGE
        package = importlib.import_module(provider_package)
        package_dir = os.path.dirname(package.__file__)
        
        # 列出目录中的所有文件
        provider_files = []
        for filename in os.listdir(package_dir):
            if filename.endswith('.py'):
                file_path = os.path.join(package_dir, filename)
                module_name = os.path.splitext(filename)[0]
                provider_files.append({
                    "filename": filename,
                    "module_name": module_name,
                    "path": file_path,
                    "size": os.path.getsize(file_path),
                    "last_modified": os.path.getmtime(file_path),
                    "is_loaded": module_name in provider_manager._loaded_modules
                })
        
        return {
            "status": "success",
            "provider_directory": package_dir,
            "files": provider_files,
            "loaded_modules": list(provider_manager._loaded_modules)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"扫描提供商目录失败: {str(e)}"
        )


@router.get("/{model_id}", response_model=ModelResponse)
async def read_model(
    *,
    db: Session = Depends(get_db),
    model_id: str,
) -> Any:
    """
    通过ID获取模型详情
    """
    model = get_model(db=db, model_id=model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="模型未找到"
        )
    return model.to_dict()


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model_api(
    *,
    db: Session = Depends(get_db),
    model_id: str,
    model_in: ModelUpdate,
) -> Any:
    """
    更新模型
    """
    model = get_model(db=db, model_id=model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="模型未找到"
        )
    
    # 如果更新了提供商，检查新提供商是否存在
    if model_in.provider and model_in.provider != model.provider:
        provider_manager.get_provider(model_in.provider)
        
    model = update_model(db=db, db_obj=model, obj_in=model_in)
    return model.to_dict()


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_api(
    *,
    db: Session = Depends(get_db),
    model_id: str,
) -> None:
    """
    删除模型
    """
    model = get_model(db=db, model_id=model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="模型未找到"
        )
    delete_model(db=db, model_id=model_id)


@router.post("/{model_id}/test", response_model=ModelTestResponse)
async def test_model_api(
    *,
    db: Session = Depends(get_db),
    model_id: str,
) -> Any:
    """
    测试模型连接
    """
    model = get_model(db=db, model_id=model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="模型未找到"
        )
    
    result = await test_model_connection(model)
    return result 
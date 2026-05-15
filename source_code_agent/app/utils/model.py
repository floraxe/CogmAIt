from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import json

from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.models.model import Model
from app.providers.manager import provider_manager
from app.schemas.model import ModelCreate, ModelUpdate
from app.utils.provider_icon_mapper import extract_icon_from_url, get_icon_filename


def get_model(db: Session, model_id: str) -> Optional[Model]:
    """
    通过ID获取模型
    
    参数:
        db (Session): 数据库会话
        model_id (str): 模型ID
    
    返回:
        Optional[Model]: 模型对象或None
    """
    return db.query(Model).filter(Model.id == model_id).first()


def get_models(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    name: Optional[str] = None,
    provider: Optional[str] = None,
    type: Optional[str] = None,
    status: Optional[str] = None,
    vision_support: Optional[bool] = None
) -> List[Model]:
    """
    获取模型列表，支持过滤
    
    参数:
        db (Session): 数据库会话
        skip (int): 跳过的记录数
        limit (int): 限制返回的记录数
        name (Optional[str]): 按名称过滤
        provider (Optional[str]): 按提供商过滤
        type (Optional[str]): 按类型过滤
        status (Optional[str]): 按状态过滤
        vision_support (Optional[bool]): 按是否支持图像识别过滤
    
    返回:
        List[Model]: 模型列表
    """
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
    
    # 应用分页
    return query.offset(skip).limit(limit).all()


def create_model(db: Session, model_in: ModelCreate, user_id: str = None) -> Model:
    """
    创建新模型
    
    参数:
        db (Session): 数据库会话
        model_in (ModelCreate): 模型创建模式
        user_id (str): 创建者用户ID
    
    返回:
        Model: 创建的模型
    """
    # 检查提供商是否存在并获取图标
    icon = None
    provider = provider_manager.get_provider(model_in.provider)
    icon = provider.icon
        
    
    db_model = Model(
        name=model_in.name,
        provider=model_in.provider,
        type=model_in.type,
        api_key=model_in.api_key,
        base_url=model_in.base_url,
        description=model_in.description,
        config=model_in.config,
        icon=icon,  # 使用从provider获取的图标，而不是用户提供的
        tool_call_support=model_in.tool_call_support,
        function_call_support=model_in.function_call_support,
        vision_support=model_in.vision_support,
        thinking_support=model_in.thinking_support,
        default_prompt=model_in.default_prompt,
        max_context_length=model_in.max_context_length,
        extra_body_params=model_in.extra_body_params,
        user_id=user_id  # 添加用户ID
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model


def update_model(
    db: Session, 
    db_obj: Model,
    obj_in: Union[ModelUpdate, Dict[str, Any]]
) -> Model:
    """
    更新模型
    
    参数:
        db (Session): 数据库会话
        db_obj (Model): 要更新的模型对象
        obj_in (Union[ModelUpdate, Dict[str, Any]]): 更新数据
    
    返回:
        Model: 更新后的模型
    """
    update_data = obj_in.dict(exclude_unset=True) if isinstance(obj_in, ModelUpdate) else obj_in
    
    # 添加调试日志
    print(f"更新模型 ID: {db_obj.id}, 名称: {db_obj.name}")
    print(f"更新数据: {update_data}")
    
    # 特别记录mcp_support字段的值
    if 'mcp_support' in update_data:
        print(f"MCP支持字段值: {update_data['mcp_support']}, 类型: {type(update_data['mcp_support'])}")
    else:
        print("更新数据中不包含mcp_support字段")
    
    # 更新模型属性
    for field, value in update_data.items():
        if hasattr(db_obj, field):
            print(f"更新字段 {field}: {getattr(db_obj, field)} -> {value}")
            setattr(db_obj, field, value)
    
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    
    # 验证更新后的值
    print(f"更新后的模型 mcp_support: {db_obj.mcp_support}")
    
    return db_obj


def delete_model(db: Session, model_id: str) -> None:
    """
    删除模型
    
    参数:
        db (Session): 数据库会话
        model_id (str): 模型ID
    """
    model = get_model(db=db, model_id=model_id)
    if model:
        db.delete(model)
        db.commit()


async def test_model_connection(model: Model) -> Dict[str, Any]:
    """
    测试模型连接
    
    参数:
        model (Model): 模型对象
    
    返回:
        Dict[str, Any]: 测试结果
    """
    try:
        # 获取对应的提供商
        provider = provider_manager.get_provider(model.provider)
        
        # 执行连接测试
        result = await provider.test_connection(
            api_key=model.api_key,
            base_url=model.base_url
        )
        
        return result
    except Exception as e:
        return {
            "status": "failed",
            "message": f"连接测试失败: {str(e)}"
        }


async def execute_model_inference(
    db: Session,
    model_id: str,
    payload: Dict[str, Any],
) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
    """
    执行模型推理（兼容入口，委托 ModelInferenceFacade）。
    """
    from app.services.inference.facade import model_inference_facade

    return await model_inference_facade.run(db, model_id, payload)
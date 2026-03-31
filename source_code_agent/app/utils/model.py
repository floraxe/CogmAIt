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
    payload: Dict[str, Any]
) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
    """
    执行模型推理
    
    参数:
        db (Session): 数据库会话
        model_id (str): 模型ID
        payload (Dict[str, Any]): 请求负载
    
    返回:
        Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]: 模型推理结果或流式生成器
    """
    try:
        # 获取模型
        model = get_model(db, model_id)
        if not model:
            return {"error": f"找不到模型: {model_id}"}
        
        if model.status != "active":
            return {"error": f"模型状态不是活动状态: {model.status}"}
        
        # 获取提供商
        provider = provider_manager.get_provider(model.provider)
        if not provider:
            return {"error": f"找不到提供商: {model.provider}"}
        
        # 处理不同类型的模型请求
        model_type = payload.get("model_type", model.type)
        
        # 确保payload的完整性 - 改用更灵活的检查方式
        # 原来严格要求某些参数必须存在，现在改为提供默认值
        if model_type == "chat":
            # 如果缺少messages参数，则返回错误
            if "messages" not in payload:
                return {"error": "请求中缺少必要参数: messages"}
            
            # 对于其他参数，我们可以提供默认值而不是直接返回错误
            if "temperature" not in payload:
                payload["temperature"] = 0.7
                
            if "max_tokens" not in payload:
                payload["max_tokens"] = 1000
        
        # 处理对话请求
        if model_type == "chat":
            try:
                print(payload,model.name,model.base_url)
                # 检查是否需要流式响应
                stream = payload.get("stream", False)
                
                # 调用提供商的聊天API
                response = await provider.chat_completion(
                    api_key=model.api_key,
                    base_url=model.base_url,
                    model=model.name,
                    messages=payload.get("messages", []),
                    temperature=payload.get("temperature", 0.7),
                    max_tokens=payload.get("max_tokens", 1000),
                    stream=stream,
                    **{k: v for k, v in payload.items() if k not in ["messages", "temperature", "max_tokens", "model_type", "stream"]}
                )
                
                # 如果启用了流式响应，确保直接返回异步生成器
                if stream:
                    # 记录返回类型以便调试
                    print(f"流式响应类型: {type(response)}")
                    return response  # 直接返回异步生成器
                
                return response
            except Exception as e:
                print(f"执行聊天模型推理时出错: {str(e)},{payload}")
                import traceback
                traceback.print_exc()
                return {"error": f"执行模型推理时出错: {str(e)}"}
            
        elif model_type == "embedding":
            # 嵌入请求
            texts = payload.get("input", [])
            if not texts:
                return {"error": "请求中缺少文本输入"}
            
            # 调用提供商的嵌入API
            embeddings_result = await provider.embedding(
                api_key=model.api_key,
                base_url=model.base_url,
                model=model.name,
                text=texts
            )
            
            # 处理返回结果，支持多种格式
            embeddings = []
            if isinstance(embeddings_result, dict):
                # 处理标准格式的返回结果
                if "embeddings" in embeddings_result:
                    # 如果已经解析好了直接使用
                    embeddings = embeddings_result.get("embeddings", [])
                elif "data" in embeddings_result:
                    # 处理OpenAI风格的API返回
                    data = embeddings_result.get("data", [])
                    if data and isinstance(data, list):
                        embeddings = [item.get("embedding", []) for item in data if "embedding" in item]
            elif isinstance(embeddings_result, list):
                # 如果直接返回了向量列表
                embeddings = embeddings_result
            
            # 检查是否成功获取到向量
            if not embeddings and isinstance(embeddings_result, dict):
                # 记录详细返回结果以便调试
                print(f"获取到的embedding结果: {json.dumps(embeddings_result, ensure_ascii=False)[:500]}...")
                
                # 尝试从返回结果中提取embedding
                if "embeddings" in embeddings_result:
                    embeddings = embeddings_result["embeddings"]
            
            # 最终检查
            if not embeddings:
                return {
                    "error": "无法从API返回中提取embedding数据", 
                    "raw_response": str(embeddings_result)[:1000]  # 包含部分原始响应以辅助调试
                }
            
            return {
                "success": True,
                "embeddings": embeddings,
                "model": model.name,
                "provider": model.provider,
                "dimensions": len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0
            }
            
        elif model_type == "completion":
            # 补全请求
            prompt = payload.get("prompt", "")
            if not prompt:
                return {"error": "请求中缺少提示输入"}
            
            # 提取其他参数
            temperature = payload.get("temperature", 0.7)
            max_tokens = payload.get("max_tokens", 1000)
            
            # 调用提供商的补全API
            response = await provider.text_completion(
                api_key=model.api_key,
                base_url=model.base_url,
                model_name=model.name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response
            
        else:
            return {"error": f"不支持的模型类型: {model_type}"}
            
    except Exception as e:
        print(f"执行模型推理时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"执行模型推理时出错: {str(e)}"} 
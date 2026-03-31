import os
from typing import Any, List, Optional, Dict
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.config import settings
from app.db.session import get_db
from app.models.knowledge import Knowledge
from app.models.knowledge import KnowledgeFile
import app.models as models  # 导入所有模型
from app.schemas.knowledge import (
    KnowledgeCreate, 
    KnowledgeUpdate, 
    KnowledgeResponse, 
    KnowledgeListResponse,
    KnowledgeFileResponse,
    FileProcessRequest,
    ChunkingConfig,
    KnowledgeFileListResponse,
    ChunkingConfig,
    ChunkingMethod
)
from app.utils.knowledge import (
    get_knowledge,
    get_knowledge_list,
    create_knowledge,
    update_knowledge,
    delete_knowledge,
    get_knowledge_file,
    get_knowledge_files,
    upload_file_to_knowledge,
    delete_knowledge_file,
    process_knowledge_file
)
from app.utils.deps import get_current_active_user, get_knowledge_admin
from app.utils.model import get_model, get_models, execute_model_inference
from app.utils.embedding import EmbeddingManager, HAS_PYMILVUS
from app.utils.file_processor import extract_text_from_file

import uuid
import shutil
import asyncio
from pathlib import Path

router = APIRouter()


@router.get("/chunking-methods")
async def get_chunking_methods(
    current_user: Any = Depends(get_current_active_user),
):
    """
    获取所有可用的分段方法
    """
    methods = []
    for method in ChunkingMethod:
        method_info = {
            "value": method.value,
            "name": method.name,
            "description": ""
        }
        
        # 添加描述信息
        if method == ChunkingMethod.CHARACTER:
            method_info["description"] = "按字符数量分段，适合处理纯文本"
        elif method == ChunkingMethod.TOKEN:
            method_info["description"] = "按Token分段，适合大多数语言模型"
        elif method == ChunkingMethod.SENTENCE:
            method_info["description"] = "按句子分段，保持句子的完整性"
        elif method == ChunkingMethod.PARAGRAPH:
            method_info["description"] = "按段落分段，保持段落的语义完整性"
        elif method == ChunkingMethod.SEMANTIC:
            method_info["description"] = "按语义分段，使用embedding模型计算语义边界"
        elif method == ChunkingMethod.RECURSIVE:
            method_info["description"] = "递归分段，综合多种分隔符，适合复杂文档"
        
        methods.append(method_info)
    
    return methods


@router.get("/embedding-models", response_model=List[Dict[str, Any]])
async def get_embedding_models(
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    获取所有可用的嵌入模型列表
    """
    # 过滤出类型为embedding且状态为active的模型
    embedding_models = get_models(db, type="embedding", status="active")
    
    # 转换为JSON格式
    result = []
    for model in embedding_models:
        result.append({
            "id": model.id,
            "name": model.name,
            "provider": model.provider,
            "description": model.description,
            "icon": model.icon
        })
    
    return result


@router.get("/", response_model=KnowledgeListResponse)
async def read_knowledge_list(
    db: Session = Depends(get_db),
    name: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: Any = Depends(get_current_active_user),
):
    """
    获取知识库列表
    """
    skip = (page - 1) * limit
    knowledge_list = get_knowledge_list(
        db, 
        skip=skip, 
        limit=limit, 
        name=name,
        status=status
    )
    total = db.query(Knowledge).count()
    
    return {
        "total": total,
        "items": [k.to_dict() for k in knowledge_list]
    }


@router.post("/", response_model=KnowledgeResponse, status_code=status.HTTP_201_CREATED)
async def create_new_knowledge(
    knowledge_in: KnowledgeCreate,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_knowledge_admin),
):
    """
    创建新知识库，支持配置分段规则
    """
    # 验证embedding_model是否存在
    if knowledge_in.embedding_model:
        model = get_model(db, knowledge_in.embedding_model)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="指定的向量化模型不存在"
            )
        if model.type != "embedding":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="指定的模型不是向量化模型"
            )
    
    # 处理分段配置
    if knowledge_in.chunking_config:
        # 确保config字段存在
        if not knowledge_in.config:
            knowledge_in.config = {}
        
        # 将分段配置转换为字典存储在config中
        knowledge_in.config["chunking_config"] = {
            "method": knowledge_in.chunking_config.method,
            "chunk_size": knowledge_in.chunking_config.chunk_size,
            "chunk_overlap": knowledge_in.chunking_config.chunk_overlap,
            "separator": knowledge_in.chunking_config.separator
        }
    
    # 传递当前用户ID
    knowledge = create_knowledge(db=db, knowledge_in=knowledge_in, user_id=current_user.id)
    return knowledge.to_dict()


@router.get("/{knowledge_id}", response_model=KnowledgeResponse)
async def read_knowledge_detail(
    knowledge_id: str,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    获取知识库详情
    """
    knowledge = get_knowledge(db=db, knowledge_id=knowledge_id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )
    return knowledge.to_dict()


@router.put("/{knowledge_id}", response_model=KnowledgeResponse)
async def update_knowledge_api(
    knowledge_id: str,
    knowledge_in: KnowledgeUpdate,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_knowledge_admin),
):
    """
    更新知识库，支持修改分段规则
    """
    knowledge = get_knowledge(db=db, knowledge_id=knowledge_id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )
    
    # 验证embedding_model是否存在
    if knowledge_in.embedding_model:
        model = get_model(db, knowledge_in.embedding_model)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="指定的向量化模型不存在"
            )
        if model.type != "embedding":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="指定的模型不是向量化模型"
            )
    
    # 处理分段配置
    if knowledge_in.chunking_config:
        # 确保config字段存在
        if not knowledge_in.config:
            knowledge_in.config = {} if not knowledge.config else knowledge.config
        
        # 将分段配置转换为字典存储在config中
        knowledge_in.config["chunking_config"] = {
            "method": knowledge_in.chunking_config.method,
            "chunk_size": knowledge_in.chunking_config.chunk_size,
            "chunk_overlap": knowledge_in.chunking_config.chunk_overlap,
            "separator": knowledge_in.chunking_config.separator
        }
    
    updated_knowledge = update_knowledge(db=db, knowledge=knowledge, knowledge_in=knowledge_in)
    return updated_knowledge.to_dict()


@router.delete("/{knowledge_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_api(
    knowledge_id: str,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_knowledge_admin),
):
    """
    删除知识库
    """
    knowledge = get_knowledge(db=db, knowledge_id=knowledge_id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )
    
    delete_knowledge(db=db, knowledge_id=knowledge_id)


@router.get("/{knowledge_id}/files", response_model=KnowledgeFileListResponse)
async def read_knowledge_files(
    knowledge_id: str,
    db: Session = Depends(get_db),
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: Any = Depends(get_current_active_user),
):
    """
    获取知识库文件列表
    """
    # 检查知识库是否存在
    knowledge = get_knowledge(db=db, knowledge_id=knowledge_id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )
    
    skip = (page - 1) * limit
    
    # 构建查询
    query = db.query(models.KnowledgeFile).filter(
        models.KnowledgeFile.knowledge_id == knowledge_id
    )
    
    # 应用状态过滤
    if status:
        query = query.filter(models.KnowledgeFile.status == status)
    
    # 获取总数
    total = query.count()
    
    # 获取分页数据，按创建时间倒序排序
    files = query.order_by(models.KnowledgeFile.created_at.desc()).offset(skip).limit(limit).all()
    
    # 转换为响应格式
    items = []
    for file in files:
        item = {
            "id": file.id,
            "knowledge_id": file.knowledge_id,
            "file_id": file.file_id,
            "filename": file.filename,
            "original_filename": file.original_filename,
            "file_type": file.file_type,
            "file_size": file.file_size,
            "status": file.status,
            "embedding_status": file.embedding_status,
            "chunk_count": file.chunk_count or 0,
            "vector_count": file.vector_count or 0,
            "error": file.error,
            "created_at": file.created_at.isoformat() if file.created_at else None,
            "updated_at": file.updated_at.isoformat() if file.updated_at else None,
            "chunking_time": file.chunking_time.isoformat() if file.chunking_time else None,
            "extra_data": file.extra_data or {}
        }
        items.append(item)
    print("返回知识库的文件列表：：",items)
    return {
        "total": total,
        "items": items
    }


@router.post("/{knowledge_id}/upload")
async def upload_file(
    knowledge_id: str,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_knowledge_admin),
):
    """
    上传单个或多个文件到知识库，支持异步处理
    
    参数:
        knowledge_id: 知识库ID
        files: 上传的文件列表
        description: 文件描述(可选)
    """
    try:
        # 检查知识库是否存在
        knowledge = get_knowledge(db=db, knowledge_id=knowledge_id)
        if not knowledge:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="知识库不存在"
            )
        
        uploaded_files = []
        for file in files:
            # 上传文件到知识库，传递当前用户ID
            uploaded_file = await upload_file_to_knowledge(
                    db=db, 
                    knowledge_id=knowledge_id, 
                    file=file,
                    background_tasks=background_tasks,
                description=description,
                user_id=current_user.id
                )
            uploaded_files.append(uploaded_file.to_dict())
        
        return {
            "message": f"成功上传 {len(uploaded_files)} 个文件",
                "files": uploaded_files
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上传文件失败: {str(e)}"
        )


@router.delete("/{knowledge_id}/files/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    knowledge_id: str,
    file_id: str,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_knowledge_admin),
):
    """
    删除知识库文件关联
    """
    try:
    # 检查知识库是否存在
        knowledge = get_knowledge(db=db, knowledge_id=knowledge_id)
        if not knowledge:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="知识库不存在"
            )
    
        # 检查知识库文件是否存在
        knowledge_file = db.query(models.KnowledgeFile).filter(
            models.KnowledgeFile.knowledge_id == knowledge_id,
            models.KnowledgeFile.id == file_id
        ).first()
        
        if not knowledge_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                    detail="文件关联不存在"
            )
    
        # 获取原始文件信息
        file = db.query(models.File).filter(models.File.id == knowledge_file.file_id).first()
        
        # 从向量存储中删除向量
        try:
            vector_store = EmbeddingManager.get_vector_store()
            if vector_store:
                vector_store.delete_vectors(knowledge_id=knowledge_id, file_id=knowledge_file.file_id)
        except Exception as e:
            print(f"删除向量失败: {str(e)}")
            # 继续执行，不中断删除流程

        # 删除知识库文件记录
        db.delete(knowledge_file)

        # 更新知识库文件计数和大小
        knowledge.file_count = db.query(models.KnowledgeFile).filter(
            models.KnowledgeFile.knowledge_id == knowledge_id
        ).count()
        knowledge.total_size = db.query(func.sum(models.KnowledgeFile.file_size)).filter(
            models.KnowledgeFile.knowledge_id == knowledge_id
        ).scalar() or 0

        # 提交更改
        db.commit()

    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除文件失败: {str(e)}"
        )


@router.post("/{knowledge_id}/files/{file_id}/reprocess")
async def reprocess_file(
    knowledge_id: str,
    file_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_knowledge_admin),
):
    """
    重新处理知识库文件，进行分段和向量化
    """
    # 检查知识库是否存在
    knowledge = get_knowledge(db=db, knowledge_id=knowledge_id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )
    print(f"重新处理知识库文件: {knowledge_id}, {file_id}")
    # 检查文件是否存在
    file = get_knowledge_file(db=db, file_id=file_id)
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在"
        )
    
    # 检查文件是否属于指定知识库
    if file.knowledge_id != knowledge_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文件不属于该知识库"
        )
    
    # # 获取文件路径
    # if file.file_type:
    #     file_extension = file.file_type
    #     file_name_with_ext = f"{file.filename}.{file_extension}"
    # else:
    #     file_name_with_ext = file.filename
    
    # file_path = os.path.join(
    #     os.getcwd(), 
    #     "uploads", 
    #     "raw", 
    #     knowledge_id,
    #     file_name_with_ext
    # )
    
    # # 检查文件是否存在
    # if not os.path.exists(file_path):
    #     raise HTTPException(
    #         status_code=status.HTTP_404_NOT_FOUND,
    #         detail="原始文件不存在"
    #     )
    
    # 重置文件状态
    file.status = "processing"
    file.embedding_status = "pending"
    file.error = None
    db.commit()
    
    # 添加后台任务重新处理文件
    background_tasks.add_task(
        process_knowledge_file,
        file_id,
        file.file_type,
        knowledge_id,
        file.filename
    )
    
    return {
        "message": "文件重新处理已开始"
    }
        
   


@router.post("/{knowledge_id}/retrieve", response_model=Dict[str, Any])
async def test_knowledge_retrieval(
    knowledge_id: str,
    params: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    测试知识库检索功能
    
    参数:
        - knowledge_id: 知识库ID
        - params: 检索参数，包含:
            - query: 查询文本
            - similarity_threshold: 相似度阈值，默认0.7
            - top_k: 返回结果数量，默认5
    """
    try:
        # 检查知识库是否存在
        knowledge = get_knowledge(db=db, knowledge_id=knowledge_id)
        if not knowledge:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="知识库不存在"
            )
        
        # 获取查询参数
        query = params.get("query", "")
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="查询文本不能为空"
            )
        
        # 解析前端参数，确保支持不同的参数名格式
        similarity_threshold = params.get("similarity_threshold", 0.7)
        # 兼容前端可能使用的不同参数名
        if "similarityThreshold" in params:
            similarity_threshold = params.get("similarityThreshold")
        
        top_k = params.get("top_k", 5)
        # 兼容前端可能使用的不同参数名
        if "topK" in params:
            top_k = params.get("topK")
        
        # 将参数转换为正确的类型
        try:
            similarity_threshold = float(similarity_threshold)
            top_k = int(top_k)
        except (ValueError, TypeError):
            # 如果转换失败，使用默认值
            similarity_threshold = 0.7
            top_k = 5
        
        # 获取知识库的嵌入模型
        model_id = knowledge.embedding_model
        if not model_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="知识库未配置嵌入模型"
            )
        
        # 检查是否有足够的文件进行检索
        file_count_query = db.query(func.count(KnowledgeFile.id)).filter(
            KnowledgeFile.knowledge_id == knowledge_id,
            KnowledgeFile.status == "indexed"  # 只考虑已索引的文件
        )
        file_count = file_count_query.scalar() or 0
        
        if file_count == 0:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "message": "知识库中没有已索引的文件，无法进行检索"
            }
        
        # 获取向量存储
        vector_store = EmbeddingManager.get_vector_store()
        if not vector_store or vector_store.client is None:
            # 如果向量存储不可用，返回友好的错误信息
            return {
                "query": query,
                "results": [],
                "total": 0,
                "message": "向量存储服务不可用，请检查 pymilvus 安装和配置"
            }
        
        # 使用模型将查询文本转换为向量
        query_embedding_result = await execute_model_inference(
            db,
            model_id,
            {
                "input": [query],
                "model_type": "embedding"
            }
        )
        
        if "error" in query_embedding_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"生成查询向量失败: {query_embedding_result['error']}"
            )
        
        # 从结果中提取查询向量
        query_embedding = query_embedding_result.get("embeddings", [])[0]
        
        # 在向量数据库中检索相似文档
        results = vector_store.search_similar(
            knowledge_id=knowledge_id,
            query_vector=query_embedding,
            limit=top_k,
            filter_expr=None  # 可以添加过滤条件
        )
        print(results)
        # 处理结果，补充文件信息
        processed_results = []
        for hit in results:
            # 获取对应的文件信息
            file_id = hit.get("file_id", "")
            file_info = get_knowledge_file(db, file_id)
            file_name = file_info.original_filename if file_info else "未知文件"
            
            # 计算得分，确保分数在0到1之间
            score = hit.get("score", 0)
            if score < 0:
                score = 0
            elif score > 1:
                score = 1
            
            # 只添加符合相似度阈值的结果
            if score >= similarity_threshold:
                processed_results.append({
                    "content": hit.get("text", ""),
                    "score": score,
                    "source_file": file_name,
                    "file_id": file_id,
                    "chunk_id": hit.get("chunk_index", 0)
                })
        
        return {
            "query": query,
            "results": processed_results,
            "total": len(processed_results)
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"知识库检索失败: {str(e)}"
        )


async def extract_embeddings_from_result(embedding_result: Dict[str, Any]) -> List[List[float]]:
    """
    从API返回的结果中提取embeddings，支持不同格式的返回结果
    
    Args:
        embedding_result (Dict[str, Any]): API返回的结果
        
    Returns:
        List[List[float]]: 提取的embeddings列表
    """
    # 如果已经是标准格式
    if "embeddings" in embedding_result and isinstance(embedding_result["embeddings"], list):
        return embedding_result["embeddings"]
    
    # 处理OpenAI和类似格式
    if "data" in embedding_result and isinstance(embedding_result["data"], list):
        data = embedding_result["data"]
        embeddings = []
        for item in data:
            if isinstance(item, dict) and "embedding" in item:
                embeddings.append(item["embedding"])
        if embeddings:
            return embeddings
    
    # 如果有嵌套的embeddings字段
    for key, value in embedding_result.items():
        if key == "embeddings" and isinstance(value, list):
            return value
    
    # 尝试找到可能的嵌套结构
    raw_response = embedding_result.get("raw_response", "")
    if raw_response and isinstance(raw_response, str):
        try:
            import json
            # 尝试将字符串解析为JSON
            response_dict = json.loads(raw_response)
            # 寻找嵌套的embeddings
            if "embeddings" in response_dict:
                return response_dict["embeddings"]
            if "data" in response_dict and isinstance(response_dict["data"], list):
                embeddings = []
                for item in response_dict["data"]:
                    if isinstance(item, dict) and "embedding" in item:
                        embeddings.append(item["embedding"])
                if embeddings:
                    return embeddings
        except:
            pass
    
    # 如果没有找到标准格式，记录原始数据并返回空列表
    print(f"无法从结果中提取embeddings: {str(embedding_result)[:500]}...")
    return []


async def process_text_chunks(
    db: Session,
    knowledge_id: str,
    file_id: str,
    text_chunks: List[str] = None,
    chunking_config: Dict[str, Any] = None,
    embedding_model_id: str = None
):
    """
    处理文本块，生成向量并存储到向量数据库
    
    Args:
        db: 数据库会话
        knowledge_id: 知识库ID
        file_id: 文件ID
        text_chunks: 文本块列表(可选，如果为None则从数据库中读取)
        chunking_config: 分段配置(可选，如果为None则从知识库配置中读取)
        embedding_model_id: 嵌入模型ID(可选，如果为None则从知识库配置中读取)
    """
    # 判断是否需要在函数结束时关闭db会话
    should_close_db = False
    
    try:
        # 确保db是数据库会话对象
        if not isinstance(db, Session):
            print(f"错误: db参数类型错误，期望Session，得到{type(db)}")
            from app.db.session import SessionLocal
            db = SessionLocal()
            should_close_db = True
            print("已创建新的数据库会话")
            
        # 获取文件对象
        file = get_knowledge_file(db, file_id)
        if not file:
            print(f"文件不存在: {file_id}")
            return
        
        # 获取知识库对象
        knowledge = get_knowledge(db, knowledge_id)
        if not knowledge:
            print(f"知识库不存在: {knowledge_id}")
            file.embedding_status = "failed"
            file.error = "知识库不存在"
            db.commit()
            return
            
        # 如果没有提供嵌入模型ID，则从知识库配置中获取
        if not embedding_model_id:
            embedding_model_id = knowledge.embedding_model
            
        if not embedding_model_id:
            file.embedding_status = "failed"
            file.error = "未配置嵌入模型"
            db.commit()
            print(f"文件 {file_id} 处理失败: 未配置嵌入模型")
            return
            
        # 如果没有提供分段配置，则从知识库配置中获取
        if chunking_config is None:
            chunking_config = knowledge.config.get("chunking_config", {})
            if not chunking_config:
                print(f"警告: 知识库 {knowledge_id} 没有配置分段参数，将使用默认配置")
                chunking_config = {
                    "method": "recursive",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "separators": ["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
                }
        
        # 如果没有提供文本块列表，则从数据库中读取
        if text_chunks is None:
            if file.text_chunking_result:
                print(f"从数据库获取已存储的分段结果，文件ID: {file_id}")
                text_chunks = file.text_chunking_result
            else:
                # 如果数据库中没有分段结果，但有文本内容，则进行分段
                if file.text_content:
                    # 更新状态为分段中
                    file.status = "chunking"
                    db.commit()
                    
                    # 根据配置进行文本分段
                    try:
                        if chunking_config.get("method") == "recursive":
                            # 使用递归分段
                            from langchain.text_splitter import RecursiveCharacterTextSplitter
                            
                            # 确保分隔符是列表形式
                            separators = chunking_config.get("separators", ["\n\n", "\n", "。", "！", "？", ".", "!", "?"])
                            if isinstance(separators, str):
                                separators = separators.split(",")
                            
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=chunking_config.get("chunk_size", 1000),
                                chunk_overlap=chunking_config.get("chunk_overlap", 200),
                                separators=separators
                            )
                            text_chunks = splitter.split_text(file.text_content)
                        elif chunking_config.get("method") == "character":
                            # 使用字符分段
                            from langchain.text_splitter import CharacterTextSplitter
                            splitter = CharacterTextSplitter(
                                chunk_size=chunking_config.get("chunk_size", 1000),
                                chunk_overlap=chunking_config.get("chunk_overlap", 200),
                                separator=chunking_config.get("separator", "\n\n")
                            )
                            text_chunks = splitter.split_text(file.text_content)
                        elif chunking_config.get("method") == "token":
                            # 使用Token分段
                            from langchain.text_splitter import TokenTextSplitter
                            splitter = TokenTextSplitter(
                                chunk_size=chunking_config.get("chunk_size", 1000),
                                chunk_overlap=chunking_config.get("chunk_overlap", 200),
                                encoding_name=chunking_config.get("encoding_name", "cl100k_base")
                            )
                            text_chunks = splitter.split_text(file.text_content)
                        else:
                            # 默认使用简单分段
                            from langchain.text_splitter import CharacterTextSplitter
                            splitter = CharacterTextSplitter(
                                chunk_size=chunking_config.get("chunk_size", 1000),
                                chunk_overlap=chunking_config.get("chunk_overlap", 200),
                                separator=chunking_config.get("separator", "\n\n")
                            )
                            text_chunks = splitter.split_text(file.text_content)
                        
                        # 保存分段结果到数据库
                        file.text_chunking_result = text_chunks
                        file.chunking_time = datetime.now()
                        file.chunk_count = len(text_chunks)
                        file.status = "chunked"  # 更新状态为分段完成
                        db.commit()
                        
                        print(f"生成并保存分段结果到数据库，文件ID: {file_id}，分段数: {len(text_chunks)}")
                    except Exception as e:
                        file.embedding_status = "failed"
                        file.error = f"分段失败: {str(e)}"
                        db.commit()
                        print(f"文件分段失败: {str(e)}")
                        return
                else:
                    file.embedding_status = "failed"
                    file.error = "文本内容不存在"
                    db.commit()
                    print(f"文件 {file_id} 没有文本内容")
                    return
        
        if not text_chunks or len(text_chunks) == 0:
            file.embedding_status = "failed"
            file.error = "分段结果为空"
            db.commit()
            print(f"文件 {file_id} 分段结果为空")
            return
                
        # 更新状态为训练中
        file.embedding_status = "processing"
        db.commit()
        print(f"开始处理文件 {file_id} 的向量化，共 {len(text_chunks)} 个文本块")
        
        # 将文本块分批处理，每批最多100个块
        batch_size = 100
        total_vectors = 0
        
        # 获取向量存储
        vector_store = EmbeddingManager.get_vector_store()
        if not vector_store:
            file.embedding_status = "failed"
            file.error = "向量存储未初始化，请安装 pymilvus 包: pip install pymilvus"
            db.commit()
            print(f"文件 {file_id} 处理失败: 向量存储未初始化，请安装 pymilvus 包")
            return
            
        # 检查向量存储客户端是否可用
        if vector_store.client is None:
            file.embedding_status = "failed"
            file.error = "向量存储客户端初始化失败，请检查 pymilvus 配置"
            db.commit()
            print(f"文件 {file_id} 处理失败: 向量存储客户端初始化失败")
            return
        
        # 分批处理
        for i in range(0, len(text_chunks), batch_size):
            batch_chunks = text_chunks[i:i+batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1}，共 {len(batch_chunks)} 个文本块")
            
            try:
                # 使用模型生成向量
                embedding_result = await execute_model_inference(
                    db,
                    embedding_model_id,
                    {
                        "input": batch_chunks,
                        "model_type": "embedding"
                    }
                )
                
                if "error" in embedding_result:
                    raise Exception(f"生成向量失败: {embedding_result['error']}")
                
                # 获取生成的向量
                embeddings = embedding_result.get("embeddings", [])

                if not embeddings:
                    # 尝试使用新函数提取embeddings
                    embeddings = await extract_embeddings_from_result(embedding_result)
                    
                if not embeddings:
                    # 尝试从原始响应中解析
                    raw_response = embedding_result.get("raw_response", "")
                    if raw_response:
                        print(f"尝试从原始响应解析embeddings: {raw_response[:200]}...")
                    # 记录更多的调试信息
                    print(f"完整的embedding_result: {str(embedding_result)[:500]}...")
                    raise Exception(f"生成向量为空，请检查嵌入模型 {embedding_model_id} 的配置和状态")
                
                if len(embeddings) != len(batch_chunks):
                    print(f"警告: 生成的向量数量 ({len(embeddings)}) 与文本块数量 ({len(batch_chunks)}) 不匹配")
                
                # 准备要存储的数据
                vectors_to_store = []
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    vectors_to_store.append({
                        "knowledge_id": knowledge_id,
                        "file_id": file_id,
                        "chunk_index": i + j,
                        "text": chunk,
                        "embedding": embedding,
                        "metadata": {
                            "file_name": file.original_filename if hasattr(file, 'original_filename') else file.filename,
                            "file_type": file.file_type
                        }
                    })
                
                # 批量存储到向量数据库
                if vectors_to_store:
                    # 准备数据格式
                    chunks_list = []
                    embeddings_list = []
                    for item in vectors_to_store:
                        chunks_list.append({
                            "text": item["text"],
                            "metadata": {
                                "file_id": file_id,
                                "chunk_index": item["chunk_index"],
                                "knowledge_id": knowledge_id,
                                **item["metadata"]
                            }
                        })
                        embeddings_list.append(item["embedding"])
                    
                    vector_store.insert_vectors(
                        knowledge_id=knowledge_id,
                        file_id=file_id,
                        chunks=chunks_list,
                        embeddings=embeddings_list
                    )
                    total_vectors += len(vectors_to_store)
                    print(f"已存储 {len(vectors_to_store)} 个向量，总计 {total_vectors} 个")
            
            except Exception as e:
                print(f"批次处理失败: {str(e)}")
                # 继续处理下一批，而不是完全失败
                continue
        
        # 更新文件状态
        if total_vectors > 0:
            file.embedding_status = "completed"  # 更新状态为训练完成
            file.vector_count = total_vectors
            file.status = "indexed"  # 文件整体状态为索引完成
            file.error = None
            print(f"文件 {file_id} 向量化完成，共处理 {total_vectors} 个向量")
        else:
            file.embedding_status = "failed"
            file.error = "未生成任何向量"
            print(f"文件 {file_id} 向量化失败: 未生成任何向量")
        
        db.commit()
        return total_vectors
    
    except Exception as e:
        # 尝试更新文件状态为失败
        try:
            file = get_knowledge_file(db, file_id)
            if file:
                file.embedding_status = "failed"
                file.error = str(e)
                db.commit()
        except Exception as update_error:
            print(f"更新文件状态失败: {str(update_error)}")
        
        print(f"处理文本块失败: {str(e)}")
        raise Exception(f"处理文本块失败: {str(e)}")
    finally:
        # 如果是我们创建的会话，在最后关闭它
        if should_close_db and db:
            db.close()
            print(f"已关闭数据库会话")


@router.post("/{knowledge_id}/files/{file_id}/process")
async def process_file(
    knowledge_id: str,
    file_id: str,
    data: FileProcessRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    处理知识库文件，包括关联文件和向量化
    """
    try:
        # 检查知识库是否存在
        knowledge = get_knowledge(db=db, knowledge_id=knowledge_id)
        if not knowledge:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="知识库不存在"
            )
        
        # 检查文件是否存在
        file = db.query(models.File).filter(models.File.id == file_id).first()
        if not file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="文件不存在"
            )
        
        # 检查文件是否已经关联到该知识库
        existing = db.query(models.KnowledgeFile).filter(
            models.KnowledgeFile.knowledge_id == knowledge_id,
            models.KnowledgeFile.file_id == file_id
        ).first()

        if existing:
            # 如果文件已关联，检查是否需要重新处理
            if existing.status == "indexed" and not data.force_reprocess:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                        detail="文件已经处理完成，如需重新处理请设置 force_reprocess=True"
                    )
                
            # 更新现有记录的状态
            existing.status = "processing"
            existing.embedding_status = "pending"
            existing.error = None
            existing.updated_at = datetime.now()
            db.add(existing)
            knowledge_file = existing
        else:
            # 创建新的知识库文件记录
            knowledge_file = models.KnowledgeFile(
                knowledge_id=knowledge_id,
                file_id=file_id,
                filename=file.filename,
                original_filename=file.original_filename,
                file_type=file.file_type,
                file_size=file.file_size,
                status="processing",  # 设置初始状态为处理中
                embedding_status="pending",  # 设置向量化状态为等待处理
                text_content=file.text_content,  # 复制文件内容
                extra_data=file.extra_data  # 复制额外数据
            )
            db.add(knowledge_file)

        # 更新知识库文件计数和大小
        knowledge.file_count = db.query(models.KnowledgeFile).filter(
            models.KnowledgeFile.knowledge_id == knowledge_id
        ).count()
        knowledge.total_size = db.query(func.sum(models.KnowledgeFile.file_size)).filter(
            models.KnowledgeFile.knowledge_id == knowledge_id
        ).scalar() or 0

        # 提交数据库更改
        db.commit()
        db.refresh(knowledge_file)
        
        # 添加后台处理任务
        background_tasks.add_task(
            process_knowledge_file,
            knowledge_file.id,
            file.file_type,
            knowledge_id,
            file.filename
        )
        
        return {
            "code": 200,
            "message": "文件处理任务已启动",
            "data": {
                "knowledge_file_id": knowledge_file.id,
                "status": knowledge_file.status,
                "embedding_status": knowledge_file.embedding_status
            }
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理文件失败: {str(e)}"
        )


@router.post("/{knowledge_id}/files/{file_id}/embed")
async def embed_file(
    knowledge_id: str,
    file_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    对知识库文件进行向量化处理
    """
    try:
        # 检查向量存储组件是否可用
        if not HAS_PYMILVUS:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "code": 500,
                    "msg": "pymilvus 模块未安装，向量化功能不可用。请使用 pip install pymilvus 安装。",
                    "data": None
                }
            )
            
        # 获取向量存储实例
        vector_store = EmbeddingManager.get_vector_store()
        if not vector_store or vector_store.client is None:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "code": 500,
                    "msg": "向量存储组件初始化失败，请确保已安装 pymilvus 并配置正确。",
                    "data": None
                }
            )
        
        # 检查知识库是否存在
        knowledge = get_knowledge(db=db, knowledge_id=knowledge_id)
        if not knowledge:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "code": 404,
                    "msg": "知识库不存在",
                    "data": None
                }
            )
        
        # 检查文件是否存在
        file = get_knowledge_file(db=db, file_id=file_id)
        if not file:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "code": 404,
                    "msg": "文件不存在",
                    "data": None
                }
            )
        
        # 检查文件是否属于指定知识库
        if file.knowledge_id != knowledge_id:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "code": 400,
                    "msg": "文件不属于该知识库",
                    "data": None
                }
            )
        
        # 检查是否已有文本内容
        text_content = None
        if file.text_content:
            print(f"从数据库获取已提取的文本内容，文件ID: {file_id}")
            text_content = file.text_content
        else:
            # 尝试从原始文件中提取文本
            try:
                # 获取原始文件路径
                if file.file_type:
                    file_extension = file.file_type
                    file_name_with_ext = f"{file.filename}.{file_extension}"
                else:
                    file_name_with_ext = file.filename
                
                original_file_path = os.path.join(
                    os.getcwd(),
                    "uploads",
                    "raw",
                    knowledge_id,
                    file_name_with_ext
                )
                
                if not os.path.exists(original_file_path):
                    return JSONResponse(
                        status_code=status.HTTP_200_OK,
                        content={
                            "code": 404,
                            "msg": "原始文件不存在，无法提取文本",
                            "data": None
                        }
                    )
                
                # 从原始文件中提取文本
                text_content = await extract_text_from_file(original_file_path, file.file_type)
                
                # 保存提取的文本到数据库
                file.text_content = text_content
                file.text_extraction_time = datetime.now()
                db.commit()
                
                print(f"已从原始文件提取文本并保存到数据库，文件ID: {file_id}")
            except Exception as e:
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "code": 500,
                        "msg": f"从原始文件提取文本失败: {str(e)}",
                        "data": None
                    }
                )
        
        # 获取知识库的分段配置
        chunking_config = knowledge.config.get("chunking_config", {})
        
        # 根据配置进行文本分段
        text_chunks = []
        try:
            if chunking_config.get("method") == "recursive":
                # 使用递归分段
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                
                # 确保分隔符是列表形式
                separators = chunking_config.get("separators", ["\n\n", "\n", "。", "！", "？", ".", "!", "?"])
                if isinstance(separators, str):
                    separators = separators.split(",")
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunking_config.get("chunk_size", 1000),
                    chunk_overlap=chunking_config.get("chunk_overlap", 200),
                    separators=separators
                )
                text_chunks = splitter.split_text(text_content)
            elif chunking_config.get("method") == "character":
                # 使用字符分段
                from langchain.text_splitter import CharacterTextSplitter
                splitter = CharacterTextSplitter(
                    chunk_size=chunking_config.get("chunk_size", 1000),
                    chunk_overlap=chunking_config.get("chunk_overlap", 200),
                    separator=chunking_config.get("separator", "\n\n")
                )
                text_chunks = splitter.split_text(text_content)
            elif chunking_config.get("method") == "token":
                # 使用Token分段
                from langchain.text_splitter import TokenTextSplitter
                splitter = TokenTextSplitter(
                    chunk_size=chunking_config.get("chunk_size", 1000),
                    chunk_overlap=chunking_config.get("chunk_overlap", 200),
                    encoding_name=chunking_config.get("encoding_name", "cl100k_base")
                )
                text_chunks = splitter.split_text(text_content)
            else:
                # 默认使用简单分段
                from langchain.text_splitter import CharacterTextSplitter
                splitter = CharacterTextSplitter(
                    chunk_size=chunking_config.get("chunk_size", 1000),
                    chunk_overlap=chunking_config.get("chunk_overlap", 200),
                    separator=chunking_config.get("separator", "\n\n")
                )
                text_chunks = splitter.split_text(text_content)
            
            # 保存分段结果到数据库
            file.text_chunking_result = text_chunks
            file.chunking_time = datetime.now()
            db.commit()
            
            print(f"已完成文本分段并保存结果到数据库，文件ID: {file_id}，分段数: {len(text_chunks)}")
            
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": f"文本分段失败: {str(e)}"}
            )
        
        # 更新文件状态
        file.status = "indexed"
        file.chunk_count = len(text_chunks)
        file.embedding_status = "processing"
        db.commit()
        
        # 检查嵌入模型是否存在
        if not knowledge.embedding_model:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": "知识库未配置嵌入模型，无法进行向量化"}
            )
        
        # 添加后台任务处理文本块
        background_tasks.add_task(
            process_text_chunks,
            db,
            knowledge_id,
            file_id,
            text_chunks,
            chunking_config,
            knowledge.embedding_model
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "code": 200,
                "msg": "文件向量化处理已开始",
                "data": {
                    "chunk_count": len(text_chunks),
                    "file_id": file_id
                }
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "code": 500,
                "msg": f"文件向量化处理失败: {str(e)}",
                "data": None
            }
        )


@router.post("/trigger_file_embedding/{knowledge_id}/{file_id}", response_model=List[Dict[str, Any]])
async def trigger_file_embedding_after_processing(
    knowledge_id: str,
    file_id: str,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    在文件处理完成后自动触发向量化过程
    """
    # 如果是通过路由调用，使用依赖注入的db
    # 如果是通过后台任务调用，则创建新的数据库会话
    if isinstance(db, Session):
        # 使用传入的会话
        db_session = db
        should_close_db = False
    else:
        # 创建新的会话
        from app.db.session import SessionLocal
        db_session = SessionLocal()
        should_close_db = True
    
    try:
        # 检查知识库是否存在
        knowledge = db_session.query(models.Knowledge).filter(models.Knowledge.id == knowledge_id).first()
        if not knowledge:
            if should_close_db:
                db_session.close()
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": f"知识库 ID {knowledge_id} 不存在"}
            )
        
        # 检查文件是否存在
        file = db_session.query(models.KnowledgeFile).filter(
            models.KnowledgeFile.id == file_id,
            models.KnowledgeFile.knowledge_id == knowledge_id
        ).first()
        
        if not file:
            if should_close_db:
                db_session.close()
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": f"文件 ID {file_id} 不存在或不属于指定知识库"}
            )
        
        # 检查文件是否已完成处理
        if file.status != "indexed":
            if should_close_db:
                db_session.close()
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": f"文件 ID {file_id} 未完成索引，当前状态: {file.status}"}
            )
        
        # 检查嵌入模型是否存在
        if not knowledge.embedding_model:
            if should_close_db:
                db_session.close()
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "知识库未配置嵌入模型，无法进行向量化"}
            )
        
        # 获取分块配置
        chunking_config = knowledge.config.get("chunking_config", {})
        
        # 更新文件状态
        file.embedding_status = "processing"
        db_session.commit()
        
        # 添加后台任务处理文本块
        background_tasks.add_task(
            process_text_chunks,
            db_session,
            knowledge_id,
            file_id,
            file.text_chunking_result,
            chunking_config,
            knowledge.embedding_model
        )
        
        if should_close_db:
            # 如果是后台任务调用，不关闭会话，因为它被传递给了process_text_chunks
            # db_session将由process_text_chunks负责关闭
            pass
            
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": f"已触发文件 ID {file_id} 的向量化处理"}
        )
    except Exception as e:
        print(f"触发文件向量化时出错: {str(e)}")
        if should_close_db:
            db_session.close()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"触发文件向量化失败: {str(e)}"}
        )


@router.get("/file/{file_id}")
async def get_file_detail(
    file_id: str,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    获取文件详情，包括分段和向量化信息
    """
    try:
        # 查询文件
        file = db.query(KnowledgeFile).filter(KnowledgeFile.id == file_id).first()
        if not file:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": f"文件 ID {file_id} 不存在"}
            )
        
        # 返回文件详情，包含文本分段结果
        file_data = file.to_dict()
        
        # 确保返回文本分段结果
        if hasattr(file, 'text_chunking_result') and file.text_chunking_result:
            file_data['text_chunking_result'] = file.text_chunking_result
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "code": 200,
                "msg": "获取文件详情成功",
                "data": file_data
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"获取文件详情失败: {str(e)}"}
        )


@router.get("/file/{file_id}/chunks")
async def get_file_chunks(
    file_id: str,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    获取文件的文本分段
    """
    try:
        # 查询文件
        file = db.query(KnowledgeFile).filter(KnowledgeFile.id == file_id).first()
        if not file:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": f"文件 ID {file_id} 不存在"}
            )
        
        # 检查文件是否有分段结果
        if not file.text_chunking_result:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "code": 404,
                    "msg": "文件尚未完成分段处理",
                    "data": {
                        "chunks": []
                    }
                }
            )
        
        # 将纯文本分段转换为带有元数据的对象
        chunks = []
        for i, text in enumerate(file.text_chunking_result):
            chunks.append({
                "text": text,
                "metadata": {
                    "index": i,
                    "file_id": file_id,
                    "knowledge_id": file.knowledge_id
                }
            })
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "code": 200,
                "msg": "获取文件分段成功",
                "data": {
                    "chunks": chunks,
                    "total": len(chunks)
                }
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"获取文件分段失败: {str(e)}"}
        )


@router.get("/file/{file_id}/embeddings")
async def get_file_embeddings(
    file_id: str,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    获取文件的向量表示
    """
    try:
        # 查询文件
        file = db.query(KnowledgeFile).filter(KnowledgeFile.id == file_id).first()
        if not file:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": f"文件 ID {file_id} 不存在"}
            )
        
        # 检查文件是否已向量化
        if file.embedding_status != "completed":
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "code": 400,
                    "msg": f"文件尚未完成向量化，当前状态: {file.embedding_status}",
                    "data": {
                        "embeddings": []
                    }
                }
            )
        
        # 检查是否有分段结果
        if not file.text_chunking_result or len(file.text_chunking_result) == 0:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "code": 404,
                    "msg": "文件没有文本分段数据",
                    "data": {
                        "embeddings": []
                    }
                }
            )
        
        # 从向量数据库获取向量
        try:
            # 获取向量存储实例
            vector_store = EmbeddingManager.get_vector_store()
            if not vector_store or not vector_store.client:
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "code": 500,
                        "msg": "向量存储服务不可用",
                        "data": {
                            "embeddings": []
                        }
                    }
                )
            
            # 使用Milvus的query API获取向量数据
            knowledge_id = file.knowledge_id
            collection_name = vector_store.get_collection_name(knowledge_id)
            
            # 构建过滤表达式，获取特定文件ID的所有向量
            filter_expr = f"file_id == '{file_id}'"
            
            try:
                # 先查询所有匹配的实体，包括向量数据
                query_result = vector_store.client.query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=["id", "text", "vector", "file_id", "chunk_index", "knowledge_id"]
                )
                
                print(f"从Milvus查询文件 {file_id} 的向量，返回结果: {len(query_result) if query_result else 0} 条")
                
                if not query_result or len(query_result) == 0:
                    print(f"未在Milvus中找到文件 {file_id} 的向量数据，可能使用了不同的ID格式存储")
                    # 尝试使用不同的过滤条件再次查询
                    # 假设向量存储时可能使用了一些前缀/后缀
                    fallback_results = vector_store.client.query(
                        collection_name=collection_name,
                        limit=100,  # 限制返回数量
                        output_fields=["id", "text", "vector", "file_id", "chunk_index", "knowledge_id"]
                    )
                    
                    # 在返回结果中找出匹配的文件ID
                    if fallback_results:
                        filtered_results = [
                            item for item in fallback_results 
                            if item.get("file_id") == file_id or str(item.get("file_id")).endswith(file_id[-8:])
                        ]
                        if filtered_results:
                            query_result = filtered_results
                            print(f"通过后备方法找到了 {len(filtered_results)} 条匹配的向量数据")
                
                # 如果查询失败或没有结果，使用模拟数据
                if not query_result or len(query_result) == 0:
                    # 从分段文本生成带元数据的分段
                    chunks = []
                    embeddings = []
                    
                    for i, text in enumerate(file.text_chunking_result):
                        chunks.append({
                            "text": text,
                            "metadata": {
                                "index": i,
                                "file_id": file_id,
                                "knowledge_id": knowledge_id
                            }
                        })
                        
                        # 生成模拟向量
                        embeddings.append(
                            [float(i%10) * 0.1 for i in range(768)]
                        )
                    
                    return JSONResponse(
                        status_code=status.HTTP_200_OK,
                        content={
                            "code": 200,
                            "msg": "未能从向量数据库获取真实向量，返回模拟数据",
                            "data": {
                                "embeddings": embeddings,
                                "chunks": chunks,
                                "total": len(embeddings),
                                "is_mock": True
                            }
                        }
                    )
                
                # 处理查询结果
                chunks = []
                embeddings = []
                
                # 按chunk_index排序
                query_result.sort(key=lambda x: x.get("chunk_index", 0))
                
                for item in query_result:
                    vector = item.get("vector")
                    text = item.get("text", "")
                    chunk_index = item.get("chunk_index", 0)
                    
                    if vector and isinstance(vector, (list, tuple)) or hasattr(vector, '__iter__'):
                        # 将vector转换为普通Python float列表以确保可JSON序列化
                        # 处理numpy数组或其他非原生类型
                        try:
                            # 对于numpy数组，需要显式转换为Python原生float类型
                            vector_list = [float(v) for v in vector]
                            embeddings.append(vector_list)
                            chunks.append({
                                "text": text,
                                "metadata": {
                                    "index": chunk_index,
                                    "file_id": file_id,
                                    "knowledge_id": knowledge_id,
                                    "milvus_id": item.get("id")
                                }
                            })
                        except Exception as convert_error:
                            print(f"向量格式转换出错: {str(convert_error)}, 类型: {type(vector)}")
                
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "code": 200,
                        "msg": "成功从向量数据库获取向量",
                        "data": {
                            "embeddings": embeddings,
                            "chunks": chunks,
                            "total": len(embeddings),
                            "is_mock": False
                        }
                    }
                )
                
            except Exception as query_error:
                print(f"查询Milvus向量时出错: {str(query_error)}")
                
                # 如果查询出错，使用模拟数据
                chunks = []
                embeddings = []
                
                for i, text in enumerate(file.text_chunking_result):
                    chunks.append({
                        "text": text,
                        "metadata": {
                            "index": i,
                            "file_id": file_id,
                            "knowledge_id": knowledge_id
                        }
                    })
                    
                    # 生成模拟向量
                    embeddings.append(
                        [float(i%10) * 0.1 for i in range(768)]
                    )
                
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "code": 200,
                        "msg": f"查询向量数据库时出错，返回模拟数据: {str(query_error)}",
                        "data": {
                            "embeddings": embeddings,
                            "chunks": chunks,
                            "total": len(embeddings),
                            "is_mock": True
                        }
                    }
                )
            
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "code": 500,
                    "msg": f"从向量数据库获取向量失败: {str(e)}",
                    "data": {
                        "embeddings": []
                    }
                }
            )
        
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"获取文件向量表示失败: {str(e)}"}
        )


@router.post("/{knowledge_id}/associate-files")
async def associate_files(
    knowledge_id: str,
    data: Dict[str, List[str]],  # {'file_ids': [id1, id2, ...]}
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_knowledge_admin),
):
    """
    将文件关联到知识库
    """
    # 检查知识库是否存在
    knowledge = get_knowledge(db=db, knowledge_id=knowledge_id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )
    
    file_ids = data.get('file_ids', [])
    if not file_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="未提供文件ID列表"
        )
    
    # 获取所有文件记录
    files = db.query(models.File).filter(models.File.id.in_(file_ids)).all()
    if not files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="未找到指定的文件"
        )
    
    # 记录成功和失败的文件
    success_files = []
    failed_files = []
    
    try:
        # 遍历文件，创建知识库文件记录
        for file in files:
            try:
                # 检查文件是否已经关联到该知识库
                existing = db.query(models.KnowledgeFile).filter(
                    models.KnowledgeFile.knowledge_id == knowledge_id,
                    models.KnowledgeFile.file_id == file.id
            ).first()
            
                if existing:
                    failed_files.append({
                        "id": file.id,
                        "name": file.original_filename,
                        "reason": "文件已关联到该知识库"
                    })
                continue
            
                # 创建新的知识库文件记录
                knowledge_file = models.KnowledgeFile(
                knowledge_id=knowledge_id,
                    file_id=file.id,
                filename=file.filename,
                original_filename=file.original_filename,
                file_type=file.file_type,
                file_size=file.file_size,
                    status="processing",  # 设置初始状态为处理中
                    embedding_status="pending",  # 设置向量化状态为等待处理
                    text_content=file.text_content,  # 复制文件内容
                    extra_data=file.extra_data  # 复制额外数据
            )
            
                db.add(knowledge_file)
                success_files.append({
                    "id": file.id,
                    "name": file.original_filename
                })
            
            # 更新知识库文件计数和大小
                knowledge.file_count = db.query(models.KnowledgeFile).filter(
                    models.KnowledgeFile.knowledge_id == knowledge_id
                ).count()
                knowledge.total_size = db.query(func.sum(models.KnowledgeFile.file_size)).filter(
                    models.KnowledgeFile.knowledge_id == knowledge_id
                ).scalar() or 0
                
            except Exception as e:
                failed_files.append({
                    "id": file.id,
                    "name": file.original_filename,
                    "reason": str(e)
                })
        
        # 提交所有更改
            db.commit()
            
        # 为每个成功关联的文件添加后台处理任务
        for file in success_files:
            background_tasks.add_task(
                process_knowledge_file,
                file["id"],
                file["name"],
                knowledge_id,
                file["id"]
            )
        
        return {
            "code": 200,
                "message": f"成功关联 {len(success_files)} 个文件，失败 {len(failed_files)} 个",
                "data": {
            "success": success_files,
            "failed": failed_files
        } 
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"关联文件失败: {str(e)}"
        )


@router.post("/{knowledge_id}/files/{file_id}/preview-chunking")
async def preview_file_chunking(
    knowledge_id: str,
    file_id: str,
    data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    预览文件分段结果，不进行实际存储
    
    参数:
        knowledge_id: 知识库ID
        file_id: 文件ID
        data: 包含chunking_config的字典，指定分段配置
    
    返回:
        分段结果列表
    """
    try:
        # 检查知识库是否存在
        knowledge = get_knowledge(db=db, knowledge_id=knowledge_id)
        if not knowledge:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "知识库不存在"}
            )
        
        # # 检查文件是否存在
        # file = get_knowledge_file(db=db, file_id=file_id)
        # if not file:
        #     return JSONResponse(
        #         status_code=status.HTTP_404_NOT_FOUND,
        #         content={"message": "文件不存在"}
        #     )
        
        # # 检查文件是否属于指定知识库
        # if file.knowledge_id != knowledge_id:
        #     return JSONResponse(
        #         status_code=status.HTTP_400_BAD_REQUEST,
        #         content={"message": "文件不属于该知识库"}
        #     )
        
        # 从files表中获取文件内容
        original_file = db.query(models.File).filter(models.File.id == file_id).first()
        if not original_file:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "原始文件不存在"}
            )
            
        # 获取文本内容
        text_content = original_file.text_content
        if not text_content:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "文件未提取文本内容，请先处理文件"}
            )
        
        # 获取分段配置
        chunking_config = data.get("chunking_config", {})
        if not chunking_config:
            # 如果未提供配置，使用知识库默认配置
            chunking_config = knowledge.config.get("chunking_config", {})
            if not chunking_config:
                # 使用默认配置
                chunking_config = {
                    "method": "recursive",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "separators": ["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
                }
        
        # 根据配置进行文本分段
        text_chunks = []
        try:
            if chunking_config.get("method") == "recursive":
                # 使用递归分段
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                
                # 确保分隔符是列表形式
                separators = chunking_config.get("separators", ["\n\n", "\n", "。", "！", "？", ".", "!", "?"])
                if isinstance(separators, str):
                    separators = separators.split(",")
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunking_config.get("chunk_size", 1000),
                    chunk_overlap=chunking_config.get("chunk_overlap", 200),
                    separators=separators
                )
                text_chunks = splitter.split_text(text_content)
            elif chunking_config.get("method") == "character":
                # 使用字符分段
                from langchain.text_splitter import CharacterTextSplitter
                splitter = CharacterTextSplitter(
                    chunk_size=chunking_config.get("chunk_size", 1000),
                    chunk_overlap=chunking_config.get("chunk_overlap", 200),
                    separator=chunking_config.get("separator", "\n\n")
                )
                text_chunks = splitter.split_text(text_content)
            elif chunking_config.get("method") == "token":
                # 使用Token分段
                from langchain.text_splitter import TokenTextSplitter
                splitter = TokenTextSplitter(
                    chunk_size=chunking_config.get("chunk_size", 1000),
                    chunk_overlap=chunking_config.get("chunk_overlap", 200),
                    encoding_name=chunking_config.get("encoding_name", "cl100k_base")
                )
                text_chunks = splitter.split_text(text_content)
            elif chunking_config.get("method") == "sentence":
                # 使用句子分段
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunking_config.get("chunk_size", 1000),
                    chunk_overlap=chunking_config.get("chunk_overlap", 200),
                    separators=[chunking_config.get("separator", "。") or "。", "！", "？", ".", "!", "?"]
                )
                text_chunks = splitter.split_text(text_content)
            elif chunking_config.get("method") == "paragraph":
                # 使用段落分段
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunking_config.get("chunk_size", 1000),
                    chunk_overlap=chunking_config.get("chunk_overlap", 200),
                    separators=[chunking_config.get("separator", "\n\n") or "\n\n", "\n", "\r\n"]
                )
                text_chunks = splitter.split_text(text_content)
            else:
                # 默认使用简单分段
                from langchain.text_splitter import CharacterTextSplitter
                splitter = CharacterTextSplitter(
                    chunk_size=chunking_config.get("chunk_size", 1000),
                    chunk_overlap=chunking_config.get("chunk_overlap", 200),
                    separator=chunking_config.get("separator", "\n\n")
                )
                text_chunks = splitter.split_text(text_content)
            
            # 限制返回的文本块数量，避免响应过大
            max_chunks = 50
            if len(text_chunks) > max_chunks:
                sample_chunks = text_chunks[:max_chunks]
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "code": 200,
                        "message": f"分段成功，共 {len(text_chunks)} 个分段，显示前 {max_chunks} 个",
                        "data": {
                            "chunks": sample_chunks,
                            "total": len(text_chunks),
                            "is_sample": True,
                            "max_chunks": max_chunks
                        }
                    }
                )
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "code": 200,
                    "message": f"分段成功，共 {len(text_chunks)} 个分段",
                    "data": {
                        "chunks": text_chunks,
                        "total": len(text_chunks),
                        "is_sample": False
                    }
                }
            )
            
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "code": 500,
                    "message": f"文本分段失败: {str(e)}",
                    "data": None
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "code": 500,
                "message": f"处理请求失败: {str(e)}",
                "data": None
            }
        )


@router.post("/{knowledge_id}/files/{file_id}/process-with-config")
async def process_file_with_config(
    knowledge_id: str,
    file_id: str,
    data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    使用自定义分段配置处理知识库文件，进行分段和向量化
    
    参数:
        knowledge_id: 知识库ID
        file_id: 文件ID
        data: 包含chunking_config的字典，指定分段配置
    """
    try:
        # 检查知识库是否存在
        knowledge = get_knowledge(db=db, knowledge_id=knowledge_id)
        if not knowledge:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "知识库不存在"}
            )
        
        # 检查文件是否存在
        file = get_knowledge_file(db=db, file_id=file_id)
        if not file:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "文件不存在"}
            )
        
        # 检查文件是否属于指定知识库
        if file.knowledge_id != knowledge_id:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "文件不属于该知识库"}
            )
        
        # 获取自定义分段配置
        chunking_config = data.get("chunking_config", {})
        if not chunking_config:
            # 如果未提供配置，使用知识库默认配置
            chunking_config = knowledge.config.get("chunking_config", {})
        
        # 获取文本内容
        text_content = file.text_content
        if not text_content:
            # 尝试从原始文件中提取文本
            try:
                # 获取原始文件路径
                if file.file_type:
                    file_extension = file.file_type
                    file_name_with_ext = f"{file.filename}.{file_extension}"
                else:
                    file_name_with_ext = file.filename
                
                original_file_path = os.path.join(
                    os.getcwd(),
                    "uploads",
                    "raw",
                    knowledge_id,
                    file_name_with_ext
                )
                
                if not os.path.exists(original_file_path):
                    return JSONResponse(
                        status_code=status.HTTP_404_NOT_FOUND,
                        content={
                            "code": 404,
                            "message": "原始文件不存在，无法提取文本",
                            "data": None
                        }
                    )
                
                # 从原始文件中提取文本
                text_content = await extract_text_from_file(original_file_path, file.file_type)
                
                # 保存提取的文本到数据库
                file.text_content = text_content
                file.text_extraction_time = datetime.now()
                db.commit()
            except Exception as e:
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "code": 500,
                        "message": f"提取文本失败: {str(e)}",
                        "data": None
                    }
                )
        
        # 根据配置进行文本分段
        text_chunks = []
        try:
            if chunking_config.get("method") == "recursive":
                # 使用递归分段
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                
                # 确保分隔符是列表形式
                separators = chunking_config.get("separators", ["\n\n", "\n", "。", "！", "？", ".", "!", "?"])
                if isinstance(separators, str):
                    separators = separators.split(",")
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunking_config.get("chunk_size", 1000),
                    chunk_overlap=chunking_config.get("chunk_overlap", 200),
                    separators=separators
                )
                text_chunks = splitter.split_text(text_content)
            elif chunking_config.get("method") == "character":
                # 使用字符分段
                from langchain.text_splitter import CharacterTextSplitter
                splitter = CharacterTextSplitter(
                    chunk_size=chunking_config.get("chunk_size", 1000),
                    chunk_overlap=chunking_config.get("chunk_overlap", 200),
                    separator=chunking_config.get("separator", "\n\n")
                )
                text_chunks = splitter.split_text(text_content)
            elif chunking_config.get("method") == "token":
                # 使用Token分段
                from langchain.text_splitter import TokenTextSplitter
                splitter = TokenTextSplitter(
                    chunk_size=chunking_config.get("chunk_size", 1000),
                    chunk_overlap=chunking_config.get("chunk_overlap", 200),
                    encoding_name=chunking_config.get("encoding_name", "cl100k_base")
                )
                text_chunks = splitter.split_text(text_content)
            else:
                # 默认使用简单分段
                from langchain.text_splitter import CharacterTextSplitter
                splitter = CharacterTextSplitter(
                    chunk_size=chunking_config.get("chunk_size", 1000),
                    chunk_overlap=chunking_config.get("chunk_overlap", 200),
                    separator=chunking_config.get("separator", "\n\n")
                )
                text_chunks = splitter.split_text(text_content)
            
            # 保存分段结果到数据库
            file.text_chunking_result = text_chunks
            file.chunking_time = datetime.now()
            file.chunk_count = len(text_chunks)
            file.status = "chunked"  # 更新状态为分段完成
            db.commit()
            
            # 更新文件状态
            file.embedding_status = "processing"
            db.commit()
            
            # 检查嵌入模型是否存在
            if not knowledge.embedding_model:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "code": 400,
                        "message": "知识库未配置嵌入模型，无法进行向量化",
                        "data": None
                    }
                )
            
            # 添加后台任务进行向量化
            background_tasks.add_task(
                process_text_chunks,
                db,
                knowledge_id,
                file_id,
                text_chunks,
                chunking_config,
                knowledge.embedding_model
            )
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "code": 200,
                    "message": f"文件处理已开始，分段成功，共 {len(text_chunks)} 个分段",
                    "data": {
                        "chunk_count": len(text_chunks),
                        "file_id": file_id
                    }
                }
            )
            
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "code": 500,
                    "message": f"文本分段失败: {str(e)}",
                    "data": None
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "code": 500,
                "message": f"处理请求失败: {str(e)}",
                "data": None
            }
        ) 
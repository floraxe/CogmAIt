import os
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import or_
from fastapi import UploadFile, HTTPException, BackgroundTasks
from app.models.file import File as FileModel
from app.models.knowledge import Knowledge, KnowledgeFile
from app.schemas.knowledge import KnowledgeCreate, KnowledgeUpdate, ChunkingConfig, ChunkingMethod as SchemaChunkingMethod
from app.utils.file_processor import chunk_text, extract_text_from_file
from app.utils.chunker import TextChunker, ChunkingMethod
try:
    from app.utils.embedding import EmbeddingManager, HAS_PYMILVUS
except ImportError:
    HAS_PYMILVUS = False
from app.utils.model import get_model, execute_model_inference


def get_knowledge(db: Session, knowledge_id: str) -> Optional[Knowledge]:
    """
    通过ID获取知识库
    
    参数:
        db (Session): 数据库会话
        knowledge_id (str): 知识库ID
    
    返回:
        Optional[Knowledge]: 知识库对象或None
    """
    return db.query(Knowledge).filter(Knowledge.id == knowledge_id).first()


def get_knowledge_list(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    name: Optional[str] = None,
    status: Optional[str] = None
) -> List[Knowledge]:
    """
    获取知识库列表，支持过滤
    
    参数:
        db (Session): 数据库会话
        skip (int): 跳过的记录数
        limit (int): 限制返回的记录数
        name (Optional[str]): 按名称过滤
        status (Optional[str]): 按状态过滤
    
    返回:
        List[Knowledge]: 知识库列表
    """
    query = db.query(Knowledge)
    
    # 应用过滤条件
    if name:
        query = query.filter(Knowledge.name.ilike(f"%{name}%"))
    if status:
        query = query.filter(Knowledge.status == status)
    
    # 应用分页
    return query.offset(skip).limit(limit).all()


def create_knowledge(db: Session, knowledge_in: KnowledgeCreate, user_id: str = None) -> Knowledge:
    """
    创建新知识库
    
    参数:
        db (Session): 数据库会话
        knowledge_in (KnowledgeCreate): 知识库创建模式
        user_id (str): 创建者用户ID
    
    返回:
        Knowledge: 创建的知识库
    """
    # 创建新知识库对象
    db_knowledge = Knowledge(
        name=knowledge_in.name,
        description=knowledge_in.description,
        vector_type=knowledge_in.vector_type,
        embedding_model=knowledge_in.embedding_model,
        config=knowledge_in.config,
        user_id=user_id  # 添加用户ID
    )
    
    # 添加到数据库
    db.add(db_knowledge)
    db.commit()
    db.refresh(db_knowledge)
    
    return db_knowledge


def update_knowledge(db: Session, knowledge: Knowledge, knowledge_in: KnowledgeUpdate) -> Knowledge:
    """
    更新知识库信息
    
    参数:
        db (Session): 数据库会话
        knowledge (Knowledge): 要更新的知识库
        knowledge_in (KnowledgeUpdate): 知识库更新模式
    
    返回:
        Knowledge: 更新后的知识库
    """
    # 获取更新数据
    update_data = knowledge_in.dict(exclude_unset=True)
    
    # 更新模型属性
    for field, value in update_data.items():
        if hasattr(knowledge, field) and value is not None:
            setattr(knowledge, field, value)
    
    db.add(knowledge)
    db.commit()
    db.refresh(knowledge)
    
    return knowledge


def delete_knowledge(db: Session, knowledge_id: str) -> None:
    """
    删除知识库
    
    参数:
        db (Session): 数据库会话
        knowledge_id (str): 知识库ID
    """
    knowledge = get_knowledge(db, knowledge_id)
    if knowledge:
        db.delete(knowledge)
        db.commit()

def get_file_by_id(db: Session, file_id: str) -> Optional[FileModel]:
    """
    通过ID获取文件
    """
    return db.query(FileModel).filter(FileModel.id == file_id).first()

def get_knowledge_file(db: Session, file_id: str) -> Optional[KnowledgeFile]:
    """
    通过ID获取知识库文件
    
    参数:
        db (Session): 数据库会话
        file_id (str): 文件ID
    
    返回:
        Optional[KnowledgeFile]: 文件对象或None
    """
    return db.query(KnowledgeFile).filter(KnowledgeFile.id == file_id).first()


def get_knowledge_files(
    db: Session, 
    knowledge_id: str,
    skip: int = 0, 
    limit: int = 100,
    status: Optional[str] = None
) -> List[KnowledgeFile]:
    """
    获取知识库的文件列表
    
    参数:
        db (Session): 数据库会话
        knowledge_id (str): 知识库ID
        skip (int): 跳过的记录数
        limit (int): 限制返回的记录数
        status (Optional[str]): 按状态过滤
    
    返回:
        List[KnowledgeFile]: 文件列表
    """
    query = db.query(KnowledgeFile).filter(KnowledgeFile.knowledge_id == knowledge_id)
    
    # 应用过滤条件
    if status:
        query = query.filter(KnowledgeFile.status == status)
    
    # 应用分页和排序
    return query.order_by(KnowledgeFile.created_at.desc()).offset(skip).limit(limit).all()


async def upload_file_to_knowledge(
    db: Session, 
    knowledge_id: str, 
    file: UploadFile,
    background_tasks: BackgroundTasks,
    upload_dir: str = "uploads",
    description: Optional[str] = None,
    user_id: str = None
) -> KnowledgeFile:
    """
    上传文件到知识库，按照新的处理流程异步处理文件
    
    参数:
        db (Session): 数据库会话
        knowledge_id (str): 知识库ID
        file (UploadFile): 上传的文件
        background_tasks (BackgroundTasks): 后台任务对象，用于异步处理文件
        upload_dir (str): 上传目录
        description (Optional[str]): 文件描述
        user_id (str): 创建者用户ID
    
    返回:
        KnowledgeFile: 创建的文件记录
    """
    # 检查知识库是否存在
    knowledge = get_knowledge(db, knowledge_id)
    if not knowledge:
        raise HTTPException(status_code=404, detail="知识库不存在")
    
    # 获取文件信息
    original_filename = file.filename
    file_extension = original_filename.split('.')[-1] if '.' in original_filename else ''
    file_type = file_extension.lower()
    
    filename_uuid = uuid.uuid4().hex
    # 生成唯一文件名
    unique_filename = f"{filename_uuid}.{file_extension}"
    
    # 创建上传目录
    knowledge_dir = os.path.join(upload_dir,"raw", knowledge_id)
    os.makedirs(knowledge_dir, exist_ok=True)
    
    # 文件路径
    file_path = os.path.join(knowledge_dir, unique_filename)
    
    # 创建额外数据字段，存储描述信息
    extra_data = {}
    if description:
        extra_data["description"] = description
    
    # 创建文件记录，初始状态为上传中
    db_file = KnowledgeFile(
        knowledge_id=knowledge_id,
        filename=filename_uuid,  # 存储的文件名为UUID
        original_filename=original_filename,  # 保存原始文件名为显示名称
        file_type=file_type,
        path="/uploads",
        status="uploading",  # 设置初始状态为上传中
        embedding_status="pending",  # 设置向量化状态为等待处理
        extra_data=extra_data,  # 添加额外数据，包含描述
        user_id=user_id  # 添加用户ID
    )
    
    # 添加到数据库
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    
    try:
        # 保存文件
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            file_size = len(content)
        
        # 更新文件大小和状态为上传完成
        db_file.file_size = file_size
        db_file.status = "uploaded"  # 上传完成
        db.add(db_file)
        
        # 更新知识库文件计数和大小
        knowledge.file_count += 1
        knowledge.total_size += file_size
        db.add(knowledge)
        
        db.commit()
        db.refresh(db_file)
        
        # 添加后台任务处理文件内容提取和向量化
        background_tasks.add_task(
            process_knowledge_file,
            db_file.id,
            file_path,
            file_type,
            knowledge_id,
            filename_uuid
        )
        
        # 检查向量存储是否可用
        if HAS_PYMILVUS:
            # 添加后台任务，在文件处理完成后自动触发向量化
            from app.api.v1.endpoints.knowledge import trigger_file_embedding_after_processing
            background_tasks.add_task(
                trigger_file_embedding_after_processing,
                knowledge_id,
                db_file.id
            )
        else:
            # 记录一个警告，但允许文件上传继续
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"pymilvus 未安装，文件 {db_file.id} 不会自动触发向量化。请安装 pymilvus 包。")
            
            # 更新文件记录，添加提示信息
            db_file.embedding_status = "failed"
            db_file.error = "向量存储组件未安装 (pymilvus)，无法进行向量化"
            db.add(db_file)
            db.commit()
        
        return db_file

    except Exception as e:
        # 如果上传过程中出错，更新状态为失败
        db_file.status = "failed"
        db_file.error = f"文件上传失败: {str(e)}"
        db.add(db_file)
        db.commit()
        
        # 重新抛出异常，让上层处理
        raise e


async def process_knowledge_file(file_id: str,  file_type: str, knowledge_id: str, filename_uuid:str) -> None:
    """
    异步处理知识库文件，包含文本提取、分段和向量化准备
    
    参数:
        file_id (str): 文件记录ID
        file_path (str): 文件路径
        file_type (str): 文件类型
        knowledge_id (str): 知识库ID
        filename_uuid (str): 文件UUID
    """
    from app.db.session import SessionLocal
    import datetime
    
    db = SessionLocal()
    
    try:
        # 获取文件记录
        db_file = get_knowledge_file(db, file_id)
        if not db_file:
            print(f"无法找到文件记录: {file_id}")
            return
        
        # 获取知识库设置
        knowledge = get_knowledge(db, knowledge_id)
        if not knowledge:
            db_file.status = "failed"
            db_file.error = "找不到相关知识库"
            db.add(db_file)
            db.commit()
            return
        
        # 1. 文本提取阶段
        try:
            # 更新状态为解析中
            # db_file.status = "parsing"
            # db.add(db_file)
            # db.commit()
            
            # 提取文本内容
            # text_content = await extract_text_from_file(file_path, file_type)
            file = get_file_by_id(db=db, file_id=filename_uuid)
            text_content = file.text_content
            # # 创建简单的元数据
            # metadata = {
            #     "file_id": file_id,
            #     "file_name": db_file.original_filename,
            #     "file_type": file_type,
            #     "knowledge_id": knowledge_id,
            #     "extraction_time": datetime.datetime.now().isoformat()
            # }
            
            # # 检查文本是否提取成功
            # if not text_content or (isinstance(text_content, str) and text_content.startswith("提取") and "出错" in text_content):
            #     db_file.status = "failed"
            #     db_file.error = text_content if text_content else "文本提取失败，未能获取内容"
            #     db.add(db_file)
            #     db.commit()
            #     return
                
            # 将提取的文本保存到数据库
            db_file.text_content = text_content
            # db_file.text_extraction_time = datetime.datetime.now()
            # db_file.extra_data = metadata
            # db_file.status = "parsed"  # 更新状态为解析完成
            # db.add(db_file)
            # db.commit()
        
            print(f"文件 {file_id} 的文本内容已提取并保存到数据库，文本长度: {len(text_content)}")
        
            # 2. 文本分段阶段
            db_file.status = "chunking"  # 更新状态为分段中
            db.add(db_file)
            db.commit()
            
            # 获取知识库的分段配置
            chunking_config = knowledge.config.get("chunking_config", {})
            if not chunking_config:
                chunking_config = {
                    "method": "recursive",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "separators": ["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
                }
            
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
                db_file.text_chunking_result = text_chunks
                db_file.chunking_time = datetime.datetime.now()
                db_file.chunk_count = len(text_chunks)
                db_file.status = "chunked"  # 更新状态为分段完成
                
                # 如果还需要向量化，设置状态为indexed以允许后续处理
                if knowledge.embedding_model:
                    db_file.status = "indexed"
                
                    db.add(db_file)
                    db.commit()
                
                print(f"文件 {file_id} 文本分段完成，共 {len(text_chunks)} 个分段")
        
                rs = await process_text_chunks(db=db, text_chunks=text_chunks, knowledge_id=knowledge_id, file_id=file_id)
                
                print(f"文件 {file_id} 向量化完成,{rs}")
        
            except Exception as chunk_error:
                db_file.status = "failed"
                db_file.error = f"文本分段失败: {str(chunk_error)}"
                db.add(db_file)
                db.commit()
                print(f"文件 {file_id} 分段失败: {str(chunk_error)}")
                return
                
        except Exception as e:
            db_file.status = "failed"
            db_file.error = f"提取文本内容出错: {str(e)}"
            db.add(db_file)
            db.commit()
            print(f"提取文本内容出错: {str(e)}")
            return
        
    except Exception as e:
        # 处理失败，记录错误
        try:
            db_file = get_knowledge_file(db, file_id)
            if db_file:
                db_file.status = "failed"
                db_file.error = str(e)
                db.add(db_file)
                db.commit()
        except Exception as inner_error:
            # 内部错误记录日志
            print(f"更新文件状态失败: {str(inner_error)}")
    
    finally:
        db.close()


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


def delete_knowledge_file(db: Session, file_id: str) -> None:
    """
    删除知识库文件
    
    参数:
        db (Session): 数据库会话
        file_id (str): 文件ID
    """
    file = get_knowledge_file(db, file_id)
    if file:
        # 获取知识库
        knowledge = get_knowledge(db, file.knowledge_id)
        
        # 从文件系统删除文件
        if os.path.exists(file.path):
            os.remove(file.path)
        
        # 更新知识库文件计数和大小
        if knowledge:
            knowledge.file_count = max(0, knowledge.file_count - 1)
            knowledge.total_size = max(0, knowledge.total_size - file.file_size)
            db.add(knowledge)
        
        # 从数据库删除记录
        db.delete(file)
        db.commit() 
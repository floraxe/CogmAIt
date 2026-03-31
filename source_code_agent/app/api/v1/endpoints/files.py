import os
import uuid
import shutil
import asyncio
import tempfile
import mimetypes
import logging
from datetime import datetime
from typing import List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File as FormFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
import concurrent.futures
import io
from pydantic import BaseModel
import urllib.parse

from app.utils.deps import get_db, get_current_active_user
from app.models.user import User
from app.models.file import File as FileModel
from app.utils.file_processor import process_file
from app.schemas.file import FileCreate, FileResponse, FileUpdate, FileListResponse, ProcessFilesRequest
from app.core.minio_client import (
    upload_file_minIO, upload_file_stream, get_file_url, get_file_stream, upload_file_object,
    RAW_BUCKET, PROCESSED_BUCKET, IMAGE_BUCKET, delete_file, download_file, list_files, client
)
from app.db.session import SessionLocal

# 设置日志
logger = logging.getLogger(__name__)

router = APIRouter()

# 添加请求体模型
class ProcessFilesRequest(BaseModel):
    file_ids: List[str]
    vision_model_id: Optional[str] = None

@router.post("/upload", response_model=FileResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = FormFile(...),
    description: str = Form(None),
    skip_processing: bool = Form(False),  # 新增参数，控制是否跳过处理
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """上传文件并进行处理
    
    参数:
        file: 上传的文件
        description: 文件描述
        skip_processing: 是否跳过处理步骤，仅上传存储
    
    返回:
        处理后的文件信息
    """
    if not file:
        raise HTTPException(status_code=400, detail="没有收到文件")
    
    # 获取文件信息
    original_filename = file.filename
    file_size = 0
    file_extension = os.path.splitext(original_filename)[1][1:].lower()
    file_uuid = uuid.uuid4().hex
    
    # 构建MinIO对象名
    object_name = f"{file_uuid}.{file_extension}"
    
    try:
        # 计算文件大小并上传到MinIO
        file_content = await file.read()
        file_size = len(file_content)
        
        content_type = mimetypes.guess_type(original_filename)[0]
        if not content_type:
            content_type = "application/octet-stream"
        
        # 上传至MinIO - 使用upload_file_stream
        file_stream = io.BytesIO(file_content)
        upload_result = upload_file_stream(
            file_stream=file_stream,
            bucket_name=RAW_BUCKET,
            object_name=object_name,
            content_type=content_type,
            file_size=file_size
        )
        
        if not upload_result:
            raise HTTPException(status_code=500, detail="文件上传到MinIO失败")
        
        # 创建文件记录
        db_file = FileModel(
            id=file_uuid,
            filename=file_uuid,
            original_filename=original_filename,
            file_type=file_extension,
            file_size=file_size,
            path=f"{RAW_BUCKET}/{object_name}",  # 保存MinIO路径
            status="uploaded",  # 初始状态为已上传，待处理
            description=description,
            created_by=current_user.username,
            user_id=current_user.id,  # 添加用户ID
            extra_data={"upload_by": current_user.username}
        )
        
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        
        # 只有在不跳过处理时才添加后台任务
        if not skip_processing:
            # 在后台处理文件，不阻塞接口
            background_tasks.add_task(
                process_file_background, 
                file_content=file_content,
                object_name=object_name,
                file_uuid=file_uuid,
                db=db
            )
        
        return db_file.to_dict()
    
    except Exception as e:
        # 如果处理过程中出错，删除MinIO中的文件
        try:
            delete_file(RAW_BUCKET, object_name)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"文件上传或处理失败: {str(e)}")

@router.post("/upload/batch", response_model=List[FileResponse])
async def upload_multiple_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = FormFile(...),
    description: str = Form(None),
    skip_processing: bool = Form(False),  # 新增参数，控制是否跳过处理
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """批量上传多个文件并进行并行处理
    
    参数:
        files: 上传的多个文件
        description: 文件描述（所有文件共用）
        skip_processing: 是否跳过处理步骤，仅上传存储
    
    返回:
        处理后的所有文件信息列表
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="没有收到文件")
    
    results = []
    uploaded_files = []
    
    # 获取当前时间，用于所有文件的创建时间
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%dT%H:%M:%S')
    
    # 处理单文件上传的情况
    if not isinstance(files, list):
        files = [files]
    
    for file in files:
        try:
            # 获取文件信息
            original_filename = file.filename
            file_extension = os.path.splitext(original_filename)[1][1:].lower() if original_filename else "txt"
            file_uuid = uuid.uuid4().hex
            
            # 构建MinIO对象名
            object_name = f"{file_uuid}.{file_extension}"
            
            # 读取文件内容并获取大小
            file_content = await file.read()
            file_size = len(file_content)
            
            # 猜测内容类型
            content_type = mimetypes.guess_type(original_filename)[0]
            if not content_type:
                content_type = "application/octet-stream"
            
            # 上传至MinIO - 使用upload_file_stream替代upload_file_object
            file_stream = io.BytesIO(file_content)
            upload_result = upload_file_stream(
                file_stream=file_stream,
                bucket_name=RAW_BUCKET,
                object_name=object_name,
                content_type=content_type,
                file_size=file_size
            )
            
            if not upload_result:
                print(f"文件 {file.filename} 上传到MinIO失败")
                continue
            
            # 创建文件记录
            db_file = FileModel(
                id=file_uuid,
                filename=file_uuid,
                original_filename=original_filename,
                file_type=file_extension,
                file_size=file_size,
                path=f"{RAW_BUCKET}/{object_name}",  # 保存MinIO路径
                status="uploaded",  # 初始状态为已上传，待处理
                description=description,
                created_by=current_user.username,
                user_id=current_user.id,  # 添加用户ID
                created_at=current_time,
                updated_at=current_time,
                extra_data={"upload_by": current_user.username}
            )
            
            db.add(db_file)
            
            # 只有在不跳过处理时才添加到处理列表
            if not skip_processing:
                uploaded_files.append((file_content, object_name, file_uuid))
            
            # 将对象转换为字典
            file_dict = db_file.to_dict()
            
            # 确保时间字段为字符串格式
            if file_dict.get("created") is None:
                file_dict["created"] = formatted_time
            if file_dict.get("updated") is None:
                file_dict["updated"] = formatted_time
                
            results.append(file_dict)
            
        except Exception as e:
            # 如果处理过程中出错，记录错误
            print(f"文件 {file.filename} 上传失败: {str(e)}")
            continue
    
    # 提交所有文件记录
    db.commit()
    
    # 在后台并行处理所有文件，不阻塞接口
    if not skip_processing and uploaded_files:
        background_tasks.add_task(
            process_multiple_files_background,
            uploaded_files=uploaded_files,
            db=db
        )
    
    return results

def process_multiple_files_background(uploaded_files: List[tuple], db: Session):
    """
    后台并行处理多个文件的任务
    
    参数:
        uploaded_files: 已上传文件的(file_content, object_name, file_uuid)元组列表
        db: 数据库会话 - 注意：这个会话仅用于初始查询，不传递给子线程
    
    说明:
        1. 每个线程使用独立的数据库会话，避免会话共享问题
        2. 使用process_single_file_background_safe包装处理函数，确保异常处理和资源释放
        3. 正确处理数据库事务（提交或回滚）
        4. 处理异常情况下更新文件状态为失败
    """
    print(f"开始并行处理 {len(uploaded_files)} 个文件")
    
    # 不再将db会话传递给子线程，每个线程将创建自己的会话
    file_info_list = [
        (file_content, object_name, file_uuid) 
        for file_content, object_name, file_uuid in uploaded_files
    ]
    
    # 使用线程池并行处理文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(file_info_list))) as executor:
        # 创建任务列表
        futures = [
            executor.submit(
                process_single_file_background_safe,
                file_content=file_content,
                object_name=object_name,
                file_uuid=file_uuid
            )
            for file_content, object_name, file_uuid in file_info_list
        ]
        
        # 等待所有任务完成
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                # 获取任务结果
                result = future.result()
                print(f"文件处理任务 {i+1}/{len(file_info_list)} 完成")
            except Exception as e:
                print(f"文件处理任务 {i+1}/{len(file_info_list)} 失败: {str(e)}")
    
    print("所有文件处理任务完成")

def process_single_file_background_safe(file_content, object_name, file_uuid):
    """
    安全处理单个文件的后台任务，创建新的数据库会话
    
    参数:
        file_content: 文件内容
        object_name: MinIO中的对象名
        file_uuid: 文件UUID
    """
    # 为每个处理任务创建独立的数据库会话
    db = SessionLocal()
    try:
        result = process_single_file_background(file_content, object_name, file_uuid, db)
        return result
    except Exception as e:
        print(f"处理文件 {file_uuid} 时发生异常: {str(e)}")
        # 确保任何异常都会回滚事务
        try:
            db.rollback()
        except:
            pass
        raise
    finally:
        # 始终关闭数据库会话
        db.close()

def process_single_file_background(file_content, object_name, file_uuid, db: Session):
    """
    处理单个文件的后台任务
    
    参数:
        file_content: 文件内容
        object_name: MinIO中的对象名
        file_uuid: 文件UUID
        db: 数据库会话
    """
    logger.info(f"开始处理文件 {file_uuid}")
    
    try:
        # 获取文件记录
        db_file = db.query(FileModel).filter(FileModel.id == file_uuid).first()
        if not db_file:
            logger.error(f"文件记录不存在: {file_uuid}")
            return {"status": "error", "message": "文件记录不存在"}
        
        # 更新状态为处理中
        db_file.status = "processing"
        db.commit()
        logger.info(f"更新文件状态为processing: {file_uuid}")
        
        # 创建临时文件用于处理
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(object_name)[1]) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
            logger.info(f"创建临时文件: {temp_file_path}")
        
        try:
            # 提取文件名(不带扩展名)作为处理目录
            filename_uuid = db_file.filename
            
            # 检查是否有配置的多模态模型ID
            vision_model_id = None
            if db_file.extra_data and "vision_model_id" in db_file.extra_data:
                vision_model_id = db_file.extra_data["vision_model_id"]
                logger.info(f"检测到配置的多模态模型ID: {vision_model_id}")
            
            # 调用文件处理函数
            logger.info(f"调用process_file处理文件: {file_uuid}")
            result = process_file(
                file_path=temp_file_path, 
                output_dir=None,
                filename_uuid=filename_uuid,
                knowledge_id="files",  # 使用files作为目录标识
                vision_model_id=vision_model_id  # 传递多模态模型ID
            )
            
            logger.info(f"文件处理完成，结果: {result}")
            
            # 创建保存处理后文件的目录
            processed_dir = os.path.join("uploads", "processed_backup")
            os.makedirs(processed_dir, exist_ok=True)
            
            # 上传处理结果到MinIO
            if "error" not in result:
                # 存储关联路径
                markdown_minio_path = None
                visual_minio_path = None
                images_minio_folder = None
                
                # 如果生成了Markdown文件，上传到MinIO
                if "markdown_path" in result and os.path.exists(result["markdown_path"]):
                    try:
                        with open(result["markdown_path"], "r", encoding="utf-8") as f:
                            markdown_content = f.read()
                            db_file.text_content = markdown_content
                            
                        # 上传Markdown文件到MinIO
                        markdown_object_name = f"{filename_uuid}.md"
                        logger.info(f"上传Markdown文件到MinIO: {markdown_object_name}")
                        
                        # 确保文件存在并上传
                        if os.path.exists(result["markdown_path"]):
                            # 先保存一个备份
                            backup_md_path = os.path.join(processed_dir, f"{filename_uuid}.md")
                            try:
                                shutil.copy2(result["markdown_path"], backup_md_path)
                                logger.info(f"已保存Markdown文件备份: {backup_md_path}")
                            except Exception as e:
                                logger.error(f"保存Markdown文件备份失败: {str(e)}")
                            
                            # 同步上传文件 - 确保直接调用非异步函数
                            upload_success = upload_file_minIO(
                                local_path=result["markdown_path"],
                                bucket_name=PROCESSED_BUCKET,
                                object_name=markdown_object_name
                            )
                            
                            if upload_success:
                                # 设置MinIO路径
                                markdown_minio_path = f"{PROCESSED_BUCKET}/{markdown_object_name}"
                                db_file.markdown_path = markdown_minio_path
                                logger.info(f"Markdown文件上传成功，路径: {markdown_minio_path}")
                            else:
                                logger.error(f"Markdown文件上传失败: {result['markdown_path']}")
                        else:
                            logger.error(f"Markdown文件不存在: {result['markdown_path']}")
                    except Exception as e:
                        error_msg = f"读取Markdown文件失败: {str(e)}"
                        db_file.error = error_msg
                        logger.error(error_msg)
                
                # 如果有visual_path，上传到MinIO
                if "visual_path" in result and os.path.exists(result["visual_path"]):
                    visual_object_name = f"{filename_uuid}_visual.pdf"
                    logger.info(f"上传可视化PDF到MinIO: {visual_object_name}")
                    
                    # 确保文件存在并上传
                    if os.path.exists(result["visual_path"]):
                        # 先保存一个备份
                        backup_pdf_path = os.path.join(processed_dir, f"{filename_uuid}_visual.pdf")
                        try:
                            shutil.copy2(result["visual_path"], backup_pdf_path)
                            logger.info(f"已保存PDF文件备份: {backup_pdf_path}")
                        except Exception as e:
                            logger.error(f"保存PDF文件备份失败: {str(e)}")
                        
                        # 同步上传文件 - 确保直接调用非异步函数
                        upload_success = upload_file_minIO(
                            local_path=result["visual_path"],
                            bucket_name=PROCESSED_BUCKET,
                            object_name=visual_object_name
                        )
                        
                        if upload_success:
                            # 设置MinIO路径
                            visual_minio_path = f"{PROCESSED_BUCKET}/{visual_object_name}"
                            db_file.visual_path = visual_minio_path
                            logger.info(f"可视化PDF上传成功，路径: {visual_minio_path}")
                            
                            # 更新文件记录的extra_data
                            db_file.extra_data = {
                                **(db_file.extra_data or {}),
                                "visual_path": visual_minio_path
                            }
                            logger.info(f"更新文件记录的extra_data: {db_file.extra_data}")
                        else:
                            logger.error(f"可视化PDF上传失败: {result['visual_path']}")
                    else:
                        logger.error(f"可视化PDF不存在: {result['visual_path']}")
                
                # 上传生成的图片
                if "visual_path" in result:
                    images_dir = os.path.join(os.path.dirname(result["visual_path"]), "images")
                    if os.path.exists(images_dir):
                        images_minio_folder = f"{IMAGE_BUCKET}/{filename_uuid}/images"
                        db_file.images_folder = images_minio_folder
                        
                        # 创建图片备份目录
                        backup_images_dir = os.path.join(processed_dir, f"{filename_uuid}_images")
                        os.makedirs(backup_images_dir, exist_ok=True)
                        
                        image_count = 0
                        image_list = []
                        
                        for img_file in os.listdir(images_dir):
                            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                                img_path = os.path.join(images_dir, img_file)
                                img_object_name = f"{filename_uuid}/images/{img_file}"
                                
                                # 确保文件存在并上传
                                if os.path.exists(img_path):
                                    # 先保存一个备份
                                    backup_img_path = os.path.join(backup_images_dir, img_file)
                                    try:
                                        shutil.copy2(img_path, backup_img_path)
                                    except Exception as e:
                                        logger.error(f"保存图片备份失败: {img_file}, 错误: {str(e)}")
                                    # 同步上传文件 - 确保直接调用非异步函数
                                    upload_success = upload_file_minIO(
                                        local_path=img_path,
                                        bucket_name=IMAGE_BUCKET,
                                        object_name=img_object_name
                                    )
                                    
                                    if upload_success:
                                        image_count += 1
                                        image_list.append(f"{IMAGE_BUCKET}/{img_object_name}")
                                    else:
                                        logger.error(f"图片上传失败: {img_path}")
                                else:
                                    logger.error(f"图片文件不存在: {img_path}")
                        
                        logger.info(f"上传了 {image_count} 个图片到MinIO，备份目录: {backup_images_dir}")
                        
                        # 更新文件记录，添加图片列表
                        db_file.extra_data = {
                            **(db_file.extra_data or {}),
                            "images": image_list
                        }
                
                # 如果有图片分析结果，保存到extra_data
                if "image_analysis" in result:
                    db_file.extra_data = {
                        **(db_file.extra_data or {}),
                        "image_analysis": result["image_analysis"]
                    }
                    logger.info("保存图片分析结果到extra_data")
                
                # 如果有多个图片分析结果，保存到extra_data
                if "image_analyses" in result:
                    db_file.extra_data = {
                        **(db_file.extra_data or {}),
                        "image_analyses": result["image_analyses"]
                    }
                    logger.info(f"保存 {len(result['image_analyses'])} 个图片分析结果到extra_data")
                
                # 更新状态为已处理
                db_file.status = "processed"
                db_file.text_extraction_time = datetime.now()
                logger.info(f"更新文件状态为processed: {file_uuid}")
            else:
                # 处理失败
                db_file.status = "failed"
                db_file.error = result["error"]
                logger.error(f"文件处理失败: {file_uuid}, 错误: {result['error']}")
            
            # 提交数据库更改
            try:
                db.commit()
                logger.info(f"已提交数据库更新: {file_uuid}")
            except Exception as db_error:
                db.rollback()
                logger.error(f"提交数据库更新失败，已回滚: {file_uuid}, 错误: {str(db_error)}")
                # 尝试再次更新状态
                try:
                    db_file = db.query(FileModel).filter(FileModel.id == file_uuid).first()
                    if db_file:
                        db_file.status = "failed"
                        db_file.error = f"数据库更新失败: {str(db_error)}"
                        db.commit()
                except Exception as e:
                    logger.error(f"重试更新状态失败: {str(e)}")
        
        finally:
            # 删除临时文件，但保留处理后的文件
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    logger.info(f"删除临时文件: {temp_file_path}")
            except Exception as e:
                logger.error(f"删除临时文件失败: {temp_file_path}, 错误: {str(e)}")
        
        return {"status": "success", "file_id": file_uuid}
        
    except Exception as e:
        # 处理失败，更新状态
        logger.error(f"处理文件发生异常: {file_uuid}, 错误: {str(e)}")
        try:
            # 回滚任何未提交的更改
            db.rollback()
            
            # 尝试获取文件记录并更新状态
            db_file = db.query(FileModel).filter(FileModel.id == file_uuid).first()
            if db_file:
                db_file.status = "failed"
                db_file.error = str(e)
                db.commit()
                logger.info(f"更新文件状态为failed: {file_uuid}")
        except Exception as db_error:
            logger.error(f"更新文件状态失败: {file_uuid}, 错误: {str(db_error)}")
            # 尝试再次回滚
            try:
                db.rollback()
            except:
                pass
        
        return {"status": "error", "message": str(e), "file_id": file_uuid}

def process_file_background(file_content, object_name, file_uuid, db: Session):
    """
    后台处理文件的任务
    
    参数:
        file_content: 文件内容
        object_name: MinIO中的对象名
        file_uuid: 文件UUID
        db: 数据库会话
    """
    # 使用安全的处理函数，不直接传递db会话
    return process_single_file_background_safe(file_content, object_name, file_uuid)

@router.get("/", response_model=FileListResponse)
async def list_files(
    skip: int = 0,
    limit: int = 10,
    filename: Optional[str] = None,
    file_type: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    获取文件列表
    
    参数:
        skip: 跳过的数量
        limit: 返回的条数
        filename: 文件名过滤
        file_type: 文件类型过滤
        status: 状态过滤
    
    返回:
        文件列表和总数
    """
    query = db.query(FileModel)
    
    # 应用过滤条件
    if filename:
        query = query.filter(FileModel.original_filename.ilike(f"%{filename}%"))
    
    if file_type:
        query = query.filter(FileModel.file_type == file_type)
    
    if status:
        query = query.filter(FileModel.status == status)
    
    # 获取总数
    total = query.count()
    
    # 分页
    files = query.order_by(FileModel.created_at.desc()).offset(skip).limit(limit).all()
    
    # 为每个文件添加访问URL
    file_list = []
    for file in files:
        file_dict = file.to_dict()
        
        # 获取原始文件URL
        try:
            if file.path and '/' in file.path:
                bucket, object_name = file.path.split('/', 1)
                file_url = get_file_url(bucket, object_name)
                if file_url:
                    file_dict["file_url"] = file_url
        except Exception as e:
            print(f"获取文件URL失败: {str(e)}")
        
        file_list.append(file_dict)
    
    return {
        "items": file_list,
        "total": total
    }

@router.get("/{file_id}", response_model=FileResponse)
async def get_file_detail(
    file_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    获取文件详情
    
    参数:
        file_id: 文件ID
    
    返回:
        文件详情
    """
    file = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    file_dict = file.to_dict()
    
    try:
        # 获取原始文件URL
        if file.path and '/' in file.path:
            bucket, object_name = file.path.split('/', 1)
            file_url = get_file_url(bucket, object_name)
            if file_url:
                file_dict["file_url"] = file_url
        
        # 获取Markdown文件URL
        if file.markdown_path and '/' in file.markdown_path:
            bucket, object_name = file.markdown_path.split('/', 1)
            markdown_url = get_file_url(bucket, object_name)
            if markdown_url:
                file_dict["markdown_url"] = markdown_url
        
        # 获取处理后的可视化文件URL
        if file.visual_path and '/' in file.visual_path:
            bucket, object_name = file.visual_path.split('/', 1)
            visual_url = get_file_url(bucket, object_name)
            if visual_url:
                file_dict["visual_url"] = visual_url
        # 兼容旧格式
        elif file.extra_data and file.extra_data.get("visual_path") and '/' in file.extra_data.get("visual_path"):
            bucket, object_name = file.extra_data.get("visual_path").split('/', 1)
            visual_url = get_file_url(bucket, object_name)
            if visual_url:
                file_dict["visual_url"] = visual_url
    except Exception as e:
        logger.error(f"获取文件URL失败: {str(e)}")
    
    return file_dict

@router.put("/{file_id}", response_model=FileResponse)
async def update_file(
    file_id: str,
    file_update: FileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    更新文件信息
    
    参数:
        file_id: 文件ID
        file_update: 更新的文件信息
    
    返回:
        更新后的文件信息
    """
    file = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 更新字段
    if file_update.original_filename is not None:
        file.original_filename = file_update.original_filename
    
    if file_update.description is not None:
        file.description = file_update.description
    
    db.commit()
    db.refresh(file)
    
    return file.to_dict()

@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    删除文件
    
    参数:
        file_id: 文件ID
    
    返回:
        删除结果
    """
    file = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    try:
        # 从MinIO删除原始文件
        if file.path and '/' in file.path:
            bucket, object_name = file.path.split('/', 1)
            delete_file(bucket, object_name)
        
        # 删除处理后的文件
        filename_uuid = file.filename
        
        # 删除处理后的Markdown文件
        delete_file(PROCESSED_BUCKET, f"{filename_uuid}.md")
        
        # 删除处理后的PDF文件
        delete_file(PROCESSED_BUCKET, f"{filename_uuid}_visual.pdf")
        
        # 删除处理后的图片文件（使用前缀删除所有相关图片）
        # 这里需要实现遍历删除或使用MinIO的批量删除功能
        
        # 从数据库中删除记录
        db.delete(file)
        db.commit()
        
        return {"message": "文件删除成功"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")

@router.post("/{file_id}/reprocess")
async def reprocess_file(
    file_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    重新处理文件
    
    参数:
        file_id: 文件ID
    
    返回:
        重新处理结果
    """
    file = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 获取MinIO的存储路径
    if not file.path or '/' not in file.path:
        raise HTTPException(status_code=400, detail="原始文件路径信息不正确，无法重新处理")
    
    try:
        # 从MinIO获取文件
        bucket, object_name = file.path.split('/', 1)
        
        # 获取文件数据
        response = get_file_stream(bucket, object_name)
        if not response:
            raise HTTPException(status_code=400, detail="原始文件不存在，无法重新处理")
    
        file_content = response.read()
        
        # 更新文件状态
        file.status = "processing"
        file.error = None
        db.commit()
        
        # 在后台重新处理文件
        background_tasks.add_task(
            process_file_background, 
                file_content=file_content,
                object_name=object_name,
            file_uuid=file.id,
            db=db
        )
        
        return {"message": "文件已加入重新处理队列"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新处理文件失败: {str(e)}")

@router.post("/batch-reprocess")
async def batch_reprocess_files(
    file_ids: List[str],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    批量重新处理多个文件
    
    参数:
        file_ids: 文件ID列表
    
    返回:
        重新处理结果
    """
    if not file_ids:
        raise HTTPException(status_code=400, detail="未提供文件ID列表")
    
    processed_files = []
    failed_files = []
    
    for file_id in file_ids:
        file = db.query(FileModel).filter(FileModel.id == file_id).first()
        
        if not file:
            failed_files.append({"id": file_id, "reason": "文件不存在"})
            continue
        
        # 获取MinIO的存储路径
        if not file.path or '/' not in file.path:
            failed_files.append({"id": file_id, "reason": "原始文件路径信息不正确"})
            continue
        
        try:
            # 从MinIO获取文件
            bucket, object_name = file.path.split('/', 1)
            
            # 获取文件数据
            response = get_file_stream(bucket, object_name)
            if not response:
                failed_files.append({"id": file_id, "reason": "原始文件不存在"})
                continue
        
            file_content = response.read()
            
            # 更新文件状态
            file.status = "processing"
            file.error = None
            processed_files.append((file_content, object_name, file.id))
        
        except Exception as e:
            failed_files.append({"id": file_id, "reason": str(e)})
            continue
    
    # 提交所有状态更新
    db.commit()
    
    if processed_files:
        # 在后台并行重新处理文件
        background_tasks.add_task(
            process_multiple_files_background,
            uploaded_files=processed_files,
            db=db
        )
    
    return {
        "message": f"已成功提交 {len(processed_files)} 个文件进行重新处理",
        "processed": len(processed_files),
        "failed": failed_files
    }

@router.put("/{file_id}/content")
async def update_file_content(
    file_id: str,
    content: dict,
    db: Session = Depends(get_db),
    # current_user: User = Depends(get_current_active_user),
):
    """
    更新文件内容
    
    参数:
        file_id: 文件ID
        content: 更新的内容，应包含content字段
    
    返回:
        更新后的文件信息
    """
    file = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    if "content" not in content:
        raise HTTPException(status_code=400, detail="缺少内容字段")
    
    # 更新文本内容
    file.text_content = content["content"]
    
    # 同时更新MinIO中的Markdown文件
    if file.filename:
        markdown_object_name = f"{file.filename}.md"
        upload_file_object(
            file_data=content["content"].encode('utf-8'),
            bucket_name=PROCESSED_BUCKET,
            object_name=markdown_object_name,
            content_type="text/markdown"
        )
    
    db.commit()
    db.refresh(file)
    
    return file.to_dict()

@router.get("/{file_id}/content", response_model=FileResponse)
async def get_file_content(
    file_id: str,
    db: Session = Depends(get_db),
    # current_user: User = Depends(get_current_active_user),
):
    """
    获取文件详情
    
    参数:
        file_id: 文件ID
    
    返回:
        文件详情
    """
    file = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 尝试从MinIO获取Markdown内容
    file_content = None
    if file.filename:
        markdown_object_name = f"{file.filename}.md"
        try:
            response = get_file_stream(PROCESSED_BUCKET, markdown_object_name)
            if response:
                file_content = response.read().decode('utf-8')
                file.text_content = file_content  # 更新数据库中的内容
                db.commit()
        except:
            pass
    
    # 如果MinIO中没有找到，使用数据库中保存的内容
    if not file_content and file.text_content:
        file_content = file.text_content
    
    file_dict = file.to_dict()
    file_dict["content"] = file_content
    
    return file_dict

@router.get("/status/{file_id}", response_model=FileResponse)
async def get_file_status(
    file_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    获取文件处理状态
    
    参数:
        file_id: 文件ID
    
    返回:
        文件详情，包含处理状态
    """
    file = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return file.to_dict()

@router.post("/process")
async def process_files(
    request: ProcessFilesRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    处理已上传但尚未处理的文件
    
    参数:
        request: 包含file_ids列表的请求体
    
    返回:
        处理结果
    """
    file_ids = request.file_ids
    vision_model_id = request.vision_model_id
    
    if not file_ids:
        raise HTTPException(status_code=400, detail="未提供文件ID列表")
    
    print(f"收到处理请求，文件ID: {file_ids}")  # 添加调试日志
    
    # 如果提供了多模态模型ID，验证该模型是否存在且支持视觉
    vision_model = None
    if vision_model_id:
        from app.models.model import Model
        vision_model = db.query(Model).filter(
            Model.id == vision_model_id,
            Model.vision_support == True
        ).first()
        
        if vision_model:
            print(f"使用多模态模型进行图片分析: {vision_model.name} (ID: {vision_model.id})")
        else:
            print(f"找不到指定的多模态模型ID {vision_model_id} 或该模型不支持视觉功能")
    
    processed_files = []
    failed_files = []
    
    for file_id in file_ids:
        file = db.query(FileModel).filter(FileModel.id == file_id).first()
        
        if not file:
            failed_files.append({"id": file_id, "reason": "文件不存在"})
            print(f"文件不存在: {file_id}")  # 添加调试日志
            continue
        
        # 检查文件状态
        if file.status not in ["uploaded", "failed"]:
            failed_files.append({"id": file_id, "reason": f"文件状态为 {file.status}，不需要处理"})
            print(f"文件状态不需要处理: {file_id}, 状态: {file.status}")  # 添加调试日志
            continue
        
        # 获取MinIO的存储路径
        if not file.path or '/' not in file.path:
            failed_files.append({"id": file_id, "reason": "原始文件路径信息不正确"})
            print(f"文件路径不正确: {file_id}, 路径: {file.path}")  # 添加调试日志
            continue
        
        try:
            # 从MinIO获取文件
            bucket, object_name = file.path.split('/', 1)
            print(f"准备从MinIO获取文件: {bucket}/{object_name}")  # 添加调试日志
            
            # 获取文件数据
            response = get_file_stream(bucket, object_name)
            if not response:
                failed_files.append({"id": file_id, "reason": "原始文件不存在"})
                print(f"MinIO中文件不存在: {bucket}/{object_name}")  # 添加调试日志
                continue
            
            file_content = response.read()
            print(f"读取文件内容成功，大小: {len(file_content)} 字节")  # 添加调试日志
            
            # 更新文件状态
            file.status = "processing"
            file.error = None
            
            # 如果有多模态模型，将其ID保存到文件的extra_data中
            if vision_model:
                file.extra_data = {
                    **(file.extra_data or {}),
                    "vision_model_id": vision_model.id,
                    "vision_model_name": vision_model.name,
                    "vision_model_provider": vision_model.provider
                }
            
            processed_files.append((file_content, object_name, file.id))
            print(f"文件已加入处理队列: {file_id}")  # 添加调试日志
        
        except Exception as e:
            failed_files.append({"id": file_id, "reason": str(e)})
            print(f"处理文件时出错: {file_id}, 错误: {str(e)}")  # 添加调试日志
            continue
    
    # 提交所有状态更新
    db.commit()
    
    if processed_files:
        print(f"启动后台处理任务，文件数: {len(processed_files)}")  # 添加调试日志
        # 在后台并行处理文件
        background_tasks.add_task(
            process_multiple_files_background,
            uploaded_files=processed_files,
            db=db
        )
    
    result = {
        "message": f"已成功提交 {len(processed_files)} 个文件进行处理",
        "processed": len(processed_files),
        "failed": failed_files,
        "with_vision_analysis": vision_model is not None
    }
    print(f"处理请求完成，结果: {result}")  # 添加调试日志
    return result

@router.get("/{file_id}/download")
async def download_file(
    file_id: str,
    db: Session = Depends(get_db),
):
    """
    下载原始文件
    
    参数:
        file_id: 文件ID
    
    返回:
        文件流
    """
    file = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 获取MinIO的存储路径
    if not file.path or '/' not in file.path:
        raise HTTPException(status_code=400, detail="原始文件路径信息不正确，无法下载")
    
    try:
        logger.info(f"开始下载文件: file_id={file_id}, path={file.path}")
        
        # 从MinIO获取文件
        bucket, object_name = file.path.split('/', 1)
        
        logger.info(f"解析后的MinIO参数: bucket={bucket}, object_name={object_name}")
        
        # 获取文件流
        response = get_file_stream(bucket, object_name)
        if not response:
            raise HTTPException(status_code=404, detail="文件不存在或访问失败")
        
        # 安全处理文件名 - 使用urllib.parse进行URL编码
        safe_filename = urllib.parse.quote(file.original_filename)
        
        # 设置响应头
        headers = {
            "Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}",
        }
        
        # 根据文件类型设置Content-Type
        content_type = mimetypes.guess_type(file.original_filename)[0]
        if content_type:
            headers["Content-Type"] = content_type
        
        # 返回文件流
        return StreamingResponse(
            response, 
            headers=headers,
            media_type=content_type or "application/octet-stream"
        )
    
    except Exception as e:
        logger.error(f"下载文件失败: file_id={file_id}, error={str(e)}")
        # 返回一般错误，不暴露具体异常信息
        raise HTTPException(status_code=500, detail="文件下载失败，请重试或联系管理员")

@router.get("/minio-download/{bucket}/{object_name:path}")
async def direct_download_from_minio(
    bucket: str,
    object_name: str,
    download: bool = Query(False),
):
    """
    直接从MinIO下载或预览文件
    
    参数:
        bucket: MinIO桶
        object_name: 对象名称（可能包含多个路径部分）
        download: 是否为下载模式，True为下载，False为预览
    
    返回:
        文件流
    """
    try:
        # 打印调试信息
        logger.info(f"尝试从MinIO获取文件: bucket={bucket}, object_name={object_name}, download={download}")
        
        # 从MinIO获取文件
        response = get_file_stream(bucket, object_name)
        if not response:
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 获取原始文件名（取路径最后一部分）
        filename = object_name.split('/')[-1]
        
        # 生成一个简单的安全文件名，避免编码问题
        try:
            # 尝试使用URL编码方式
            safe_filename = urllib.parse.quote(filename)
            
            # 根据文件名猜测Content-Type
            content_type = mimetypes.guess_type(filename)[0]
            
            # 如果是PDF或图片，且是预览模式，设置为inline
            inline_types = ['application/pdf', 'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml']
            
            # 设置响应头
            headers = {}
            
            if download:
                # 下载模式，强制下载
                headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{safe_filename}"
            elif content_type in inline_types:
                # 预览模式且是可预览类型，使用inline模式
                headers["Content-Disposition"] = f"inline; filename*=UTF-8''{safe_filename}"
            else:
                # 无法识别的类型，强制下载
                headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{safe_filename}"
            
            if content_type:
                headers["Content-Type"] = content_type
            
            # 返回文件流
            return StreamingResponse(
                response, 
                headers=headers,
                media_type=content_type or "application/octet-stream"
            )
        except Exception as encoding_error:
            logger.error(f"设置文件名编码失败: {str(encoding_error)}")
            
            # 回退方案：使用简单文件名
            simple_filename = f"file_{bucket}_{object_name.replace('/', '_')}"
            if download:
                headers = {"Content-Disposition": f"attachment; filename=\"{simple_filename}\""}
            else:
                headers = {"Content-Disposition": f"inline; filename=\"{simple_filename}\""}
                
            # 确保PDF文件可以被预览
            if object_name.lower().endswith('.pdf'):
                headers["Content-Type"] = "application/pdf"
                media_type = "application/pdf"
            else:
                media_type = "application/octet-stream"
                
            return StreamingResponse(
                response,
                headers=headers,
                media_type=media_type
            )
    
    except Exception as e:
        logger.error(f"获取MinIO文件失败: bucket={bucket}, object_name={object_name}, error={str(e)}")
        # 返回一般错误，不暴露具体异常信息
        raise HTTPException(status_code=500, detail="文件访问失败，请重试或联系管理员")

@router.get("/{file_id}/preview-pdf")
async def preview_pdf_file(
    file_id: str,
    type: str = Query("processed", description="预览的PDF类型: 'original'为源文件, 'processed'为处理后文件"),
    db: Session = Depends(get_db),
):
    """
    预览PDF文件（源文件或处理后）
    
    参数:
        file_id: 文件ID
        type: 预览类型，original为原始文件，processed为处理后文件
    
    返回:
        PDF文件流，用于在线预览
    """
    file = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    try:
        file_path = None
        
        # 决定使用哪个文件路径
        if type == "original":
            # 使用源文件路径
            if file.path and '/' in file.path:
                file_path = file.path
            else:
                raise HTTPException(status_code=400, detail="源文件路径无效")
            
            # 检查是否为PDF文件
            if file.file_type.lower() != "pdf":
                raise HTTPException(status_code=400, detail="源文件不是PDF格式")
        else:
            # 使用处理后文件路径
            if file.visual_path and '/' in file.visual_path:
                file_path = file.visual_path
            elif file.extra_data and file.extra_data.get("visual_path") and '/' in file.extra_data.get("visual_path"):
                file_path = file.extra_data.get("visual_path")
            else:
                raise HTTPException(status_code=404, detail="处理后的PDF文件不存在")
        
        # 分割bucket和object_name
        if not file_path:
            raise HTTPException(status_code=404, detail="文件路径无效")
            
        logger.info(f"预览PDF文件: file_id={file_id}, type={type}, path={file_path}")
        bucket, object_name = file_path.split('/', 1)
        
        # 获取文件流
        response = get_file_stream(bucket, object_name)
        if not response:
            raise HTTPException(status_code=404, detail="PDF文件不存在或无法访问")
        
        # 安全处理文件名 - 使用ASCII编码，避免编码问题
        try:
            # 生成一个简单安全的文件名，只使用ASCII字符
            safe_filename = f"document_{file_id}.pdf"
            
            # 设置为inline模式用于浏览器预览
            headers = {
                "Content-Disposition": f"inline; filename=\"{safe_filename}\"",
                "Content-Type": "application/pdf",
            }
            
            # 返回文件流
            return StreamingResponse(
                response,
                headers=headers,
                media_type="application/pdf"
            )
        except Exception as e:
            logger.error(f"设置响应头失败: {str(e)}")
            # 尝试不设置Content-Disposition
            return StreamingResponse(
                response,
                media_type="application/pdf"
            )
    
    except Exception as e:
        logger.error(f"预览PDF失败: file_id={file_id}, type={type}, error={str(e)}")
        raise HTTPException(status_code=500, detail="预览PDF失败，请检查文件格式或稍后再试")

@router.get("/image/{file_id}/{image_name:path}")
async def serve_image(
    file_id: str,
    image_name: str,
    db: Session = Depends(get_db),
):
    """
    提供Markdown中引用的图片服务
    
    参数:
        file_id: 文件ID
        image_name: 图片名称
        
    返回:
        图片内容
    """
    try:
        logger.info(f"请求图片: file_id={file_id}, image_name={image_name}")
        
        # 构造MinIO中的图片路径
        object_name = f"{file_id}/images/{image_name}"
        bucket_name = IMAGE_BUCKET
        
        logger.info(f"尝试从MinIO获取图片: bucket={bucket_name}, object_name={object_name}")
        
        # 获取图片内容
        response = get_file_stream(bucket_name, object_name)
        if not response:
            # 尝试查找其他可能的路径
            logger.warning(f"图片未找到，尝试备用路径: bucket={bucket_name}, object_name={object_name}")
            
            # 查询数据库中的文件记录，可能有额外信息
            file = db.query(FileModel).filter(FileModel.id == file_id).first()
            if file and file.extra_data and "images" in file.extra_data:
                # 尝试从extra_data.images中查找匹配的图片路径
                for img_path in file.extra_data["images"]:
                    if image_name in img_path:
                        # 找到匹配的图片路径
                        if '/' in img_path:
                            img_bucket, img_object = img_path.split('/', 1)
                            logger.info(f"从extra_data找到图片: bucket={img_bucket}, object={img_object}")
                            response = get_file_stream(img_bucket, img_object)
                            if response:
                                break
            
            # 如果仍找不到图片
            if not response:
                raise HTTPException(status_code=404, detail="图片不存在")
        
        # 确定图片的Content-Type
        content_type = None
        if image_name.lower().endswith('.jpg') or image_name.lower().endswith('.jpeg'):
            content_type = "image/jpeg"
        elif image_name.lower().endswith('.png'):
            content_type = "image/png"
        elif image_name.lower().endswith('.gif'):
            content_type = "image/gif"
        elif image_name.lower().endswith('.webp'):
            content_type = "image/webp"
        elif image_name.lower().endswith('.svg'):
            content_type = "image/svg+xml"
        else:
            content_type = "application/octet-stream"
        
        # 返回图片内容
        return StreamingResponse(
            response,
            media_type=content_type
        )
    
    except Exception as e:
        logger.error(f"获取图片失败: file_id={file_id}, image_name={image_name}, error={str(e)}")
        raise HTTPException(status_code=500, detail="图片访问失败")

@router.get("/markdown-image/{path:path}")
async def serve_markdown_image(path: str):
    """
    处理Markdown中的图片路径，将相对路径映射到MinIO上的图片
    
    参数:
        path: 图片路径，如 uploads/processed/files/{file_id}/image.jpg
        
    返回:
        图片内容
    """
    try:
        logger.info(f"请求Markdown图片: path={path}")
        
        # 标准化路径（处理Windows和Linux路径差异）
        normalized_path = path.replace('\\', '/')
        
        # 尝试提取文件ID和图片名称
        file_id = None
        image_name = None
        
        # 尝试从路径中提取文件ID
        if "files/" in normalized_path:
            parts = normalized_path.split("files/")
            if len(parts) > 1:
                after_files = parts[1]
                id_parts = after_files.split('/')
                if len(id_parts) > 1:
                    file_id = id_parts[0]
                    image_name = '/'.join(id_parts[1:])
        
        if not file_id or not image_name:
            # 如果无法提取，尝试直接获取路径最后的部分作为图片名
            parts = normalized_path.split('/')
            image_name = parts[-1]
            # 查找可能的文件ID（通常是UUID格式）
            for part in parts:
                if len(part) >= 32 and all(c in '0123456789abcdefABCDEF-' for c in part):
                    file_id = part
                    break
        
        if not file_id:
            raise HTTPException(status_code=400, detail="无法确定图片所属的文件ID")
        
        logger.info(f"解析Markdown图片路径: file_id={file_id}, image_name={image_name}")
        
        # 构造MinIO中的图片路径
        object_name = f"{file_id}/images/{image_name}"
        bucket_name = IMAGE_BUCKET
        
        logger.info(f"尝试从MinIO获取图片: bucket={bucket_name}, object_name={object_name}")
        
        # 获取图片内容
        response = get_file_stream(bucket_name, object_name)
        if not response:
            # 尝试其他可能的路径格式
            alternative_paths = [
                f"{file_id}/{image_name}",  # 直接使用文件ID和图片名
                f"images/{file_id}/{image_name}",  # images/文件ID/图片名
                image_name  # 只使用图片名
            ]
            
            for alt_path in alternative_paths:
                logger.info(f"尝试备用路径: {alt_path}")
                response = get_file_stream(bucket_name, alt_path)
                if response:
                    break
            
            # 如果仍找不到图片
            if not response:
                raise HTTPException(status_code=404, detail="图片不存在")
        
        # 确定图片的Content-Type
        content_type = None
        if image_name.lower().endswith('.jpg') or image_name.lower().endswith('.jpeg'):
            content_type = "image/jpeg"
        elif image_name.lower().endswith('.png'):
            content_type = "image/png"
        elif image_name.lower().endswith('.gif'):
            content_type = "image/gif"
        elif image_name.lower().endswith('.webp'):
            content_type = "image/webp"
        elif image_name.lower().endswith('.svg'):
            content_type = "image/svg+xml"
        else:
            content_type = "application/octet-stream"
        
        # 返回图片内容
        return StreamingResponse(
            response,
            media_type=content_type
        )
    
    except Exception as e:
        logger.error(f"获取Markdown图片失败: path={path}, error={str(e)}")
        raise HTTPException(status_code=500, detail="图片访问失败")

@router.get("/img/{path:path}")
async def serve_image_smart(
    path: str,
    db: Session = Depends(get_db),
):
    """
    智能图片服务API，能够处理多种不同格式的图片路径
    
    参数:
        path: 图片路径，可以是多种格式
        
    返回:
        图片内容
    """
    try:
        logger.info(f"智能图片服务请求: path={path}")
        
        # 标准化路径
        normalized_path = path.replace('\\', '/')
        
        # 提取可能的文件ID和图片名
        file_id = None
        image_name = None
        
        # 检查是否已经包含了bucket/path格式
        if '/' in normalized_path and normalized_path.split('/')[0] in [RAW_BUCKET, PROCESSED_BUCKET, IMAGE_BUCKET]:
            # 直接尝试作为bucket/object格式访问
            parts = normalized_path.split('/', 1)
            bucket = parts[0]
            object_name = parts[1]
            response = get_file_stream(bucket, object_name)
            if response:
                # 确定Content-Type
                content_type = guess_content_type_from_path(object_name)
                return StreamingResponse(response, media_type=content_type)
            
        # 尝试查找文件ID
        parts = normalized_path.split('/')
        for part in parts:
            if len(part) >= 32 and (all(c in '0123456789abcdef' for c in part) or all(c in '0123456789abcdefABCDEF-' for c in part)):
                file_id = part
                # 找到ID后的部分作为图片名
                idx = parts.index(part)
                image_name = '/'.join(parts[idx+1:]) if idx+1 < len(parts) else parts[-1]
                break
        
        # 如果没有找到图片名，使用最后一部分
        if not image_name and len(parts) > 0:
            image_name = parts[-1]
            
        logger.info(f"解析结果: file_id={file_id}, image_name={image_name}")
            
        # 如果没找到文件ID，尝试在数据库中查找文件名对应的文件
        if not file_id and image_name:
            files = db.query(FileModel).all()
            for file in files:
                if file.extra_data and "images" in file.extra_data:
                    for img_path in file.extra_data["images"]:
                        if image_name in img_path:
                            file_id = file.id
                            break
                    if file_id:
                        break
                        
        # 尝试获取图片
        if file_id and image_name:
            # 首先尝试image-files存储桶
            object_paths = [
                f"{file_id}/images/{image_name}",  # 标准路径
                f"{file_id}/{image_name}",         # 直接在文件ID目录
                f"images/{file_id}/{image_name}"   # images/文件ID目录
            ]
            
            for obj_path in object_paths:
                response = get_file_stream(IMAGE_BUCKET, obj_path)
                if response:
                    content_type = guess_content_type_from_path(image_name)
                    return StreamingResponse(response, media_type=content_type)
            
            # 在数据库中查找更多信息
            file = db.query(FileModel).filter(FileModel.id == file_id).first()
            if file and file.extra_data and "images" in file.extra_data:
                for img_path in file.extra_data["images"]:
                    if image_name in img_path and '/' in img_path:
                        img_bucket, img_object = img_path.split('/', 1)
                        response = get_file_stream(img_bucket, img_object)
                        if response:
                            content_type = guess_content_type_from_path(img_object)
                            return StreamingResponse(response, media_type=content_type)
        
        # 如果只有图片名，尝试在几个常见位置查找
        if image_name:
            # 尝试查找同名文件
            for bucket in [IMAGE_BUCKET, PROCESSED_BUCKET, RAW_BUCKET]:
                # 直接尝试文件名
                response = get_file_stream(bucket, image_name)
                if response:
                    content_type = guess_content_type_from_path(image_name)
                    return StreamingResponse(response, media_type=content_type)
                    
                # 尝试images/文件名
                response = get_file_stream(bucket, f"images/{image_name}")
                if response:
                    content_type = guess_content_type_from_path(image_name)
                    return StreamingResponse(response, media_type=content_type)
        
        # 如果都找不到
        raise HTTPException(status_code=404, detail="找不到请求的图片")
    
    except Exception as e:
        logger.error(f"智能图片服务失败: path={path}, error={str(e)}")
        raise HTTPException(status_code=500, detail="图片服务错误，请稍后再试")

def guess_content_type_from_path(path: str) -> str:
    """根据路径猜测内容类型"""
    if path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
        return "image/jpeg"
    elif path.lower().endswith('.png'):
        return "image/png"
    elif path.lower().endswith('.gif'):
        return "image/gif"
    elif path.lower().endswith('.webp'):
        return "image/webp"
    elif path.lower().endswith('.svg'):
        return "image/svg+xml"
    elif path.lower().endswith('.pdf'):
        return "application/pdf"
    else:
        return "application/octet-stream"

@router.post("/test-minio-connection")
async def test_minio_connection(
    current_user: User = Depends(get_current_active_user),
):
    """
    测试MinIO连接
    
    返回:
        连接状态信息
    """
    try:
        # 测试MinIO连接
        buckets = client.list_buckets()
        bucket_names = [bucket.name for bucket in buckets]
        
        # 检查必要的存储桶是否存在
        required_buckets = [RAW_BUCKET, PROCESSED_BUCKET, IMAGE_BUCKET]
        missing_buckets = [bucket for bucket in required_buckets if bucket not in bucket_names]
        
        if missing_buckets:
            return {
                "status": "warning",
                "message": f"MinIO连接成功，但缺少以下存储桶: {', '.join(missing_buckets)}",
                "buckets": bucket_names
            }
        
        return {
            "status": "success",
            "message": "MinIO连接成功",
            "buckets": bucket_names
        }
    except Exception as e:
        logger.error(f"测试MinIO连接失败: {str(e)}")
        return {
            "status": "error",
            "message": f"MinIO连接失败: {str(e)}"
        }

@router.get("/verify-file-access/{file_id}")
async def verify_file_access(
    file_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    验证文件访问和下载链接
    
    参数:
        file_id: 文件ID
        
    返回:
        文件访问信息
    """
    try:
        # 获取文件记录
        file = db.query(FileModel).filter(FileModel.id == file_id).first()
        
        if not file:
            raise HTTPException(status_code=404, detail="文件不存在")
            
        result = {
            "id": file.id,
            "filename": file.original_filename,
            "path": file.path,
            "links": {}
        }
        
        # 检查原始文件访问
        if file.path and '/' in file.path:
            bucket, object_name = file.path.split('/', 1)
            
            try:
                # 检查文件是否存在
                stat = client.stat_object(bucket, object_name)
                result["raw_file_exists"] = True
                result["raw_file_info"] = {
                    "size": stat.size,
                    "last_modified": stat.last_modified.isoformat()
                }
                
                # 生成下载链接
                url = get_file_url(bucket, object_name)
                if url:
                    result["links"]["raw_file"] = url
            except Exception as e:
                result["raw_file_exists"] = False
                result["raw_file_error"] = str(e)
        
        # 检查处理后的文件访问
        if file.visual_path and '/' in file.visual_path:
            try:
                bucket, object_name = file.visual_path.split('/', 1)
                
                # 检查文件是否存在
                stat = client.stat_object(bucket, object_name)
                result["visual_file_exists"] = True
                result["visual_file_info"] = {
                    "size": stat.size,
                    "last_modified": stat.last_modified.isoformat()
                }
                
                # 生成下载链接
                url = get_file_url(bucket, object_name)
                if url:
                    result["links"]["visual_file"] = url
            except Exception as e:
                result["visual_file_exists"] = False
                result["visual_file_error"] = str(e)
        elif file.extra_data and file.extra_data.get("visual_path") and '/' in file.extra_data.get("visual_path"):
            try:
                bucket, object_name = file.extra_data.get("visual_path").split('/', 1)
                
                # 检查文件是否存在
                stat = client.stat_object(bucket, object_name)
                result["visual_file_exists"] = True
                result["visual_file_info"] = {
                    "size": stat.size,
                    "last_modified": stat.last_modified.isoformat()
                }
                
                # 生成下载链接
                url = get_file_url(bucket, object_name)
                if url:
                    result["links"]["visual_file"] = url
            except Exception as e:
                result["visual_file_exists"] = False
                result["visual_file_error"] = str(e)
                
        return result
    except Exception as e:
        logger.error(f"验证文件访问失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"验证文件访问失败: {str(e)}")

@router.get("/{file_id}/preview")
async def preview_file(
    file_id: str,
    db: Session = Depends(get_db),
):
    """
    预览原始文件（在线查看）
    
    参数:
        file_id: 文件ID
    
    返回:
        文件流，用于在线预览
    """
    file = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 获取MinIO的存储路径
    if not file.path or '/' not in file.path:
        raise HTTPException(status_code=400, detail="原始文件路径信息不正确，无法预览")
    
    try:
        # 从MinIO获取文件
        bucket, object_name = file.path.split('/', 1)
        logger.info(f"预览文件: file_id={file_id}, path={file.path}")
        
        # 获取文件数据
        response = get_file_stream(bucket, object_name)
        if not response:
            raise HTTPException(status_code=400, detail="原始文件不存在，无法预览")
        
        # 根据文件类型设置Content-Type
        content_type = mimetypes.guess_type(file.original_filename)[0]
        
        # 如果是PDF或图片，设置为inline而不是attachment，让浏览器直接显示
        inline_types = ['application/pdf', 'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml']
        
        try:
            # 尝试使用安全的文件名
            if file.file_type.lower() == "pdf":
                # PDF文件使用简单名称
                safe_filename = f"document_{file_id}.pdf"
            else:
                # 其他文件尝试URL编码
                safe_filename = urllib.parse.quote(file.original_filename)
            
            # 设置响应头
            headers = {}
            if content_type in inline_types:
                headers["Content-Disposition"] = f"inline; filename=\"{safe_filename}\""
            else:
                # 对于其他类型，使用attachment强制下载
                headers["Content-Disposition"] = f"attachment; filename=\"{safe_filename}\""
            
            if content_type:
                headers["Content-Type"] = content_type
            
            # 返回文件流
            return StreamingResponse(
                response, 
                headers=headers,
                media_type=content_type or "application/octet-stream"
            )
        except Exception as encoding_error:
            logger.error(f"设置文件名编码失败: {str(encoding_error)}")
            
            # 回退方案：不设置Content-Disposition
            return StreamingResponse(
                response, 
                media_type=content_type or "application/octet-stream"
            )
    
    except Exception as e:
        logger.error(f"预览文件失败: file_id={file_id}, error={str(e)}")
        raise HTTPException(status_code=500, detail="文件预览失败，请重试或联系管理员")


import os
import mimetypes
from pathlib import Path
from typing import List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Header
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer
import tempfile

from app.utils.deps import get_db, get_current_active_user
from app.models.knowledge import KnowledgeFile
from app.models.file import File as FileModel
from app.core.config import settings
from app.utils.security import get_user_by_username
from app.core.minio_client import (
    get_file_stream, get_streaming_response, get_file_url,
    RAW_BUCKET, PROCESSED_BUCKET, IMAGE_BUCKET, download_file
)

router = APIRouter()

# 自定义认证依赖，支持URL参数token认证和Header认证
async def get_current_user_from_token_or_header(
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """
    从Header或URL参数中获取当前用户
    优先使用URL参数token，其次使用Header中的Authorization
    """
    # 先尝试URL参数token
    if token:
        try:
            # 直接解码token并验证
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            username = payload.get("sub")
            if username:
                user = get_user_by_username(db, username)
                if user and user.is_active:
                    return user
        except Exception as e:
            print(f"Token参数认证错误: {str(e)}")
    
    # 如果URL参数失败，尝试使用Authorization头
    if authorization:
        try:
            scheme, token = authorization.split()
            if scheme.lower() == "bearer":
                payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
                username = payload.get("sub")
                if username:
                    user = get_user_by_username(db, username)
                    if user and user.is_active:
                        return user
        except Exception as e:
            print(f"Authorization头认证错误: {str(e)}")
    
    # 两种方式都失败，抛出认证异常
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="认证失败",
        headers={"WWW-Authenticate": "Bearer"},
    )

@router.get("/{file_id}/original")
async def preview_original_file(
    file_id: str,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    # current_user: Any = Depends(get_current_user_from_token_or_header),
):
    """
    预览原始上传的文件
    支持通过URL参数token进行认证
    """
    # 从数据库获取文件记录
    file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
        
    if not file_record:
        # 尝试查找知识库文件
        file_record = db.query(KnowledgeFile).filter(KnowledgeFile.id == file_id).first()
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="文件不存在"
            )
        
    # 获取MinIO存储路径
    if hasattr(file_record, 'path') and file_record.path:
        if '/' in file_record.path:
            # 新格式: bucket/object_name
            bucket, object_name = file_record.path.split('/', 1)
        else:
            # 旧格式: 本地路径，需要推断MinIO路径
            file_name = file_record.filename
            file_extension = file_record.file_type
            object_name = f"{file_name}.{file_extension}"
            bucket = RAW_BUCKET
    else:
        # 构建推断的MinIO路径
        file_name = file_record.filename
        file_extension = file_record.file_type
        object_name = f"{file_name}.{file_extension}"
        bucket = RAW_BUCKET

    # 从MinIO获取文件流
    try:
        # 获取文件MIME类型
        content_type, _ = mimetypes.guess_type(object_name)
        if not content_type:
            content_type = "application/octet-stream"
        
        # 获取流式响应
        return get_streaming_response(
            bucket_name=bucket,
            object_name=object_name,
            media_type=content_type,
            filename=file_record.original_filename
        )
    except Exception as e:
        print(f"获取文件失败: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文件不存在或访问失败"
    )

@router.get("/{file_id}/processed")
async def preview_processed_file(
    file_id: str,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user_from_token_or_header),
):
    """
    预览处理后的文件(visual PDF)
    支持通过URL参数token进行认证
    """
    # 从数据库获取文件记录
    file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
        
    if not file_record:
        # 尝试查找知识库文件
        file_record = db.query(KnowledgeFile).filter(KnowledgeFile.id == file_id).first()
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="文件不存在"
            )
        
    # 获取文件名
        file_name = file_record.filename
    
    # 处理后的文件对象名
    visual_object_name = f"{file_name}_visual.pdf"
    
    # 尝试从extra_data中获取处理后文件的路径
    if hasattr(file_record, 'extra_data') and file_record.extra_data and file_record.extra_data.get("visual_path"):
        visual_path = file_record.extra_data.get("visual_path")
        if '/' in visual_path:
            bucket, visual_object_name = visual_path.split('/', 1)
        else:
            bucket = PROCESSED_BUCKET
    else:
        bucket = PROCESSED_BUCKET
    
    try:
        # 获取流式响应
        return get_streaming_response(
            bucket_name=bucket,
            object_name=visual_object_name,
            media_type="application/pdf",
            filename=f"{file_record.original_filename}_processed.pdf"
        )
    except Exception as e:
        print(f"获取处理后的文件失败: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="处理后的文件不存在或访问失败"
    )

@router.get("/{file_id}/images")
async def list_processed_images(
    file_id: str,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user_from_token_or_header),
):
    """
    获取处理后的图片列表
    支持通过URL参数token进行认证
    """
    # 从数据库获取文件记录
    file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在"
        )
    
    # 获取文件名
    file_name = file_record.filename
    
    # 构建图片前缀
    prefix = f"{file_name}/images/"
    
    try:
        # 列出MinIO中的图片
        from app.core.minio_client import client
        
        # 确保客户端已初始化
        if not client:
            return {"images": [], "extra_data": file_record.extra_data or {}}
        
        # 列出图片文件
        images = []
        objects = client.list_objects(IMAGE_BUCKET, prefix=prefix, recursive=True)
        
        for obj in objects:
            object_name = obj.object_name
            # 只包含图片文件
            if object_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                image_name = os.path.basename(object_name)
                # 生成图片URL - 使用uploads路径格式
                image_url = f"/uploads/processd/files/{file_id}/{image_name}"
                if token:
                    image_url += f"?token={token}"
                
                # 查找该图片是否有对应的分析结果
                analysis_result = None
                if file_record.extra_data and 'image_analyses' in file_record.extra_data:
                    for analysis in file_record.extra_data['image_analyses']:
                        if 'image_name' in analysis and analysis['image_name'] == image_name:
                            analysis_result = analysis
                            break
                
                images.append({
                    "name": image_name,
                    "path": image_url,
                    "size": obj.size,
                    "analysis": analysis_result
                })
        
        # 提取extra_data中的图片分析信息
        extra_data = {}
        if file_record.extra_data:
            # 仅提取与图片相关的数据
            for key in ['images', 'image_analysis', 'image_analyses', 
                        'vision_model_id', 'vision_model_name', 'vision_model_provider']:
                if key in file_record.extra_data:
                    extra_data[key] = file_record.extra_data[key]
        
        # 检查extra_data中记录的images是否有未在MinIO中找到的
        if 'images' in extra_data and isinstance(extra_data['images'], list):
            for image_path in extra_data['images']:
                # 尝试解析图片名称
                image_name = image_path.split('/')[-1] if '/' in image_path else image_path
                
                # 检查是否已在images列表中
                if not any(img['name'] == image_name for img in images):
                    # 构建正确的图片URL
                    if 'image-files' in image_path:
                        # 转换为uploads路径格式
                        image_url = f"/uploads/processd/files/{file_id}/{image_name}"
                    else:
                        # 使用uploads路径格式
                        image_url = f"/uploads/processd/files/{file_id}/{image_name}"
                    
                    if token:
                        image_url += f"?token={token}"
                    
                    # 查找该图片是否有对应的分析结果
                    analysis_result = None
                    if 'image_analyses' in extra_data:
                        for analysis in extra_data['image_analyses']:
                            if 'image_name' in analysis and analysis['image_name'] == image_name:
                                analysis_result = analysis
                                break
                    
                    # 添加到images列表
                    images.append({
                        "name": image_name,
                        "path": image_url,
                        "size": 0,  # 大小信息不可用
                        "analysis": analysis_result
                    })
        
        return {
            "images": images,
            "extra_data": extra_data
        }
    except Exception as e:
        print(f"列出图片失败: {str(e)}")
        # 即使出错也返回extra_data
        return {
            "images": [],
            "extra_data": file_record.extra_data or {}
        }

@router.get("/{file_id}/image/{image_name}")
async def preview_processed_image(
    file_id: str,
    image_name: str,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    # current_user: Any = Depends(get_current_user_from_token_or_header),
):
    """
    预览处理后的单个图片
    支持通过URL参数token进行认证
    """
    # 从数据库获取文件记录
    file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在"
        )
    
    # 获取文件名
    file_name = file_record.filename
    
    # 构建图片对象名
    object_name = f"{file_name}/images/{image_name}"
    
    try:
        # 获取图片MIME类型
        content_type, _ = mimetypes.guess_type(image_name)
        if not content_type:
            content_type = "image/jpeg"  # 默认为JPEG
        
        # 获取流式响应
        return get_streaming_response(
            bucket_name=IMAGE_BUCKET,
            object_name=object_name,
            media_type=content_type
        )
    except Exception as e:
        print(f"获取图片失败: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="图片不存在或访问失败"
    )

@router.get("/{file_id}/markdown")
async def preview_markdown(
    file_id: str,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    预览Markdown文件
    """
    # 从数据库获取文件记录
    file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在"
        )
    
    # 获取MinIO存储路径
    if file_record.markdown_path and '/' in file_record.markdown_path:
        # 新格式: bucket/object_name
        bucket, object_name = file_record.markdown_path.split('/', 1)
    else:
        # 旧格式: 构建推断路径
        file_name = file_record.filename
        object_name = f"{file_name}.md"
        bucket = PROCESSED_BUCKET

    # 从MinIO获取文件流
    try:
        # 获取流式响应
        return get_streaming_response(
            bucket_name=bucket,
            object_name=object_name,
            media_type="text/markdown",
            filename=f"{file_record.original_filename}.md"
        )
    except Exception as e:
        logger.error(f"获取Markdown文件失败: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Markdown文件不存在或访问失败"
        )

@router.get("/{file_id}/visual")
async def preview_visual(
    file_id: str,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    预览可视化PDF文件
    """
    # 从数据库获取文件记录
    file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在"
        )
    
    # 获取MinIO存储路径
    if file_record.visual_path and '/' in file_record.visual_path:
        # 新格式: bucket/object_name
        bucket, object_name = file_record.visual_path.split('/', 1)
    elif file_record.extra_data and file_record.extra_data.get("visual_path") and '/' in file_record.extra_data.get("visual_path"):
        # 旧格式从extra_data: bucket/object_name
        bucket, object_name = file_record.extra_data.get("visual_path").split('/', 1)
    else:
        # 旧格式: 构建推断路径
        file_name = file_record.filename
        object_name = f"{file_name}_visual.pdf"
        bucket = PROCESSED_BUCKET

    # 从MinIO获取文件流
    try:
        # 获取流式响应
        return get_streaming_response(
            bucket_name=bucket,
            object_name=object_name,
            media_type="application/pdf",
            filename=f"{file_record.original_filename}_processed.pdf"
        )
    except Exception as e:
        logger.error(f"获取可视化文件失败: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"可视化文件不存在或访问失败"
        )

@router.get("/{file_id}/image/{image_name}")
async def preview_image(
    file_id: str,
    image_name: str,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    预览图片文件
    """
    # 从数据库获取文件记录
    file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在"
        )
    
    # 构建MinIO对象路径
    file_name = file_record.filename
    object_name = f"{file_name}/images/{image_name}"
    bucket = IMAGE_BUCKET

    # 从MinIO获取文件流
    try:
        # 获取文件MIME类型
        content_type, _ = mimetypes.guess_type(image_name)
        if not content_type:
            content_type = "image/jpeg"  # 默认图片类型
        
        # 获取流式响应
        return get_streaming_response(
            bucket_name=bucket,
            object_name=object_name,
            media_type=content_type,
            filename=image_name
        )
    except Exception as e:
        logger.error(f"获取图片文件失败: {str(e)}")
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
            detail=f"图片文件不存在或访问失败"
    )

@router.get("/uploads/{path:path}")
async def preview_uploads_file(
    path: str,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user_from_token_or_header),
):
    """
    从uploads目录预览文件（重定向到MinIO）
    支持通过URL参数token进行认证
    """
    try:
        # 解析路径
        parts = path.split('/')
        if len(parts) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无效的路径格式"
            )
        
        # 检查是否为图片文件
        file_name = parts[-1]
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="仅支持图片文件"
            )
        
        # 从路径中提取文件ID
        # 格式通常是: processd/files/{file_id}/image.jpg
        file_id = None
        for i, part in enumerate(parts):
            if part == 'files' and i + 1 < len(parts):
                file_id = parts[i + 1]
                break
        
        if not file_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无法从路径中提取文件ID"
            )
        
        # 从数据库获取文件记录
        file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="文件不存在"
            )
        
        # 获取文件名
        db_file_name = file_record.filename
        
        # 构建MinIO中的图片对象名
        object_name = f"{db_file_name}/images/{file_name}"
        
        # 获取图片MIME类型
        content_type, _ = mimetypes.guess_type(file_name)
        if not content_type:
            content_type = "image/jpeg"  # 默认为JPEG
        
        # 尝试从MinIO获取图片
        try:
            # 获取流式响应
            return get_streaming_response(
                bucket_name=IMAGE_BUCKET,
                object_name=object_name,
                media_type=content_type
            )
        except Exception as e:
            print(f"从MinIO获取图片失败: {str(e)}")
            
            # 如果MinIO获取失败，尝试从本地文件系统获取
            local_path = os.path.join(settings.UPLOAD_DIR, path)
            if os.path.exists(local_path):
                return FileResponse(
                    path=local_path,
                    media_type=content_type,
                    filename=file_name
                )
            
            # 如果本地文件也不存在，则返回404
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="图片不存在或访问失败"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"处理uploads路径图片时出错: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="处理图片请求时出错"
        )

@router.get("/uploads/files/{file_id}/{image_name}")
async def preview_simple_image(
    file_id: str,
    image_name: str,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user_from_token_or_header),
):
    """
    使用简化路径格式预览图片
    支持通过URL参数token进行认证
    """
    # 从数据库获取文件记录
    file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在"
        )
    
    # 获取文件名
    db_file_name = file_record.filename
    
    # 构建MinIO中的图片对象名
    object_name = f"{db_file_name}/images/{image_name}"
    
    # 获取图片MIME类型
    content_type, _ = mimetypes.guess_type(image_name)
    if not content_type:
        content_type = "image/jpeg"  # 默认为JPEG
    
    # 尝试从MinIO获取图片
    try:
        # 获取流式响应
        return get_streaming_response(
            bucket_name=IMAGE_BUCKET,
            object_name=object_name,
            media_type=content_type
        )
    except Exception as e:
        print(f"从MinIO获取图片失败: {str(e)}")
        
        # 如果MinIO获取失败，尝试从本地文件系统获取
        try:
            # 尝试在uploads目录中查找
            local_path = os.path.join(settings.UPLOAD_DIR, "processd", "files", file_id, image_name)
            if os.path.exists(local_path):
                return FileResponse(
                    path=local_path,
                    media_type=content_type,
                    filename=image_name
                )
        except Exception:
            pass
        
        # 如果本地文件也不存在，则返回404
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="图片不存在或访问失败"
    )
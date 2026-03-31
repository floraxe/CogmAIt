import os
import logging
from minio import Minio
from minio.error import S3Error
from datetime import datetime, timedelta
from fastapi.responses import StreamingResponse
import io
from app.core.config import settings

# 配置日志
logger = logging.getLogger(__name__)

# MinIO配置
MINIO_ENDPOINT = settings.MINIO_ENDPOINT if hasattr(settings, 'MINIO_ENDPOINT') else 'localhost:9000'
ACCESS_KEY = settings.MINIO_ACCESS_KEY if hasattr(settings, 'MINIO_ACCESS_KEY') else 'xkk'
SECRET_KEY = settings.MINIO_SECRET_KEY if hasattr(settings, 'MINIO_SECRET_KEY') else 'xkkxkkxkk'
SECURE = settings.MINIO_SECURE if hasattr(settings, 'MINIO_SECURE') else False

# 文件存储桶配置
RAW_BUCKET = 'raw-files'         # 原始文件存储桶
PROCESSED_BUCKET = 'processed-files'  # 处理后文件存储桶
IMAGE_BUCKET = 'image-files'    # 图片文件存储桶

# 初始化MinIO客户端
client = Minio(
    endpoint=MINIO_ENDPOINT,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=SECURE
)

def ensure_bucket_exists(bucket_name):
    """确保存储桶存在，如不存在则创建"""
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"✅ 创建存储桶 '{bucket_name}' 成功")
        return True
    except S3Error as e:
        logger.error(f"❌ 存储桶操作失败: {e}")
        return False

def upload_file_object(file_data, bucket_name, object_name, content_type=None):
    """上传文件对象到MinIO"""
    try:
        ensure_bucket_exists(bucket_name)
        logger.debug(f"正在上传对象: {bucket_name}/{object_name} (大小: {len(file_data)}字节)")
        client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=file_data,
            length=len(file_data),
            content_type=content_type
        )
        logger.info(f"✅ 文件对象上传成功: {bucket_name}/{object_name}")
        return True
    except S3Error as e:
        logger.error(f"❌ 文件上传失败 {bucket_name}/{object_name}: {e}")
        return False

def upload_file_stream(file_stream, bucket_name, object_name, content_type=None, file_size=None):
    """上传文件流到MinIO"""
    try:
        ensure_bucket_exists(bucket_name)
        # 检查是否为bytes对象
        if isinstance(file_stream, bytes):
            # 如果是bytes，将其转换为BytesIO
            file_stream = io.BytesIO(file_stream)
            if file_size is None:
                file_size = len(file_stream.getvalue())
        
        logger.debug(f"正在上传文件流: {bucket_name}/{object_name} (大小: {file_size}字节)")
        client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=file_stream,
            length=file_size,
            content_type=content_type
        )
        logger.info(f"✅ 文件流上传成功: {bucket_name}/{object_name}")
        return True
    except S3Error as e:
        logger.error(f"❌ 文件流上传失败 {bucket_name}/{object_name}: {e}")
        return False

def upload_file(local_path, bucket_name, object_name):
    """上传本地文件到MinIO"""
    try:
        ensure_bucket_exists(bucket_name)
        logger.debug(f"正在上传本地文件: {local_path} → {bucket_name}/{object_name}")
        
        # 获取内容类型
        content_type = None
        if '.' in local_path:
            import mimetypes
            content_type = mimetypes.guess_type(local_path)[0]
        
        # 上传文件
        client.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=local_path,
            content_type=content_type
        )
        logger.info(f"✅ 文件上传成功: {local_path} → {bucket_name}/{object_name}")
        return True
    except S3Error as e:
        logger.error(f"❌ 文件上传失败 {local_path} → {bucket_name}/{object_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 上传文件时发生未知错误: {str(e)}")
        return False

def upload_file_minIO(local_path, bucket_name, object_name):
    """上传本地文件到MinIO (兼容重命名的函数)"""
    return upload_file(local_path, bucket_name, object_name)

async def upload_file_async(local_path, bucket_name, object_name):
    """异步上传本地文件到MinIO
    
    此函数是upload_file的异步包装器，用于在异步环境中调用
    """
    import asyncio
    
    # 在线程池中执行同步上传
    return await asyncio.to_thread(
        upload_file,
        local_path,
        bucket_name,
        object_name
    )

def download_file(bucket_name, object_name, local_path):
    """从MinIO下载文件到本地"""
    try:
        logger.debug(f"正在下载文件: {bucket_name}/{object_name} → {local_path}")
        client.fget_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=local_path
        )
        logger.info(f"✅ 文件下载成功: {bucket_name}/{object_name} → {local_path}")
        return True
    except S3Error as e:
        logger.error(f"❌ 文件下载失败: {e}")
        return False

def get_file_stream(bucket_name, object_name):
    """从MinIO获取文件流"""
    try:
        logger.debug(f"正在获取文件流: {bucket_name}/{object_name}")
        response = client.get_object(
            bucket_name=bucket_name,
            object_name=object_name
        )
        return response
    except S3Error as e:
        logger.error(f"❌ 获取文件失败: {e}")
        return None

def get_file_url(bucket_name, object_name, expires=3600):
    """生成文件的临时访问URL"""
    try:
        # 修复：将秒数转换为timedelta对象
        expires_delta = timedelta(seconds=expires)
        
        url = client.presigned_get_object(
            bucket_name=bucket_name,
            object_name=object_name,
            expires=expires_delta
        )
        logger.debug(f"生成临时URL: {bucket_name}/{object_name} (过期时间: {expires}秒)")
        return url
    except S3Error as e:
        logger.error(f"❌ 生成URL失败: {e}")
        return None

def delete_file(bucket_name, object_name):
    """从MinIO删除文件"""
    try:
        logger.debug(f"正在删除文件: {bucket_name}/{object_name}")
        client.remove_object(
            bucket_name=bucket_name,
            object_name=object_name
        )
        logger.info(f"✅ 文件删除成功: {bucket_name}/{object_name}")
        return True
    except S3Error as e:
        logger.error(f"❌ 删除文件失败: {e}")
        return False

def get_streaming_response(bucket_name, object_name, media_type=None, filename=None):
    """获取文件的流式响应对象"""
    try:
        logger.debug(f"正在获取流式响应: {bucket_name}/{object_name}")
        response = client.get_object(bucket_name, object_name)
        
        headers = {}
        if filename:
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        
        return StreamingResponse(
            response.stream(32*1024),  # 32KB块大小
            media_type=media_type or response.headers.get('Content-Type', 'application/octet-stream'),
            headers=headers
        )
    except S3Error as e:
        logger.error(f"❌ 获取文件流失败: {e}")
        return None

def list_files(bucket_name, prefix=None):
    """列出存储桶中的文件"""
    try:
        objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
        files = []
        for obj in objects:
            files.append({
                "name": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified
            })
        logger.debug(f"列出文件成功: {bucket_name}/{prefix or ''} (共{len(files)}个文件)")
        return files
    except S3Error as e:
        logger.error(f"❌ 列出文件失败: {e}")
        return []

def initialize_minio():
    """初始化所有必要的MinIO存储桶"""
    logger.info(f"开始初始化MinIO存储桶, 终端: {MINIO_ENDPOINT}")
    
    # 尝试连接MinIO服务器
    try:
        # 检查连接
        buckets = client.list_buckets()
        logger.info(f"已连接到MinIO服务器, 当前存在{len(buckets)}个存储桶")
    except Exception as e:
        logger.error(f"❌ 连接MinIO服务器失败: {e}")
        raise Exception(f"无法连接到MinIO服务器: {e}")
    
    # 初始化所需的存储桶
    ensure_bucket_exists(RAW_BUCKET)
    ensure_bucket_exists(PROCESSED_BUCKET)
    ensure_bucket_exists(IMAGE_BUCKET)
    logger.info("✅ MinIO存储桶初始化完成") 
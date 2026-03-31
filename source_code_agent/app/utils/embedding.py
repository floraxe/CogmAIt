import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import ast
from sqlalchemy.orm import Session
try:
    from pymilvus import MilvusClient
    HAS_PYMILVUS = True
except ImportError:
    HAS_PYMILVUS = False
    logging.warning("pymilvus 模块导入失败，向量存储功能将不可用。请使用 pip install pymilvus 安装。")

# 将 HAS_PYMILVUS 导出为模块变量，确保它可以被其他模块导入
__all__ = ['MilvusVectorStore', 'EmbeddingManager', 'HAS_PYMILVUS']

from app.models.model import Model
from app.utils.model import get_model
from app.providers.manager import provider_manager
from app.utils.chunker import TextChunker, ChunkingMethod
from app.core.config import settings

# 配置日志
logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """
    Milvus向量数据库存储类
    
    用于存储和检索向量数据
    """
    
    def __init__(self, 
                 db_path: Optional[str] = None, 
                 collection_name_prefix: str = "knowledge_"):
        """
        初始化Milvus向量存储
        
        参数:
            db_path (Optional[str]): 数据库路径，默认为None，使用配置中的路径
            collection_name_prefix (str): 集合名称前缀
        """
        # 检查pymilvus是否可用
        if not HAS_PYMILVUS:
            logger.error("pymilvus 模块未安装，向量存储无法初始化。请使用 pip install pymilvus 安装。")
            self.client = None
            self.db_path = None
            self.collection_name_prefix = collection_name_prefix
            return
   
        self.db_path = "http://localhost:19530"
        self.collection_name_prefix = collection_name_prefix
        print("milvus::",db_path)
        try:
            self.client = MilvusClient(self.db_path)
            logger.info(f"初始化Milvus向量存储: {self.db_path}")
        except Exception as e:
            logger.error(f"初始化Milvus向量存储失败: {str(e)}")
            self.client = None
    
    def get_collection_name(self, knowledge_id: str) -> str:
        """
        获取知识库对应的集合名称
        
        参数:
            knowledge_id (str): 知识库ID
            
        返回:
            str: 集合名称
        """
        return f"{self.collection_name_prefix}{knowledge_id}"
    
    def create_collection(self, knowledge_id: str, dimension: int = 384) -> bool:
        """
        创建集合
        
        参数:
            knowledge_id (str): 知识库ID
            dimension (int): 向量维度
            
        返回:
            bool: 是否创建成功
        """
        try:
            collection_name = self.get_collection_name(knowledge_id)
            
            # 检查集合是否已存在
            if self.client.has_collection(collection_name):
                logger.info(f"集合已存在: {collection_name}")
                return True
                
            # 创建集合
            self.client.create_collection(
                collection_name=collection_name,
                dimension=dimension
            )
            logger.info(f"成功创建集合: {collection_name}，维度: {dimension}")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {str(e)}")
            return False
    
    def delete_collection(self, knowledge_id: str) -> bool:
        """
        删除集合
        
        参数:
            knowledge_id (str): 知识库ID
            
        返回:
            bool: 是否删除成功
        """
        try:
            collection_name = self.get_collection_name(knowledge_id)
            
            # 检查集合是否存在
            if not self.client.has_collection(collection_name):
                logger.info(f"集合不存在: {collection_name}")
                return True
                
            # 删除集合
            self.client.drop_collection(collection_name)
            logger.info(f"成功删除集合: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除集合失败: {str(e)}")
            return False
    
    def insert_vectors(self, 
                        knowledge_id: str, 
                        file_id: str,
                        chunks: List[Dict[str, Any]], 
                        embeddings: List[List[float]]) -> bool:
        """
        插入向量
        
        参数:
            knowledge_id (str): 知识库ID
            file_id (str): 文件ID
            chunks (List[Dict[str, Any]]): 文本块列表
            embeddings (List[List[float]]): 嵌入向量列表
            
        返回:
            bool: 是否插入成功
        """
        try:
            if not chunks or not embeddings or len(chunks) != len(embeddings):
                logger.error(f"文本块和嵌入向量不匹配: chunks={len(chunks)}, embeddings={len(embeddings)}")
                return False
                
            collection_name = self.get_collection_name(knowledge_id)
            
            # 检查集合是否存在，不存在则创建
            if not self.client.has_collection(collection_name):
                dimension = len(embeddings[0])
                self.create_collection(knowledge_id, dimension)
            
            # 准备插入数据
            data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # 提取基本元数据
                metadata = chunk.get("metadata", {})
                
                # 使用整数ID，将字符串ID转换为整数哈希值
                chunk_id = hash(f"{file_id}_{i}") % (2**63)
                
                data.append({
                    "id": chunk_id,  # 使用整数ID
                    "vector": embedding,
                    "text": chunk["text"],
                    "file_id": file_id,
                    "chunk_index": i,
                    "knowledge_id": knowledge_id,
                    # 添加其他元数据字段
                    **{f"meta_{k}": v for k, v in metadata.items() if k != "file_id" and k != "knowledge_id"}
                })
            
            # 插入数据
            res = self.client.insert(
                collection_name=collection_name,
                data=data
            )
            
            logger.info(f"成功插入向量: {collection_name}, 数量: {len(data)}")
            return True
            
        except Exception as e:
            logger.error(f"插入向量失败: {str(e)}")
            return False
    
    def delete_file_vectors(self, knowledge_id: str, file_id: str) -> bool:
        """
        删除指定文件的向量
        
        参数:
            knowledge_id (str): 知识库ID
            file_id (str): 文件ID
            
        返回:
            bool: 是否删除成功
        """
        try:
            collection_name = self.get_collection_name(knowledge_id)
            
            # 检查集合是否存在
            if not self.client.has_collection(collection_name):
                logger.info(f"集合不存在: {collection_name}")
                return True
            
            # 构建过滤条件，删除指定文件的所有向量
            filter_expr = f"file_id == '{file_id}'"
            
            # 删除向量
            res = self.client.delete(
                collection_name=collection_name,
                filter=filter_expr
            )
            
            logger.info(f"成功删除文件向量: {collection_name}, file_id: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除文件向量失败: {str(e)}")
            return False
    
    def search_similar(self, 
                       knowledge_id: str, 
                       query_vector: List[float], 
                       limit: int = 10, 
                       filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        搜索相似向量
        
        参数:
            knowledge_id (str): 知识库ID
            query_vector (List[float]): 查询向量
            limit (int): 返回结果数量限制
            filter_expr (Optional[str]): 过滤表达式
            
        返回:
            List[Dict[str, Any]]: 搜索结果
        """
        try:
            collection_name = self.get_collection_name(knowledge_id)
            
            # 检查集合是否存在
            if not self.client.has_collection(collection_name):
                logger.warning(f"集合不存在: {collection_name}")
                return []
            
            # 搜索相似向量
            results = self.client.search(
                collection_name=collection_name,
                data=[query_vector],
                filter=filter_expr,
                limit=limit,
                output_fields=["text", "file_id", "chunk_index", "knowledge_id"]
            )
            # print("milvus results:::",results)
            # 处理结果格式
            processed_results = []
            if results and len(results) > 0:
                print("知识库HIT：：：",results,type(results))
                print("知识库HIT：：：",type(results))
                print("知识库HIT：：：",results[0])
                print("知识库HIT：：：",type(results[0]))
                for hit in results[0]:
                    item = {
                        "text":hit.get("text", "") if hit.get("text", "") else hit.get("entity", {'text':''}).get("text",""),
                        "file_id": hit.get("file_id", "") if hit.get("file_id", "") else hit.get("entity", {'file_id':''}).get("file_id",""),
                        "chunk_index": hit.get("chunk_index", "") if hit.get("chunk_index", "") else hit.get("entity", {'chunk_index':''}).get("chunk_index",""),
                        "knowledge_id": hit.get("knowledge_id", "") if hit.get("knowledge_id", "") else hit.get("entity", {'knowledge_id':''}).get("knowledge_id",""),
                        "score": hit.get("distance", 0.0),
                        "distance": hit.get("distance", 0.0)
                    }
                    item["text"] = item["text"].replace("\n","").replace("\t","").replace(" ",'')
                    processed_results.append(item)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"搜索相似向量失败: {str(e)}")
            return []


class EmbeddingManager:
    """
    嵌入管理器类
    
    负责文本向量化，支持不同的模型提供商和向量化方法
    """
    
    # 初始化向量存储
    _vector_store = None
    _vector_store_init_attempted = False
    
    @classmethod
    def get_vector_store(cls) -> Optional[MilvusVectorStore]:
        """
        获取向量存储实例
        
        返回:
            Optional[MilvusVectorStore]: 向量存储实例，如果初始化失败则返回None
        """
        if cls._vector_store is None and not cls._vector_store_init_attempted:
            cls._vector_store_init_attempted = True
            try:
                if not HAS_PYMILVUS:
                    logger.error("pymilvus 未安装，向量化功能不可用")
                    return None
                    
                cls._vector_store = MilvusVectorStore()
                # 检查是否成功初始化
                if cls._vector_store.client is None:
                    logger.error("向量存储初始化失败")
                    cls._vector_store = None
            except Exception as e:
                logger.error(f"初始化向量存储时出错: {str(e)}")
                cls._vector_store = None
                
        return cls._vector_store
    
    @staticmethod
    def get_embedding_model(db: Session, model_id: str) -> Optional[Model]:
        """
        获取嵌入模型（同步版本）
        
        参数:
            db (Session): 数据库会话
            model_id (str): 模型ID
            
        返回:
            Optional[Model]: 模型对象或None
        """
        model = get_model(db, model_id)
        if not model:
            logger.error(f"找不到模型: {model_id}")
            return None
            
        if model.type != "embedding":
            logger.error(f"模型类型不是embedding: {model.type}")
            return None
            
        if model.status != "active":
            logger.error(f"模型不是活动状态: {model.status}")
            return None
            
        return model
    
    @staticmethod
    async def get_embedding_model_async(db: Session, model_id: str) -> Optional[Model]:
        """
        获取嵌入模型（异步版本）
        
        参数:
            db (Session): 数据库会话
            model_id (str): 模型ID
            
        返回:
            Optional[Model]: 模型对象或None
        """
        return EmbeddingManager.get_embedding_model(db, model_id)
    
    @staticmethod
    async def generate_embeddings(
        model: Model,
        texts: List[str]
    ) -> List[List[float]]:
        """
        生成文本嵌入向量
        
        参数:
            model (Model): 模型对象
            texts (List[str]): 文本列表
            
        返回:
            List[List[float]]: 嵌入向量列表
        """
        try:
            # 获取提供商
            provider = provider_manager.get_provider(model.provider)
            if not provider:
                logger.error(f"找不到提供商: {model.provider}")
                return []
                
            # 调用提供商的嵌入API
            embeddings_result = await provider.embedding(
                api_key=model.api_key,
                base_url=model.base_url,
                model=model.name,
                text=texts
            )
            print("生成嵌入向量原始结果::", str(embeddings_result)[:200], "...")
            
            # 处理不同格式的返回结果
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
                
            # 最后检查
            if not embeddings and isinstance(embeddings_result, dict):
                logger.warning(f"尝试进一步解析embedding结果: {str(embeddings_result)[:200]}...")
                # 可能是其他格式，尝试直接提取
                if "embeddings" in embeddings_result:
                    embeddings = embeddings_result["embeddings"]
            
            print(f"解析后的embeddings长度: {len(embeddings)}")
            return embeddings
            
        except Exception as e:
            logger.error(f"生成嵌入向量时出错: {str(e)}")
            return []
    
    @staticmethod
    async def process_text_with_embedding(
        db: Session,
        text: str,
        metadata: Dict[str, Any],
        model_id: str,
        chunking_method: Union[str, ChunkingMethod] = ChunkingMethod.RECURSIVE,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        """
        处理文本并生成嵌入向量
        
        参数:
            db (Session): 数据库会话
            text (str): 文本内容
            metadata (Dict[str, Any]): 文本元数据
            model_id (str): 嵌入模型ID
            chunking_method (Union[str, ChunkingMethod]): 分段方法
            chunk_size (int): 分段大小
            chunk_overlap (int): 分段重叠
            separator (Optional[str]): 分隔符
            
        返回:
            Tuple[List[Dict[str, Any]], List[List[float]]]: 分段和嵌入向量
        """
        try:
            # 获取嵌入模型
            model = await EmbeddingManager.get_embedding_model_async(db, model_id)
            if not model:
                logger.error(f"无法使用模型 {model_id} 生成嵌入向量")
                return [], []
                
            # 分段文本
            chunks = await TextChunker.chunk_text(
                text=text,
                method=chunking_method,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=separator,
                metadata=metadata
            )
            
            if not chunks:
                logger.warning("没有分段结果")
                return [], []
                
            # 提取文本内容用于生成嵌入向量
            texts = [chunk["text"] for chunk in chunks]
            
            # 生成嵌入向量
            embeddings = await EmbeddingManager.generate_embeddings(model, texts)
            
            return chunks, embeddings
            
        except Exception as e:
            logger.error(f"处理文本和生成嵌入向量时出错: {str(e)}")
            return [], []
    
    @staticmethod
    async def save_embeddings(
        output_dir: str,
        file_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> Optional[str]:
        """
        保存分段和嵌入向量到向量数据库和本地文件
        
        参数:
            output_dir (str): 输出目录
            file_id (str): 文件ID
            chunks (List[Dict[str, Any]]): 分段列表
            embeddings (List[List[float]]): 嵌入向量列表
            
        返回:
            Optional[str]: 保存的文件路径或None
        """
        try:
            if not chunks or not embeddings or len(chunks) != len(embeddings):
                logger.error(f"分段和嵌入向量不匹配: chunks={len(chunks)}, embeddings={len(embeddings)}")
                return None
                
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 构建保存路径
            save_path = os.path.join(output_dir, f"{file_id}_embeddings.json")
            
            # 准备保存的数据
            save_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                save_data.append({
                    "index": i,
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "embedding": embedding
                })
                
            # 保存到JSON文件
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            # 保存到向量数据库
            knowledge_id = chunks[0]["metadata"].get("knowledge_id", "")
            if knowledge_id:
                vector_store = EmbeddingManager.get_vector_store()
                vector_store.insert_vectors(knowledge_id, file_id, chunks, embeddings)
                
            return save_path
            
        except Exception as e:
            logger.error(f"保存嵌入向量时出错: {str(e)}")
            return None 
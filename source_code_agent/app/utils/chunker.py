import logging
from typing import List, Optional, Dict, Any, Union
from enum import Enum

# 导入LangChain相关库
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_text_splitters.base import TextSplitter

# 配置日志
logger = logging.getLogger(__name__)


class ChunkingMethod(str, Enum):
    """分段方法枚举"""
    CHARACTER = "character"  # 按字符分段
    TOKEN = "token"  # 按token分段
    SENTENCE = "sentence"  # 按句子分段
    PARAGRAPH = "paragraph"  # 按段落分段
    SEMANTIC = "semantic"  # 语义分段
    RECURSIVE = "recursive"  # 递归分段


class TextChunker:
    """
    文本分段器类
    
    支持多种分段方法，用于将长文本分割成适合向量化的小段
    """
    
    @staticmethod
    def get_chunker(
        method: ChunkingMethod,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: Optional[str] = None,
        separators: Optional[List[str]] = None,
        encoding_name: Optional[str] = "cl100k_base",
        model_name: Optional[str] = "all-MiniLM-L6-v2",
        is_separator_regex: bool = False,
        keep_separator: bool = False
    ) -> TextSplitter:
        """
        根据分段方法获取相应的分段器
        
        参数:
            method (ChunkingMethod): 分段方法
            chunk_size (int): 分段大小
            chunk_overlap (int): 分段重叠
            separator (Optional[str]): 分隔符，仅在某些分段方法中使用
            separators (Optional[List[str]]): 分隔符列表，按优先级降序排列
            encoding_name (Optional[str]): TokenTextSplitter使用的编码名称
            model_name (Optional[str]): SentenceTransformersTokenTextSplitter使用的模型名称
            is_separator_regex (bool): 分隔符是否为正则表达式
            keep_separator (bool): 是否保留分隔符
            
        返回:
            TextSplitter: LangChain文本分段器
        """
        try:
            if method == ChunkingMethod.CHARACTER:
                # 使用指定分隔符或默认分隔符
                return CharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separator=separator or "\n\n",
                    is_separator_regex=is_separator_regex,
                    keep_separator=keep_separator
                )
                
            elif method == ChunkingMethod.RECURSIVE:
                # 使用自定义分隔符列表或默认分隔符列表
                default_separators = ["\n\n", "\n", "。", "！", "？", "，", "；", "：", ".", "!", "?", ",", ";", ":", " ", ""]
                custom_separators = separators if separators else default_separators
                
                return RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=custom_separators,
                    keep_separator=keep_separator
                )
                
            elif method == ChunkingMethod.TOKEN:
                # 使用指定编码或默认编码
                return TokenTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    encoding_name=encoding_name
                )
                
            elif method == ChunkingMethod.SENTENCE:
                # 使用适合中文和英文的句子分段器
                sentence_separators = ["。", "！", "？", ".", "!", "?", "\n"]
                return RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=separators if separators else sentence_separators,
                    keep_separator=keep_separator
                )
                
            elif method == ChunkingMethod.PARAGRAPH:
                # 按段落分段，使用空行作为段落分隔符
                paragraph_separators = ["\n\n", "\n"]
                return RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=separators if separators else paragraph_separators,
                    keep_separator=keep_separator
                )
                
            elif method == ChunkingMethod.SEMANTIC:
                # 语义分段，使用sentence-transformers模型计算token
                return SentenceTransformersTokenTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    model_name=model_name
                )
                
            else:
                # 默认使用递归分段
                logger.warning(f"未知的分段方法: {method}，使用递归分段")
                default_separators = ["\n\n", "\n", "。", "！", "？", "，", "；", "：", ".", "!", "?", ",", ";", ":", " ", ""]
                return RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=separators if separators else default_separators
                )
                
        except Exception as e:
            logger.error(f"创建分段器出错: {str(e)}")
            # 出错时返回一个基本的分段器
            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
    
    @staticmethod
    async def chunk_text(
        text: str,
        method: Union[ChunkingMethod, str] = ChunkingMethod.RECURSIVE,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: Optional[str] = None,
        separators: Optional[List[str]] = None,
        encoding_name: Optional[str] = "cl100k_base",
        model_name: Optional[str] = "all-MiniLM-L6-v2",
        is_separator_regex: bool = False,
        keep_separator: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        将文本分段
        
        参数:
            text (str): 要分段的文本
            method (Union[ChunkingMethod, str]): 分段方法
            chunk_size (int): 分段大小
            chunk_overlap (int): 分段重叠
            separator (Optional[str]): 分隔符
            separators (Optional[List[str]]): 分隔符列表，按优先级降序排列
            encoding_name (Optional[str]): TokenTextSplitter使用的编码名称
            model_name (Optional[str]): SentenceTransformersTokenTextSplitter使用的模型名称
            is_separator_regex (bool): 分隔符是否为正则表达式
            keep_separator (bool): 是否保留分隔符
            metadata (Optional[Dict[str, Any]]): 元数据，会添加到每个分段中
            
        返回:
            List[Dict[str, Any]]: 分段后的文本列表，每个元素包含文本和元数据
        """
        if not text:
            return []
        
        # 如果method是字符串，转换为枚举
        if isinstance(method, str):
            try:
                method = ChunkingMethod(method)
            except ValueError:
                logger.warning(f"无效的分段方法字符串: {method}，使用递归分段")
                method = ChunkingMethod.RECURSIVE
        
        # 获取分段器
        chunker = TextChunker.get_chunker(
            method=method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            separators=separators,
            encoding_name=encoding_name,
            model_name=model_name,
            is_separator_regex=is_separator_regex,
            keep_separator=keep_separator
        )
        
        # 分段
        chunks = chunker.create_documents([text], metadatas=[metadata or {}])
        
        # 将Document对象转换为字典
        result = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = chunk.metadata.copy() if chunk.metadata else {}
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            
            result.append({
                "text": chunk.page_content,
                "metadata": chunk_metadata
            })
        
        return result 
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple

class DBConnector(ABC):
    """
    数据库连接器基类
    
    定义所有数据库连接器需要实现的接口
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        连接到数据库
        
        返回:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """断开数据库连接"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """
        测试数据库连接
        
        返回:
            Tuple[bool, Optional[str]]: (是否成功, 错误信息)
        """
        pass
    
    @abstractmethod
    async def get_databases(self) -> List[str]:
        """
        获取数据库列表
        
        返回:
            List[str]: 数据库名称列表
        """
        pass
    
    @abstractmethod
    async def get_tables(self, database: Optional[str] = None) -> List[str]:
        """
        获取指定数据库中的表列表
        
        参数:
            database: 数据库名称，默认为当前数据库
            
        返回:
            List[str]: 表名列表
        """
        pass
    
    @abstractmethod
    async def get_table_schema(self, table: str, database: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取表结构
        
        参数:
            table: 表名
            database: 数据库名称，默认为当前数据库
            
        返回:
            List[Dict[str, Any]]: 字段信息列表
        """
        pass
    
    @abstractmethod
    async def execute_query(self, query: str) -> Tuple[List[Dict[str, Any]], List[str], int, int]:
        """
        执行查询
        
        参数:
            query: SQL查询语句
            
        返回:
            Tuple[List[Dict[str, Any]], List[str], int, int]: (结果行列表, 列名列表, 影响行数, 执行时间ms)
        """
        pass
    
    @abstractmethod
    async def get_db_structure(self) -> Dict[str, Any]:
        """
        获取完整数据库结构
        
        返回:
            Dict[str, Any]: 数据库结构信息
        """
        pass 
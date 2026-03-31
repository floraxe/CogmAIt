import time
from typing import Dict, List, Any, Optional, Tuple

import asyncpg

from app.utils.db_connectors.base import DBConnector


class PostgreSQLConnector(DBConnector):
    """PostgreSQL数据库连接器"""
    
    def __init__(self, host: str, port: int, database: str, username: str, password: str, **kwargs):
        """
        初始化PostgreSQL连接器
        
        参数:
            host: 数据库主机地址
            port: 端口号
            database: 数据库名
            username: 用户名
            password: 密码
            **kwargs: 额外参数，如SSL设置等
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.extra_params = kwargs
        self.pool = None
    
    async def connect(self) -> bool:
        """
        连接到PostgreSQL数据库
        
        返回:
            bool: 连接是否成功
        """
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database,
                **self.extra_params
            )
            return True
        except Exception as e:
            print(f"PostgreSQL连接失败: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """断开数据库连接"""
        try:
            if self.pool:
                await self.pool.close()
                self.pool = None
        except Exception as e:
            print(f"PostgreSQL断开连接失败: {str(e)}")
            # 即使失败也重置连接池引用
            self.pool = None
    
    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """
        测试数据库连接
        
        返回:
            Tuple[bool, Optional[str]]: (是否成功, 错误信息)
        """
        try:
            # 创建临时连接，专门用于测试
            conn = await asyncpg.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database,
                **self.extra_params
            )
            
            try:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    return True, None
                return False, "连接测试失败"
            finally:
                await conn.close()
        except Exception as e:
            return False, str(e)
    
    async def get_databases(self) -> List[str]:
        """
        获取数据库列表
        
        返回:
            List[str]: 数据库名称列表
        """
        if not self.pool:
            await self.connect()
        
        databases = []
        async with self.pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT datname FROM pg_database 
                WHERE datistemplate = false AND datname != 'postgres'
            """)
            databases = [row['datname'] for row in result]
        return databases
    
    async def get_tables(self, database: Optional[str] = None) -> List[str]:
        """
        获取指定数据库中的表列表
        
        参数:
            database: 数据库名称，默认为当前数据库
            
        返回:
            List[str]: 表名列表
        """
        if not self.pool:
            await self.connect()
        
        # PostgreSQL需要连接到具体的数据库才能查询表，所以忽略database参数
        tables = []
        
        async with self.pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            tables = [row['table_name'] for row in result]
        return tables
    
    async def get_table_schema(self, table: str, database: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取表结构
        
        参数:
            table: 表名
            database: 数据库名称，默认为当前数据库
            
        返回:
            List[Dict[str, Any]]: 字段信息列表
        """
        if not self.pool:
            await self.connect()
        
        columns = []
        
        async with self.pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT 
                    column_name, 
                    data_type,
                    character_maximum_length,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = $1
                ORDER BY ordinal_position
            """, table)
            
            for row in result:
                data_type = row['data_type']
                if row['character_maximum_length'] is not None:
                    data_type = f"{data_type}({row['character_maximum_length']})"
                
                columns.append({
                    "name": row['column_name'],
                    "type": data_type,
                    "nullable": row['is_nullable'] == 'YES',
                    "default": row['column_default']
                })
        return columns
    
    async def execute_query(self, query: str) -> Tuple[List[Dict[str, Any]], List[str], int, int]:
        """
        执行查询
        
        参数:
            query: SQL查询语句
            
        返回:
            Tuple[List[Dict[str, Any]], List[str], int, int]: (结果行列表, 列名列表, 影响行数, 执行时间ms)
        """
        if not self.pool:
            await self.connect()
        
        start_time = time.time()
        rows = []
        columns = []
        affected_rows = 0
        
        async with self.pool.acquire() as conn:
            try:
                if query.strip().upper().startswith(('SELECT', 'WITH', 'SHOW', 'EXPLAIN')):
                    stmt = await conn.prepare(query)
                    result = await stmt.fetch()
                    
                    if result:
                        columns = [desc for desc in stmt.get_attributes()]
                        rows = [dict(zip(columns, row)) for row in result]
                        affected_rows = len(rows)
                else:
                    result = await conn.execute(query)
                    # PostgreSQL在执行非查询语句时会返回如 "DELETE 5" 这样的文本
                    if result:
                        parts = result.split()
                        if len(parts) > 1 and parts[1].isdigit():
                            affected_rows = int(parts[1])
            except Exception as e:
                raise e
        
        end_time = time.time()
        execution_time = int((end_time - start_time) * 1000)  # 转换为毫秒
        
        return rows, columns, affected_rows, execution_time
    
    async def get_db_structure(self) -> Dict[str, Any]:
        """
        获取完整数据库结构
        
        返回:
            Dict[str, Any]: 数据库结构信息
        """
        if not self.pool:
            await self.connect()
        
        structure = {
            "database": self.database,
            "tables": {}
        }
        
        tables = await self.get_tables()
        
        for table in tables:
            columns = await self.get_table_schema(table)
            
            # 获取表索引
            indexes = []
            async with self.pool.acquire() as conn:
                result = await conn.fetch("""
                    SELECT
                        i.relname as index_name,
                        a.attname as column_name,
                        ix.indisunique as is_unique,
                        am.amname as index_type
                    FROM
                        pg_class t,
                        pg_class i,
                        pg_index ix,
                        pg_attribute a,
                        pg_am am
                    WHERE
                        t.oid = ix.indrelid
                        AND i.oid = ix.indexrelid
                        AND a.attrelid = t.oid
                        AND a.attnum = ANY(ix.indkey)
                        AND i.relam = am.oid
                        AND t.relname = $1
                    ORDER BY
                        i.relname, a.attnum
                """, table)
                
                for row in result:
                    indexes.append({
                        "name": row['index_name'],
                        "column": row['column_name'],
                        "unique": row['is_unique'],
                        "type": row['index_type']
                    })
            
            # 获取外键关系
            foreign_keys = []
            async with self.pool.acquire() as conn:
                result = await conn.fetch("""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM
                        information_schema.table_constraints AS tc
                        JOIN information_schema.key_column_usage AS kcu
                          ON tc.constraint_name = kcu.constraint_name
                          AND tc.table_schema = kcu.table_schema
                        JOIN information_schema.constraint_column_usage AS ccu
                          ON ccu.constraint_name = tc.constraint_name
                          AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = $1
                """, table)
                
                for row in result:
                    foreign_keys.append({
                        "column": row['column_name'],
                        "referenced_table": row['foreign_table_name'],
                        "referenced_column": row['foreign_column_name']
                    })
            
            structure["tables"][table] = {
                "columns": columns,
                "indexes": indexes,
                "foreign_keys": foreign_keys
            }
        
        return structure 
import time
from typing import Dict, List, Any, Optional, Tuple

import aiomysql
import pymysql

from app.utils.db_connectors.base import DBConnector


class MySQLConnector(DBConnector):
    """MySQL数据库连接器"""
    
    def __init__(self, host: str, port: int, database: str, username: str, password: str, **kwargs):
        """
        初始化MySQL连接器
        
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
        连接到MySQL数据库
        
        返回:
            bool: 连接是否成功
        """
        try:
            self.pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                db=self.database,
                charset='utf8mb4',
                **self.extra_params
            )
            return True
        except Exception as e:
            print(f"MySQL连接失败: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """断开数据库连接"""
        try:
            if self.pool:
                self.pool.close()
                await self.pool.wait_closed()
                self.pool = None
        except Exception as e:
            print(f"MySQL断开连接失败: {str(e)}")
            # 即使失败也重置连接池引用
            self.pool = None
    
    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """
        测试数据库连接
        
        返回:
            Tuple[bool, Optional[str]]: (是否成功, 错误信息)
        """
        try:
            # 创建临时连接，而不使用连接池，专门用于测试
            conn = await aiomysql.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                db=self.database,
                charset='utf8mb4',
                **self.extra_params
            )
            
            try:
                cur = await conn.cursor()
                await cur.execute("SELECT 1")
                result = await cur.fetchone()
                await cur.close()
                if result and result[0] == 1:
                    return True, None
                return False, "连接测试失败"
            finally:
                conn.close()
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
            async with conn.cursor() as cursor:
                await cursor.execute("SHOW DATABASES")
                result = await cursor.fetchall()
                databases = [row[0] for row in result if row[0] not in ['information_schema', 'performance_schema', 'mysql', 'sys']]
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
        
        db_name = database or self.database
        tables = []
        
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"SHOW TABLES FROM `{db_name}`")
                result = await cursor.fetchall()
                tables = [row[0] for row in result]
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
        
        db_name = database or self.database
        columns = []
        
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(f"DESCRIBE `{db_name}`.`{table}`")
                result = await cursor.fetchall()
                for row in result:
                    columns.append({
                        "name": row['Field'],
                        "type": row['Type'],
                        "nullable": row['Null'] == 'YES',
                        "key": row['Key'],
                        "default": row['Default'],
                        "extra": row['Extra']
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
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                try:
                    affected_rows = await cursor.execute(query)
                    if query.strip().upper().startswith(('SELECT', 'SHOW', 'DESCRIBE', 'EXPLAIN')):
                        result = await cursor.fetchall()
                        rows = [dict(row) for row in result]
                        if rows:
                            columns = list(rows[0].keys())
                    
                    await conn.commit()
                except Exception as e:
                    await conn.rollback()
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
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(f"SHOW INDEX FROM `{self.database}`.`{table}`")
                    result = await cursor.fetchall()
                    for row in result:
                        indexes.append({
                            "name": row['Key_name'],
                            "column": row['Column_name'],
                            "unique": not row['Non_unique'],
                            "type": row['Index_type']
                        })
            
            # 获取外键关系
            foreign_keys = []
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    try:
                        await cursor.execute(f"""
                            SELECT
                                COLUMN_NAME,
                                REFERENCED_TABLE_NAME,
                                REFERENCED_COLUMN_NAME
                            FROM
                                INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                            WHERE
                                TABLE_SCHEMA = %s AND
                                TABLE_NAME = %s AND
                                REFERENCED_TABLE_NAME IS NOT NULL
                        """, (self.database, table))
                        result = await cursor.fetchall()
                        for row in result:
                            foreign_keys.append({
                                "column": row['COLUMN_NAME'],
                                "referenced_table": row['REFERENCED_TABLE_NAME'],
                                "referenced_column": row['REFERENCED_COLUMN_NAME']
                            })
                    except Exception as e:
                        # 部分数据库可能不支持该查询
                        print(f"获取外键关系失败: {str(e)}")
            
            structure["tables"][table] = {
                "columns": columns,
                "indexes": indexes,
                "foreign_keys": foreign_keys
            }
        
        return structure 
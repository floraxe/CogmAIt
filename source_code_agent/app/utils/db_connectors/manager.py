from typing import Dict, Any, Optional, Tuple

from sqlalchemy.orm import Session

from app.models.datasource import DataSource
from app.utils.db_connectors.base import DBConnector
from app.utils.db_connectors.factory import DBConnectorFactory


class DBConnectorManager:
    """数据库连接器管理器，用于从数据库获取数据源信息并创建连接器"""
    
    @staticmethod
    async def get_connector_by_id(datasource_id: str, db: Session) -> Tuple[Optional[DBConnector], Optional[str]]:
        """
        根据数据源ID获取数据库连接器
        
        参数:
            datasource_id: 数据源ID
            db: 数据库会话
            
        返回:
            Tuple[Optional[DBConnector], Optional[str]]: (数据库连接器, 错误信息)
        """
        try:
            # 从数据库获取数据源信息
            datasource = db.query(DataSource).filter(DataSource.id == datasource_id).first()
            if not datasource:
                return None, "数据源不存在"
            
            # 检查数据源状态
            if not datasource.is_active:
                return None, "数据源未激活"
            
            # 创建数据库连接器
            connector = DBConnectorFactory.create_connector(
                datasource.type,
                datasource.host,
                datasource.port,
                datasource.database,
                datasource.username,
                datasource.password,
                **(datasource.extra_params or {})
            )
            
            if not connector:
                return None, f"不支持的数据库类型: {datasource.type}"
            
            # 尝试连接
            connection_success = await connector.connect()
            if not connection_success:
                # 连接失败时确保清理资源
                try:
                    await connector.disconnect()
                except Exception:
                    pass
                return None, "无法连接到数据库"
            
            return connector, None
        except Exception as e:
            return None, f"获取连接器失败: {str(e)}" 
import logging
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from app.models.mcp import MCPService

# 配置日志
logger = logging.getLogger(__name__)

def get_mcp_service(db: Session, service_id: str) -> Optional[Dict[str, Any]]:
    """
    获取指定ID的MCP服务
    
    参数:
        db (Session): 数据库会话
        service_id (str): MCP服务ID
        
    返回:
        Optional[Dict[str, Any]]: MCP服务对象或None
    """
    try:
        service = db.query(MCPService).filter(MCPService.id == service_id).first()
        if not service:
            return None
        
        # 将SQLAlchemy模型转换为字典
        return {
            "id": service.id,
            "name": service.name,
            "description": service.description,
            "type": service.type,
            "provider": service.provider,
            "icon": service.icon,
            "deployment_type": service.deployment_type,
            "is_official": service.is_official,
            "config_template": service.config_template,
            "tags": service.tags
        }
    except Exception as e:
        logger.error(f"获取MCP服务失败: {str(e)}")
        return None 
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.v1.endpoints import (
    users, auth, knowledge, models, role,
    agents, graph, files, file_preview, 
    datasources, mcp, dashboard
)
from app.db.session import get_db
from app.utils.deps import get_current_active_user

api_router = APIRouter()

# 包含各模块的路由
api_router.include_router(models.router, prefix="/models", tags=["模型管理"])
api_router.include_router(users.router, prefix="/users", tags=["用户管理"])
api_router.include_router(auth.router, prefix="/auth", tags=["用户认证"])
api_router.include_router(knowledge.router, prefix="/knowledge", tags=["知识库管理"]) 
api_router.include_router(role.router, prefix="/roles", tags=["角色管理"])
api_router.include_router(agents.router, prefix="/agents", tags=["智能助手"])
api_router.include_router(graph.router, prefix="/graphs", tags=["知识图谱"])
api_router.include_router(files.router, prefix="/files", tags=["文件管理"])
# api_router.include_router(chat.router, prefix="/chat", tags=["对话"])  # chat模块不存在，注释掉
api_router.include_router(file_preview.router, prefix="/file-preview", tags=["文件预览"])
api_router.include_router(datasources.router, prefix="/datasources", tags=["数据源管理"])
api_router.include_router(mcp.router, prefix="/mcp", tags=["MCP服务"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["仪表盘"])

# 添加/user路径，与个人用户相关的接口
api_router.include_router(users.router, prefix="/user", tags=["个人中心"])

# 添加智能体类型路由
@api_router.get("/agent-types", tags=["智能体管理"])
async def agent_types_endpoint(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """获取智能体类型列表"""
    # 临时返回一些固定的类型
    agent_types = [
        {"id": 1, "name": "问答助手", "value": "qa", "description": "基于知识库的问答智能体"},
        {"id": 2, "name": "对话助手", "value": "chat", "description": "通用对话智能体"},
        {"id": 3, "name": "客服助手", "value": "customer_service", "description": "客服场景智能体"}
    ]
    return agent_types 
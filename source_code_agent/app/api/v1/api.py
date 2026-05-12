from fastapi import APIRouter

from app.api.v1.endpoints import (
    users, auth, knowledge, models, role,
    agents, graph, files, file_preview,
    datasources, mcp, dashboard,
)

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
api_router.include_router(file_preview.router, prefix="/file-preview", tags=["文件预览"])
api_router.include_router(datasources.router, prefix="/datasources", tags=["数据源管理"])
api_router.include_router(mcp.router, prefix="/mcp", tags=["MCP服务"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["仪表盘"])

# 添加/user路径，与个人用户相关的接口
api_router.include_router(users.router, prefix="/user", tags=["个人中心"])
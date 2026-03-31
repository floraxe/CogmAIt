from typing import Callable, Optional

from fastapi import Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.utils.security import get_current_active_user, get_current_user, oauth2_scheme, check_permission


def get_permission_validator(required_permission: str) -> Callable:
    """
    创建权限验证依赖
    
    参数:
        required_permission (str): 所需的权限
    
    返回:
        Callable: 权限验证依赖函数
    """
    async def validate_permission(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if not check_permission(current_user, required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"权限不足: 需要 '{required_permission}' 权限"
            )
        return current_user
    
    return validate_permission


# 预定义权限依赖项
get_current_admin = get_permission_validator("admin")
get_model_admin = get_permission_validator("model:*")
get_knowledge_admin = get_permission_validator("knowledge:*")
get_graph_admin = get_permission_validator("graph:*")
get_agent_admin = get_permission_validator("agent:*")
get_user_admin = get_permission_validator("user:*")


async def get_optional_current_user(
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[User]:
    """
    获取当前用户，但不强制要求登录
    
    如果用户未登录或token无效，返回None
    
    参数:
        db (Session): 数据库会话
        token (Optional[str]): 认证token
        
    返回:
        Optional[User]: 用户对象或None
    """
    if not token:
        return None
        
    try:
        # 尝试获取当前用户
        return await get_current_user(db=db, token=token)
    except HTTPException:
        # 如果认证失败，返回None
        return None 


async def get_current_user_for_users(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme),
) -> User:
    """
    获取当前用户，用于用户管理相关接口
    允许admin(系统管理员)和department_admin(部门管理员)访问
    
    参数:
        db (Session): 数据库会话
        token (str): 认证token
        
    返回:
        User: 用户对象
    """
    # 获取当前登录用户
    user = await get_current_user(db=db, token=token)
    
    # 验证用户是否已激活
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号未激活"
        )
    
    # 只允许admin和department_admin角色访问
    if user.role not in ["admin", "department_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足：需要管理员权限"
        )
    
    return user 
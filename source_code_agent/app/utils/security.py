from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union, List

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models.user import User
from app.schemas.user import TokenData
from app.utils import utc_to_cst
from app.db.base import get_cn_datetime

# 密码上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# OAuth2 密码授权
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

# 角色权限映射
ROLE_PERMISSIONS = {
    "admin": ["*"],  # 管理员拥有所有权限
    "operator": ["model:*", "knowledge:*", "graph:*", "agent:*", "user:read"],  # 操作员可以管理模型、知识库、图谱和智能体
    "user": ["agent:read", "agent:use"],  # 普通用户只能查看和使用智能体
    "guest": ["agent:read"]  # 访客只能查看智能体
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码是否匹配"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """获取密码哈希值"""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌"""
    to_encode = data.copy()
    
    # 设置令牌过期时间，使用中国标准时间
    if expires_delta:
        expires = get_cn_datetime() + expires_delta
    else:
        expires = get_cn_datetime() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expires})
    
    # 生成JWT令牌
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """通过用户名获取用户"""
    return db.query(User).filter(User.username == username).first()


def authenticate_user(db: Session, username: str, password: str) -> Union[User, bool]:
    """验证用户身份"""
    user = get_user_by_username(db, username)
    
    # 验证用户存在且密码正确
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    
    return user


def get_permissions_for_role(role: str) -> List[str]:
    """获取角色的权限列表"""
    return ROLE_PERMISSIONS.get(role, [])


async def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的身份凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # 解码JWT令牌
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        print("用户：：",payload)
        username: str = payload.get("sub")
        
        if username is None:
            raise credentials_exception
        
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    # 获取用户
    user = get_user_by_username(db, username=token_data.username)
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="用户已禁用")
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """获取当前活动用户"""
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="用户已禁用")
    
    return current_user


def check_permission(user: User, required_permission: str) -> bool:
    """检查用户是否拥有某个权限"""
    if not user or not user.is_active:
        return False
    
    # 获取用户角色的权限列表
    permissions = get_permissions_for_role(user.role)
    
    # 检查是否包含通配符权限
    if "*" in permissions:
        return True
    
    # 检查具体权限
    if required_permission in permissions:
        return True
    
    # 检查资源级别权限 (如 "model:*")
    resource_type = required_permission.split(":")[0] if ":" in required_permission else ""
    resource_wildcard = f"{resource_type}:*" if resource_type else ""
    
    return resource_wildcard in permissions


def update_user_last_login(db: Session, user: User) -> User:
    """
    更新用户最后登录时间
    
    参数:
        db (Session): 数据库会话
        user (User): 用户对象
    
    返回:
        User: 更新后的用户
    """
    # 使用中国标准时间更新最后登录时间
    user.last_login = get_cn_datetime()
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return user 


async def get_user_by_token(token: str) -> Optional[User]:
    """
    通过token获取用户
    
    参数:
        token (str): JWT token
    
    返回:
        Optional[User]: 用户对象或None
    """
    try:
        # 解码JWT令牌
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        
        if username is None:
            return None
        
        # 获取用户
        from app.db.session import SessionLocal
        db = SessionLocal()
        try:
            user = get_user_by_username(db, username=username)
            
            if user is None or not user.is_active:
                return None
            
            return user
        finally:
            db.close()
    except JWTError:
        return None 
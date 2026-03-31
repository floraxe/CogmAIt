from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models.user import User
from app.schemas.user import Token, LoginRequest, LoginResponse, UserResponse, UserCreate
from app.utils.security import (
    authenticate_user, 
    create_access_token, 
    get_current_active_user,
    update_user_last_login
)
from app.utils.user import create_user

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """
    用户登录API
    """
    # 验证用户身份
    user = authenticate_user(db, login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 设置令牌过期时间，记住我功能
    if login_data.remember:
        # 如果选择记住我，令牌有效期更长
        access_token_expires = timedelta(days=30)
    else:
        # 默认令牌有效期
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # 创建访问令牌
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    # 更新用户最后登录时间
    update_user_last_login(db, user)
    
    # 返回令牌和用户信息，使用to_dict()方法转换用户对象
    return {
        "token": access_token,
        "user": user.to_dict()
    }


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_in: UserCreate,
    db: Session = Depends(get_db)
):
    """
    用户注册API - 公开接口，无需身份验证
    """
    # 检查用户名是否已存在
    db_user = db.query(User).filter(User.username == user_in.username).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已存在"
        )
    
    # 检查邮箱是否已存在
    if user_in.email:
        email_user = db.query(User).filter(User.email == user_in.email).first()
        if email_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已被使用"
            )
    
    # 设置默认角色为普通用户
    if not user_in.role:
        user_in.role = "user"
    
    # 创建用户
    user = create_user(db=db, user_in=user_in)
    
    # 使用用户模型的to_dict方法获取字典表示形式，确保包含created字段
    return user.to_dict()


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    OAuth2兼容令牌登录API
    """
    # 验证用户身份
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 创建访问令牌
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    # 更新用户最后登录时间
    update_user_last_login(db, user)
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/logout")
async def logout():
    """
    用户退出登录API
    
    前端处理，清除token
    """
    return {"code": 200, "message": "退出成功"}


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    获取当前用户信息
    """
    return current_user 
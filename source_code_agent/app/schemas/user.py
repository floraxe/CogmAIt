from pydantic import BaseModel, Field, EmailStr, ConfigDict
from typing import Optional, List, Dict, Any


class UserBase(BaseModel):
    """用户基础模式"""
    username: str
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    role: str = "user"
    department: Optional[str] = None
    position: Optional[str] = None
    avatar: Optional[str] = None


class UserCreate(UserBase):
    """创建用户的请求模式"""
    password: str


class UserUpdate(BaseModel):
    """更新用户的请求模式"""
    username: Optional[str] = None
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    avatar: Optional[str] = None
    status: Optional[str] = None
    password: Optional[str] = None  # 可选密码更新


class UserResponse(UserBase):
    """用户响应模式"""
    id: str
    status: str
    lastLogin: Optional[str] = None
    created: str
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        protected_namespaces=()
    )


class UserListResponse(BaseModel):
    """用户列表响应"""
    total: int
    items: List[UserResponse]


class Token(BaseModel):
    """认证Token模式"""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token内部数据模式"""
    username: Optional[str] = None
    permissions: List[str] = []


class LoginRequest(BaseModel):
    """登录请求模式"""
    username: str
    password: str
    remember: bool = False


class LoginResponse(BaseModel):
    """登录响应模式"""
    token: str
    user: UserResponse


class RoleBase(BaseModel):
    """角色基础模式"""
    name: str
    value: str
    description: Optional[str] = None
    permissions: Optional[List[str]] = None


class RoleCreate(RoleBase):
    """创建角色请求模式"""
    pass


class RoleUpdate(BaseModel):
    """更新角色请求模式"""
    name: Optional[str] = None
    description: Optional[str] = None
    permissions: Optional[List[str]] = None


class RoleResponse(RoleBase):
    """角色响应模式"""
    id: int
    created: Optional[str] = None
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        protected_namespaces=()
    )


class PasswordReset(BaseModel):
    """密码重置请求模式"""
    user_id: str


class UserProfileUpdate(BaseModel):
    """用户个人资料更新请求模式"""
    username: Optional[str] = None
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None


class PasswordUpdate(BaseModel):
    """密码更新请求模式"""
    oldPassword: str
    newPassword: str 
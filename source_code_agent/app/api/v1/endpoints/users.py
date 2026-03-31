from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.schemas.user import (
    UserCreate, 
    UserUpdate, 
    UserResponse, 
    UserListResponse,
    RoleResponse,
    PasswordReset,
    UserProfileUpdate,
    PasswordUpdate
)
from app.utils import user as user_utils
from app.utils.deps import get_current_admin, get_user_admin, get_current_user_for_users
from app.utils.security import get_current_active_user, verify_password

router = APIRouter()


@router.get("/", response_model=UserListResponse)
async def get_users(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    username: Optional[str] = None,
    name: Optional[str] = None,
    role: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user_for_users)
):
    """
    获取用户列表
    """
    skip = (page - 1) * limit
    
    # 根据用户角色确定返回的用户范围
    users = user_utils.get_users(
        db,
        skip=skip,
        limit=limit,
        username=username,
        name=name,
        role=role,
        status=status,
        current_user=current_user  # 传入当前用户，用于部门权限控制
    )
    
    # 获取总数（同样应用权限控制）
    total = user_utils.count_users(db, username=username, name=name, role=role, status=status, current_user=current_user)
    
    return {
        "total": total,
        "items": [user.to_dict() for user in users]
    }

@router.get("/departments", response_model=List[str])
async def get_departments(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_users)
):
    """
    获取所有部门列表
    """
    # 如果是系统管理员，获取所有部门
    if current_user.role == "admin":
        departments = db.query(User.department).distinct().filter(User.department != None).all()
    else:
        # 部门管理员只能看到自己的部门
        departments = [(current_user.department,)] if current_user.department else []
    
    # 提取部门名称并过滤掉None值
    department_list = [dept[0] for dept in departments if dept[0]]
    
    return department_list

@router.get("/positions", response_model=List[str])
async def get_positions(
    db: Session = Depends(get_db),
    department: Optional[str] = None,
    current_user: User = Depends(get_current_user_for_users)
):
    """
    获取职位列表，可按部门过滤
    """
    query = db.query(User.position).distinct().filter(User.position != None)
    
    # 如果不是系统管理员，只能查看自己部门的职位
    if current_user.role != "admin":
        query = query.filter(User.department == current_user.department)
    # 如果指定了部门，则按部门过滤
    elif department:
        query = query.filter(User.department == department)
    
    positions = query.all()
    
    # 提取职位名称并过滤掉None值
    position_list = [pos[0] for pos in positions if pos[0]]
    
    return position_list

# 个人中心相关接口

@router.get("/info", response_model=UserResponse)
async def get_current_user_info(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    获取当前登录用户的信息
    """
    return current_user.to_dict()


@router.get("/statistics", response_model=dict)
async def get_user_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    获取当前用户的统计数据
    
    不需要额外权限，用户可以查看自己的统计数据
    """
    # 导入需要的模型
    from app.models.agent import Agent, AgentChatHistory
    from app.models.knowledge import Knowledge
    from app.models.graph import Graph
    from app.models.file import File
    from app.models.datasource import DataSource
    from sqlalchemy import func, or_
    
    # 用户ID
    user_id = current_user.id
    
    # 统计用户创建的智能体数量 - 使用Agent模型的user_id字段
    agent_count = db.query(func.count(Agent.id)).filter(
        Agent.user_id == user_id
    ).scalar() or 0
    
    # 统计用户创建的知识库数量 - 使用Knowledge模型的user_id字段
    knowledge_count = db.query(func.count(Knowledge.id)).filter(
        Knowledge.user_id == user_id
    ).scalar() or 0
    
    # 统计用户创建的知识图谱数量 - 使用Graph模型的user_id字段
    from app.models.graph import Graph
    graph_count = db.query(func.count(Graph.id)).filter(
        Graph.user_id == user_id
    ).scalar() or 0
    
    # 统计用户上传的文件数量 - 使用File模型的user_id字段
    file_count = db.query(func.count(File.id)).filter(
        File.user_id == user_id
    ).scalar() or 0
    
    # 统计用户创建的数据源数量
    datasource_count = db.query(func.count(DataSource.id)).filter(
        DataSource.created_by == user_id
    ).scalar() or 0
    
    # 统计用户的对话次数
    chat_count = db.query(func.count(AgentChatHistory.id)).filter(
        AgentChatHistory.user_id == user_id
    ).scalar() or 0
    
    # 统计用户最近7天的对话次数
    from datetime import datetime, timedelta
    seven_days_ago = datetime.now() - timedelta(days=7)
    
    recent_chat_count = db.query(func.count(AgentChatHistory.id)).filter(
        AgentChatHistory.user_id == user_id,
        AgentChatHistory.created_at >= seven_days_ago
    ).scalar() or 0
    
    # 统计用户最常使用的智能体
    from sqlalchemy import desc
    
    most_used_agent_query = db.query(
        AgentChatHistory.agent_id,
        func.count(AgentChatHistory.id).label('count')
    ).filter(
        AgentChatHistory.user_id == user_id
    ).group_by(
        AgentChatHistory.agent_id
    ).order_by(
        desc('count')
    ).first()
    
    most_used_agent = None
    if most_used_agent_query:
        agent_id, count = most_used_agent_query
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if agent:
            most_used_agent = {
                "id": agent.id,
                "name": agent.name,
                "count": count
            }
    
    # 返回统计结果
    return {
        "agent_count": agent_count,
        "knowledge_count": knowledge_count,
        "graph_count": graph_count,
        "file_count": file_count,
        "datasource_count": datasource_count,
        "chat_count": chat_count,
        "recent_chat_count": recent_chat_count,
        "most_used_agent": most_used_agent
    }


@router.put("/profile", response_model=UserResponse)
async def update_user_profile(
    profile_update: UserProfileUpdate = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    更新当前用户的个人资料
    """
    # 不允许更改admin用户名
    if current_user.username == "admin" and profile_update.username and profile_update.username != "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不能修改管理员用户名"
        )
    
    # 检查用户名是否重复
    if profile_update.username and profile_update.username != current_user.username:
        existing_user = db.query(User).filter(User.username == profile_update.username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已存在"
            )
    
    # 构建用户更新对象
    user_in = UserUpdate(**profile_update.dict(exclude_unset=True))
    
    # 更新用户
    updated_user = user_utils.update_user(db=db, user=current_user, user_in=user_in)
    return updated_user.to_dict()


@router.put("/password")
async def change_password(
    password_update: PasswordUpdate = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    修改当前用户的密码
    """
    # 验证旧密码
    if not verify_password(password_update.oldPassword, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="旧密码不正确"
        )
    
    # 更新密码
    current_user.hashed_password = user_utils.get_password_hash(password_update.newPassword)
    db.add(current_user)
    db.commit()
    
    return {
        "code": 200,
        "message": "密码修改成功"
    }


@router.post("/avatar", response_model=UserResponse)
async def upload_avatar(
    avatar_data: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    上传用户头像
    
    接收base64编码的头像图片并保存
    """
    # 验证请求体中包含avatar字段
    if "avatar" not in avatar_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请求缺少avatar字段"
        )
    
    # 更新头像
    current_user.avatar = avatar_data["avatar"]
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    
    return current_user.to_dict()


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_new_user(
    user_in: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_users),
):
    """
    创建新用户
    """
    # 检查用户名是否已存在
    db_user = db.query(User).filter(User.username == user_in.username).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已存在"
        )
    
    # 检查邮箱是否已存在（如果提供了邮箱）
    if user_in.email:
        email_user = db.query(User).filter(User.email == user_in.email).first()
        if email_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已被其他用户使用"
            )
    
    # 检查手机号是否已存在（如果提供了手机号）
    if user_in.phone:
        phone_user = db.query(User).filter(User.phone == user_in.phone).first()
        if phone_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="手机号已被其他用户使用"
            )
    
    # 部门权限控制
    if current_user.role != "admin":
        # 部门管理员只能创建本部门用户
        if user_in.department and user_in.department != current_user.department:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足：只能创建本部门用户"
            )
        
        # 强制设置部门为当前用户的部门
        user_in.department = current_user.department
        
        # 限制创建的用户角色，不能创建管理员
        if user_in.role == "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足：不能创建管理员账号"
            )
    
    # 创建用户
    user = user_utils.create_user(db=db, user_in=user_in)
    
    return user.to_dict()


@router.get("/roles", response_model=List[RoleResponse])
async def read_roles(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    获取所有角色
    """
    roles = user_utils.get_roles(db)
    return [role.to_dict() for role in roles]


@router.get("/{user_id}", response_model=UserResponse)
async def read_user(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_users),
):
    """
    通过ID获取用户
    """
    user = user_utils.get_user(db=db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 部门权限控制
    if current_user.role != "admin":
        # 部门管理员只能查看自己部门的用户
        if user.department != current_user.department:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足：只能查看本部门用户"
            )
    
    return user.to_dict()


@router.put("/{user_id}", response_model=UserResponse)
async def update_user_info(
    user_id: str,
    user_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_users),
):
    """
    更新用户信息
    """
    user = user_utils.get_user(db=db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 不允许更改admin用户名
    if user.username == "admin" and user_in.username and user_in.username != "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不能修改管理员用户名"
        )
    
    # 部门权限控制
    if current_user.role != "admin":
        # 部门管理员只能修改自己部门的用户
        if user.department != current_user.department:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足：只能管理本部门用户"
            )
        
        # 部门管理员不能修改用户的部门
        if user_in.department and user_in.department != user.department:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足：不能修改用户所属部门"
            )
    
    # 更新用户
    updated_user = user_utils.update_user(db=db, user=user, user_in=user_in)
    return updated_user.to_dict()


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_api(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_users),
):
    """
    删除用户
    """
    user = user_utils.get_user(db=db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 不允许删除admin用户
    if user.username == "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不能删除管理员用户"
        )
    
    # 部门权限控制
    if current_user.role != "admin":
        # 部门管理员只能删除自己部门的用户
        if user.department != current_user.department:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足：只能管理本部门用户"
            )
    
    # 删除用户
    user_utils.delete_user(db=db, user_id=user_id)


@router.post("/{user_id}/reset-password")
async def reset_password(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_users),
):
    """
    重置用户密码
    """
    user = user_utils.get_user(db=db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 部门权限控制
    if current_user.role != "admin":
        # 部门管理员只能重置本部门用户密码
        if user.department != current_user.department:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足：只能重置本部门用户密码"
            )
    
    # 重置密码
    temp_password = user_utils.reset_user_password(db=db, user=user)
    
    return {
        "code": 200,
        "message": "密码已重置",
        "data": {
            "tempPassword": temp_password
        }
    }

@router.get("/test-auth")
async def test_authentication(
    current_user: User = Depends(get_current_active_user),
):
    """
    测试认证端点
    
    检查当前用户是否已认证
    """
    return {
        "code": 200,
        "message": "认证成功",
        "data": {
            "user_id": current_user.id,
            "username": current_user.username,
            "role": current_user.role
        }
    } 
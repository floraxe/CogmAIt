import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from datetime import datetime, timedelta, timezone

# ========================== 全局自动生效的Fixture：彻底隔离所有外部依赖 ==========================
@pytest.fixture(autouse=True)
def mock_all_dependencies():
    """
    自动为所有测试用例 Mock 所有外部依赖，彻底隔离真实环境
    核心修正：修复get_cn_datetime的Mock位置、返回真实datetime对象
    """
    # 1. Mock 密码上下文（彻底避免真实 bcrypt 调用，从根源解决 72 字节报错）
    mock_pwd_context = MagicMock()
    mock_pwd_context.hash.return_value = "mock_hashed_pwd"
    mock_pwd_context.verify.return_value = True
    with patch("app.utils.security.pwd_context", mock_pwd_context):
        
        # 2. Mock 数据库模型，避免 ORM 映射初始化失败
        mock_user_model = MagicMock()
        mock_token_data = MagicMock()
        with patch("app.models.user.User", mock_user_model), \
             patch("app.schemas.user.TokenData", mock_token_data):
            
            # 3. Mock 数据库会话 SessionLocal（解决 get_user_by_token 内导入的属性错误）
            mock_db_session = MagicMock()
            mock_session_local = MagicMock(return_value=mock_db_session)
            with patch("app.db.session.SessionLocal", mock_session_local):
                
                # 4. 【核心修正】Mock get_cn_datetime：修正位置、返回真实datetime对象
                # 业务代码在app.utils.security中使用，因此patch该位置
                mock_cn_time = datetime(2026, 4, 3, 16, 30, 0, tzinfo=timezone(timedelta(hours=8)))
                with patch("app.utils.security.get_cn_datetime", return_value=mock_cn_time):
                    
                    # 所有依赖 Mock 完成，向测试用例暴露 Mock 对象和返回值
                    yield {
                        "pwd_context": mock_pwd_context,
                        "session_local": mock_session_local,
                        "db_session": mock_db_session,
                        "get_cn_time": mock_cn_time  # 暴露真实datetime对象，用于断言
                    }

# ========================== 导入被测模块（所有依赖已 Mock，不会触发真实调用） ==========================
from app.utils.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_user_by_username,
    authenticate_user,
    get_permissions_for_role,
    check_permission,
    update_user_last_login,
    get_current_user,
    get_current_active_user,
    get_user_by_token
)
from app.core.config import settings

# ========================== 通用测试 Fixture ==========================
@pytest.fixture
def mock_db():
    """Mock 数据库会话，用于需要 db 参数的测试"""
    db = MagicMock()
    query_mock = MagicMock()
    query_mock.first.return_value = None
    db.query.return_value = query_mock
    return db

@pytest.fixture
def mock_user():
    """Mock 用户对象，完全隔离真实密码哈希"""
    user = MagicMock()
    user.username = "testuser"
    user.hashed_password = "mock_hashed_pwd"
    user.is_active = True
    user.role = "admin"
    user.last_login = None
    return user

# ========================== 工具函数测试 ==========================
def test_verify_password(mock_all_dependencies):
    """测试密码验证：基于 Mock 密码上下文，无真实 bcrypt 调用"""
    # 测试验证成功场景
    mock_all_dependencies["pwd_context"].verify.return_value = True
    assert verify_password("123456", "mock_hashed_pwd") is True
    
    # 测试验证失败场景
    mock_all_dependencies["pwd_context"].verify.return_value = False
    assert verify_password("wrong_pwd", "mock_hashed_pwd") is False

def test_get_password_hash(mock_all_dependencies):
    """测试密码哈希：基于 Mock 密码上下文，彻底避免 72 字节限制"""
    result = get_password_hash("any_length_password")
    assert result == "mock_hashed_pwd"
    mock_all_dependencies["pwd_context"].hash.assert_called_once()

def test_create_access_token(mock_all_dependencies):
    """测试 JWT 令牌生成"""
    test_data = {"sub": "testuser"}
    token = create_access_token(data=test_data)
    assert token is not None
    assert isinstance(token, str)

# ========================== 用户查询测试 ==========================
def test_get_user_by_username_found(mock_db, mock_user):
    """测试通过用户名获取用户：用户存在场景"""
    mock_db.query().filter().first.return_value = mock_user
    result = get_user_by_username(mock_db, "testuser")
    assert result == mock_user

def test_get_user_by_username_not_found(mock_db):
    """测试通过用户名获取用户：用户不存在场景"""
    mock_db.query().filter().first.return_value = None
    result = get_user_by_username(mock_db, "nonexistent")
    assert result is None

# ========================== 用户认证测试 ==========================
def test_authenticate_user_success(mock_db, mock_user, mock_all_dependencies):
    """测试用户认证：用户名+密码正确场景"""
    mock_db.query().filter().first.return_value = mock_user
    mock_all_dependencies["pwd_context"].verify.return_value = True
    
    result = authenticate_user(mock_db, "testuser", "123456")
    assert result == mock_user

def test_authenticate_user_wrong_password(mock_db, mock_user, mock_all_dependencies):
    """测试用户认证：密码错误场景"""
    mock_db.query().filter().first.return_value = mock_user
    mock_all_dependencies["pwd_context"].verify.return_value = False
    
    result = authenticate_user(mock_db, "testuser", "wrong_pwd")
    assert result is False

def test_authenticate_user_nonexistent(mock_db):
    """测试用户认证：用户不存在场景"""
    mock_db.query().filter().first.return_value = None
    result = authenticate_user(mock_db, "nonexistent", "any_pwd")
    assert result is False

# ========================== 权限校验测试 ==========================
def test_get_permissions_for_role():
    """测试角色权限列表获取"""
    assert get_permissions_for_role("admin") == ["*"]
    assert get_permissions_for_role("user") == ["agent:read", "agent:use"]
    assert get_permissions_for_role("guest") == ["agent:read"]
    assert get_permissions_for_role("unknown_role") == []

def test_check_permission_admin():
    """测试权限校验：管理员拥有所有权限"""
    admin_user = MagicMock(role="admin", is_active=True)
    assert check_permission(admin_user, "any:permission") is True
    assert check_permission(admin_user, "*") is True

def test_check_permission_user_specific():
    """测试权限校验：普通用户仅拥有指定权限"""
    normal_user = MagicMock(role="user", is_active=True)
    assert check_permission(normal_user, "agent:read") is True
    assert check_permission(normal_user, "model:*") is False

def test_check_permission_inactive_user():
    """测试权限校验：非活跃用户无任何权限"""
    inactive_user = MagicMock(role="user", is_active=False)
    assert check_permission(inactive_user, "agent:read") is False

# ========================== 登录时间更新测试（核心修正版） ==========================
def test_update_user_last_login(mock_db, mock_user, mock_all_dependencies):
    """测试更新用户最后登录时间：修正断言逻辑、Mock位置"""
    # 调用被测函数
    update_user_last_login(mock_db, mock_user)
    
    # 验证数据库操作被正确调用
    mock_db.add.assert_called_once_with(mock_user)
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once_with(mock_user)
    
    # 【核心修正】断言mock_user.last_login等于get_cn_datetime的返回值（真实datetime对象）
    assert mock_user.last_login == mock_all_dependencies["get_cn_time"]

# ========================== 当前用户获取测试 ==========================
@pytest.mark.asyncio
async def test_get_current_user_valid(mock_db, mock_user):
    """测试获取当前用户：有效 Token 场景"""
    valid_token = create_access_token({"sub": "testuser"})
    mock_db.query().filter().first.return_value = mock_user
    
    result = await get_current_user(db=mock_db, token=valid_token)
    assert result == mock_user

@pytest.mark.asyncio
async def test_get_current_user_invalid_token(mock_db):
    """测试获取当前用户：无效 Token 场景"""
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(db=mock_db, token="invalid.token.here")
    assert exc_info.value.status_code == 401

@pytest.mark.asyncio
async def test_get_current_user_inactive(mock_db, mock_user):
    """测试获取当前用户：用户已禁用场景"""
    valid_token = create_access_token({"sub": "inactive_user"})
    mock_user.is_active = False
    mock_db.query().filter().first.return_value = mock_user
    
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(db=mock_db, token=valid_token)
    assert exc_info.value.status_code == 403

# ========================== 活跃用户校验测试 ==========================
@pytest.mark.asyncio
async def test_get_current_active_user_valid():
    """测试获取当前活跃用户：用户活跃场景"""
    active_user = MagicMock(is_active=True)
    result = await get_current_active_user(active_user)
    assert result == active_user

@pytest.mark.asyncio
async def test_get_current_active_user_inactive():
    """测试获取当前活跃用户：用户已禁用场景"""
    inactive_user = MagicMock(is_active=False)
    with pytest.raises(HTTPException) as exc_info:
        await get_current_active_user(inactive_user)
    assert exc_info.value.status_code == 403

# ========================== Token 获取用户测试 ==========================
@pytest.mark.asyncio
async def test_get_user_by_token_valid(mock_user, mock_all_dependencies):
    """测试通过 Token 获取用户：有效 Token 场景（彻底解决 SessionLocal 错误）"""
    valid_token = create_access_token({"sub": "testuser"})
    mock_db = mock_all_dependencies["db_session"]
    mock_db.query().filter().first.return_value = mock_user
    
    result = await get_user_by_token(token=valid_token)
    assert result == mock_user
    # 验证 SessionLocal 被正确调用、会话被关闭
    mock_all_dependencies["session_local"].assert_called_once()
    mock_db.close.assert_called_once()

@pytest.mark.asyncio
async def test_get_user_by_token_invalid():
    """测试通过 Token 获取用户：无效 Token 场景"""
    result = await get_user_by_token(token="invalid.token.here")
    assert result is None
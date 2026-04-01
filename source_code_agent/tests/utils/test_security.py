import jwt
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
from sqlalchemy.orm import Session

from app.core.config import settings
from app.utils.security import (
    get_user_by_token,
    update_user_last_login,
    check_permission,
    get_current_user
)
from app.models.user import User

# 模拟User类
class MockUser:
    def __init__(self, username="test", is_active=True, role="admin"):
        self.username = username
        self.is_active = is_active
        self.role = role
        self.last_login = None

# 模拟 get_permissions_for_role 函数，根据角色返回对应权限
def mock_get_permissions_for_role(role: str):
    # 管理员：通配符权限 + 模型读写
    if role == "admin":
        return ["*", "model:read", "model:write"]
    # 普通用户：仅模型读权限
    elif role == "user":
        return ["model:read"]
    # 其他角色：无权限
    return []


# 模拟get_user_by_username
def mock_get_user_by_username(db, username):
    if username == "test":
        return MockUser(username=username, is_active=True)
    elif username == "inactive":
        return MockUser(username=username, is_active=False)
    return None


# ------------------------------
# 测试 get_user_by_token
# ------------------------------
@pytest.mark.asyncio
async def test_get_user_by_token_happy_path():
    """正常路径：有效token，获取活跃用户"""
    payload = {"sub": "test", "exp": datetime.utcnow() + timedelta(minutes=30)}
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

    # 仅mock get_user_by_username，无需mock SessionLocal
    with patch("app.utils.security.get_user_by_username", side_effect=mock_get_user_by_username):
        # 异步函数必须await
        user = await get_user_by_token(token)

    assert user is not None
    assert user.username == "test"
    assert user.is_active is True


@pytest.mark.asyncio
async def test_get_user_by_token_unhappy_expired_token():
    """异常路径：过期token，返回None"""
    payload = {"sub": "test", "exp": datetime.utcnow() - timedelta(minutes=30)}
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

    with patch("app.utils.security.get_user_by_username", side_effect=mock_get_user_by_username):
        user = await get_user_by_token(token)

    assert user is None


@pytest.mark.asyncio
async def test_get_user_by_token_unhappy_inactive_user():
    """异常路径：非活跃用户，返回None"""
    payload = {"sub": "inactive", "exp": datetime.utcnow() + timedelta(minutes=30)}
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

    with patch("app.utils.security.get_user_by_username", side_effect=mock_get_user_by_username):
        user = await get_user_by_token(token)

    assert user is None


@pytest.mark.asyncio
async def test_get_user_by_token_unhappy_invalid_token():
    """异常路径：非法token，返回None"""
    token = "invalid_token"
    user = await get_user_by_token(token)
    assert user is None

# ------------------------------
# 测试 update_user_last_login
# ------------------------------
def test_update_user_last_login_happy_path():
    """正常路径：更新用户最后登录时间"""
    mock_db = Mock(spec=Session)
    mock_user = MockUser()

    with patch("app.utils.security.get_cn_datetime", return_value=datetime(2025, 1, 1)):
        updated_user = update_user_last_login(mock_db, mock_user)

    assert updated_user.last_login == datetime(2025, 1, 1)
    mock_db.add.assert_called_once_with(mock_user)
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once_with(mock_user)


def test_update_user_last_login_unhappy_none_user():
    """异常路径：传入None用户，抛出异常（抓Issue：无空校验）"""
    mock_db = Mock(spec=Session)
    with pytest.raises(AttributeError):
        update_user_last_login(mock_db, None)


def test_update_user_last_login_unhappy_none_db():
    """异常路径：传入None数据库，抛出异常（抓Issue：无空校验）"""
    mock_user = MockUser()
    with pytest.raises(AttributeError):
        update_user_last_login(None, mock_user)

# ------------------------------
# 测试 check_permission 函数
# ------------------------------
def test_check_permission_happy_admin_wildcard():
    """正常路径：管理员通配符权限，直接放行"""
    mock_user = MockUser(role="admin", is_active=True)
    # 正确 mock get_permissions_for_role
    with patch("app.utils.security.get_permissions_for_role", side_effect=mock_get_permissions_for_role):
        result = check_permission(mock_user, "any_permission")

    assert result is True


def test_check_permission_happy_user_specific():
    """正常路径：用户有具体权限，放行"""
    mock_user = MockUser(role="user", is_active=True)
    with patch("app.utils.security.get_permissions_for_role", side_effect=mock_get_permissions_for_role):
        result = check_permission(mock_user, "model:read")

    assert result is True


def test_check_permission_unhappy_user_no_permission():
    """异常路径：用户无权限，返回False"""
    mock_user = MockUser(role="user", is_active=True)
    with patch("app.utils.security.get_permissions_for_role", side_effect=mock_get_permissions_for_role):
        result = check_permission(mock_user, "model:write")

    assert result is False


def test_check_permission_unhappy_resource_wildcard():
    """异常路径：资源级别权限校验"""
    mock_user = MockUser(role="admin", is_active=True)
    with patch("app.utils.security.get_permissions_for_role", side_effect=mock_get_permissions_for_role):
        result = check_permission(mock_user, "model:read")

    assert result is True


def test_check_permission_unhappy_inactive_user():
    """异常路径：非活跃用户，返回False"""
    mock_user = MockUser(is_active=False)
    result = check_permission(mock_user, "model:read")
    assert result is False



# ------------------------------
# 测试 get_current_user
# ------------------------------
@pytest.mark.asyncio
async def test_get_current_user_happy_path():
    """正常路径：有效token，获取当前用户"""
    payload = {"sub": "test", "exp": datetime.utcnow() + timedelta(minutes=30)}
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

    mock_db = Mock(spec=Session)
    with patch("app.utils.security.get_user_by_username", side_effect=mock_get_user_by_username):
        user = await get_current_user(db=mock_db, token=token)

    assert user is not None
    assert user.username == "test"


@pytest.mark.asyncio
async def test_get_current_user_unhappy_expired_token():
    """异常路径：过期token，抛出401（抓Issue：无过期校验）"""
    payload = {"sub": "test", "exp": datetime.utcnow() - timedelta(minutes=30)}
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

    mock_db = Mock(spec=Session)
    with pytest.raises(Exception) as excinfo:
        await get_current_user(db=mock_db, token=token)

    # 若代码无过期校验，这里不会抛出异常（Issue）；加校验后应抛出401
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_unhappy_inactive_user():
    """异常路径：非活跃用户，抛出403"""
    payload = {"sub": "inactive", "exp": datetime.utcnow() + timedelta(minutes=30)}
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

    mock_db = Mock(spec=Session)
    with patch("app.utils.security.get_user_by_username", side_effect=mock_get_user_by_username):
        with pytest.raises(Exception) as excinfo:
            await get_current_user(db=mock_db, token=token)

    assert excinfo.value.status_code == 403


import pytest
from datetime import datetime, timedelta

from app.db.base import (
    CST_TIMEZONE,
    get_cn_datetime,
    set_time_provider,
    reset_time_provider,
)


@pytest.fixture(autouse=True)
def restore_time_provider():
    """
    每个用例前后重置一次时间提供者，防止测试之间相互污染。
    """
    reset_time_provider()
    yield
    reset_time_provider()


def test_get_cn_datetime_default_uses_cst_timezone():
    """
    默认情况下，get_cn_datetime 返回当前东八区时间，且 tzinfo 为 CST_TIMEZONE。
    """
    before = datetime.now(CST_TIMEZONE)
    value = get_cn_datetime()
    after = datetime.now(CST_TIMEZONE)

    assert value.tzinfo == CST_TIMEZONE
    # 时间应落在调用前后窗口内，保证不是奇怪的固定值
    assert before <= value <= after


def test_set_time_provider_allows_injecting_fixed_time():
    """
    通过 set_time_provider 可以注入一个固定时间，便于在业务代码中写稳定的断言。
    """
    fixed = datetime(2025, 1, 1, 12, 0, 0, tzinfo=CST_TIMEZONE)

    set_time_provider(lambda: fixed)

    assert get_cn_datetime() == fixed


def test_reset_time_provider_restores_default_behavior():
    """
    reset_time_provider 会把时间提供者恢复为“当前系统时间”的实现。
    """
    fixed = datetime(2024, 6, 1, 8, 0, 0, tzinfo=CST_TIMEZONE)

    set_time_provider(lambda: fixed)
    assert get_cn_datetime() == fixed

    reset_time_provider()
    now = get_cn_datetime()

    # 恢复后不再是固定时间，而是在当前时间附近
    assert now != fixed
    # 给一个宽松窗口，避免 CI 慢机器误判
    assert fixed - timedelta(days=3650) < now < fixed + timedelta(days=3650)


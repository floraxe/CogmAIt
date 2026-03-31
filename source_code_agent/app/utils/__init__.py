"""
通用工具函数模块
"""
from datetime import datetime, timedelta, timezone

# 定义东八区时区（中国标准时间 UTC+8）
CST_TIMEZONE = timezone(timedelta(hours=8))

def utc_to_cst(utc_dt: datetime) -> datetime:
    """
    将UTC时间转换为中国标准时间（东八区，UTC+8）
    
    参数:
        utc_dt (datetime): UTC时间
    
    返回:
        datetime: 中国标准时间
    """
    if utc_dt is None:
        return None
    
    # 确保输入时间是UTC时间
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    
    # 转换为东八区时间
    return utc_dt.astimezone(CST_TIMEZONE)

def format_datetime(dt: datetime, include_timezone: bool = False) -> str:
    """
    格式化日期时间为ISO 8601格式字符串
    
    参数:
        dt (datetime): 日期时间对象
        include_timezone (bool): 是否包含时区信息
    
    返回:
        str: 格式化的日期时间字符串
    """
    if dt is None:
        return ""  # 返回空字符串而不是None
    
    # 如果没有时区信息，先转换为带时区信息的中国标准时间
    if dt.tzinfo is None:
        dt = utc_to_cst(dt)
    
    # 返回ISO格式时间
    if include_timezone:
        return dt.isoformat()
    else:
        # 不包含时区信息的格式
        return dt.strftime('%Y-%m-%dT%H:%M:%S') 
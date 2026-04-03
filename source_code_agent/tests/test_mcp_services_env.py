"""
``mcp_services`` 中与环境变量相关的纯函数单元测试（不启动网络或数据库）。
"""
import pytest

from mcp_services import build_mcp_db_config, resolve_mcp_server_bind


def test_build_mcp_db_config_defaults():
    cfg = build_mcp_db_config({})
    assert cfg == {
        "host": "localhost",
        "user": "root",
        "password": "xkkxkkxkk",
        "database": "cogmait",
    }


def test_build_mcp_db_config_from_env():
    env = {
        "DB_HOST": "db.example",
        "DB_USER": "u",
        "DB_PASSWORD": "p",
        "DB_DATABASE": "mydb",
    }
    cfg = build_mcp_db_config(env)
    assert cfg == {
        "host": "db.example",
        "user": "u",
        "password": "p",
        "database": "mydb",
    }


def test_resolve_mcp_server_bind_defaults():
    host, port = resolve_mcp_server_bind({})
    assert host == "0.0.0.0"
    assert port == 8001


def test_resolve_mcp_server_bind_from_env():
    host, port = resolve_mcp_server_bind({"MCP_HOST": "127.0.0.1", "MCP_PORT": "9000"})
    assert host == "127.0.0.1"
    assert port == 9000

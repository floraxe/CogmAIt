import json
from unittest.mock import mock_open

from app.utils.config import _ConfigLoader, DEFAULT_CONFIG


def test_loader_merge_defaults_when_file_exists():
    """
    文件存在时，应加载文件并补齐 DEFAULT_CONFIG 缺失字段。
    """
    partial = {"neo4j": {"uri": "bolt://x:7687"}, "openai": {"api_key": "k"}}
    loader = _ConfigLoader(
        config_path="config.json",
        default_config=DEFAULT_CONFIG,
        environ={},
        path_exists=lambda _: True,
        open_fn=mock_open(read_data=json.dumps(partial)),
    )

    result = loader.load()
    assert result["neo4j"]["uri"] == "bolt://x:7687"
    assert result["neo4j"]["username"] == DEFAULT_CONFIG["neo4j"]["username"]
    assert result["openai"]["model"] == DEFAULT_CONFIG["openai"]["model"]


def test_loader_fallback_to_default_and_apply_env_override():
    """
    文件不存在时，应回落到默认配置，并应用环境变量覆盖。
    """
    env = {
        "NEO4J_URI": "bolt://env:7687",
        "NEO4J_USERNAME": "env_user",
        "OPENAI_API_KEY": "sk-env",
    }
    loader = _ConfigLoader(
        config_path="missing.json",
        default_config=DEFAULT_CONFIG,
        environ=env,
        path_exists=lambda _: False,
    )

    result = loader.load()
    assert result["neo4j"]["uri"] == "bolt://env:7687"
    assert result["neo4j"]["username"] == "env_user"
    assert result["neo4j"]["password"] == DEFAULT_CONFIG["neo4j"]["password"]
    assert result["openai"]["api_key"] == "sk-env"


def test_loader_fallback_when_file_invalid_json():
    """
    文件存在但解析失败时，应记录错误并回落到默认配置。
    """
    loader = _ConfigLoader(
        config_path="broken.json",
        default_config=DEFAULT_CONFIG,
        environ={},
        path_exists=lambda _: True,
        open_fn=mock_open(read_data="{invalid-json"),
    )

    result = loader.load()
    assert result["neo4j"]["uri"] == DEFAULT_CONFIG["neo4j"]["uri"]
    assert result["openai"]["model"] == DEFAULT_CONFIG["openai"]["model"]


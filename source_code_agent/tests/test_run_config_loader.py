import os
from unittest.mock import patch, mock_open

from source_code_agent import run


def test_resolve_config_path_prefers_existing_env_path(tmp_path):
    """
    当 CONFIG_FILE 环境变量指向的路径存在时，应优先使用该路径。
    """
    cfg = tmp_path / "custom.json"
    cfg.write_text("{}", encoding="utf-8")

    result = run.resolve_config_path(str(cfg), base_dir=str(tmp_path))
    assert result == str(cfg)


def test_resolve_config_path_fallbacks_to_default_when_env_missing(tmp_path):
    """
    当 CONFIG_FILE 未提供或路径不存在时，退回到 base_dir/config.json。
    """
    base_dir = str(tmp_path)
    expected = os.path.join(base_dir, "config.json")

    result = run.resolve_config_path(None, base_dir=base_dir)
    assert result == expected

    result = run.resolve_config_path("non-existent.json", base_dir=base_dir)
    assert result == expected


def test_load_server_config_returns_dict_on_success():
    """
    load_server_config 在文件内容为合法 JSON 时，应返回解析后的字典。
    """
    data = {"host": "127.0.0.1", "port": 9000, "reload": False}
    with patch("builtins.open", mock_open(read_data='{"host": "127.0.0.1", "port": 9000, "reload": false}')):
        cfg = run.load_server_config("dummy.json")

    assert cfg == data


def test_load_server_config_handles_io_errors_gracefully():
    """
    当读取文件失败（不存在/权限等）时，返回空字典，不抛异常。
    """
    with patch("builtins.open", side_effect=FileNotFoundError("not found")):
        cfg = run.load_server_config("missing.json")

    assert cfg == {}


def test_get_uvicorn_options_uses_defaults_when_keys_missing():
    """
    get_uvicorn_options 应在缺失配置键时使用合理的默认值。
    """
    host, port, reload_flag = run.get_uvicorn_options({})
    assert host == "0.0.0.0"
    assert port == 8000
    assert reload_flag is True


def test_get_uvicorn_options_reads_values_from_config():
    """
    get_uvicorn_options 应优先使用配置中提供的值。
    """
    host, port, reload_flag = run.get_uvicorn_options(
        {"host": "127.0.0.1", "port": 9000, "reload": False}
    )
    assert host == "127.0.0.1"
    assert port == 9000
    assert reload_flag is False


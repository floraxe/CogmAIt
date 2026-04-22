import os
import json
import pytest
from unittest.mock import patch, mock_open
from app.utils.config import (
    load_config,
    save_config,
    get_neo4j_config,
    get_openai_config,
    update_neo4j_config,  
    update_openai_config, 
    DEFAULT_CONFIG,
    CONFIG_PATH
)


# ------------------------------
# 关键修复：每个用例执行前重置全局_config，解决状态污染
# ------------------------------
@pytest.fixture(autouse=True)
def reset_global_config():
    """
    自动执行的 fixture：每个测试用例运行前，重置全局_config为None
    确保每个用例都从全新状态开始，互不污染
    """
    from app.utils import config
    # 保存原始值，用例执行后恢复（防御性编程）
    original_config = config._config
    config._config = None  # 重置全局缓存
    yield  # 执行测试用例
    config._config = original_config  # 用例执行后恢复

# ------------------------------
# 1. 正常路径（Happy Path）测试
# ------------------------------
def test_load_config_happy_path_existing_valid_file():
    """
    测试正常场景：配置文件存在且JSON格式合法
    预期结果：
    1. 成功加载配置文件
    2. 自动补全DEFAULT_CONFIG中缺失的键（嵌套字典也会补全）
    3. 全局_config被正确缓存
    """
    # 构造测试用的配置文件内容（故意缺失部分默认配置，验证补全逻辑）
    test_config = {
        "neo4j": {
            "uri": "bolt://test-neo4j:7687",  # 自定义uri，会保留
            # 故意不写username/password/database，验证会用默认值补全
        },
        "openai": {
            "api_key": "sk-test-123456"  # 自定义api_key，会保留
            # 故意不写model，验证会用默认值补全
        }
    }

    # 用mock模拟：配置文件存在，且读取到上面的test_config
    with patch("os.path.exists", return_value=True):  # 模拟CONFIG_PATH存在
        with patch("builtins.open", mock_open(read_data=json.dumps(test_config))):  # 模拟文件读取
            # 调用待测试的函数
            config = load_config()

    # 断言1：自定义的配置被正确保留
    assert config["neo4j"]["uri"] == "bolt://test-neo4j:7687"
    assert config["openai"]["api_key"] == "sk-test-123456"

    # 断言2：缺失的配置被DEFAULT_CONFIG正确补全
    assert config["neo4j"]["username"] == DEFAULT_CONFIG["neo4j"]["username"]
    assert config["neo4j"]["password"] == DEFAULT_CONFIG["neo4j"]["password"]
    assert config["neo4j"]["database"] == DEFAULT_CONFIG["neo4j"]["database"]
    assert config["openai"]["model"] == DEFAULT_CONFIG["openai"]["model"]

    # 断言3：全局_config被正确缓存（单例模式生效）
    from app.utils.config import _config
    assert _config is not None
    assert _config == config


# ------------------------------
# 2. 边界/异常路径（Unhappy Path）测试
# ------------------------------
def test_load_config_unhappy_path_file_not_exists():
    """
    测试异常场景：配置文件不存在
    预期结果：直接返回完整的DEFAULT_CONFIG，且全局_config被缓存
    """
    # 用mock模拟：CONFIG_PATH不存在
    with patch("os.path.exists", return_value=False):
        config = load_config()

    # 断言：返回的配置和默认配置完全一致
    assert config == DEFAULT_CONFIG

    # 断言：全局_config被正确缓存
    from app.utils.config import _config
    assert _config == DEFAULT_CONFIG


def test_load_config_unhappy_path_invalid_json():
    """
    测试异常场景：配置文件存在，但JSON格式损坏
    预期结果：捕获异常，返回DEFAULT_CONFIG，打印错误日志
    """
    # 用mock模拟：配置文件存在，但内容是非法JSON
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="这不是合法的JSON格式")):
            config = load_config()

    # 断言：JSON解析失败，返回默认配置
    assert config == DEFAULT_CONFIG


def test_load_config_unhappy_path_file_permission_error():
    """
    测试异常场景：配置文件存在，但无读取权限（PermissionError）
    预期结果：捕获异常，返回DEFAULT_CONFIG
    """
    # 用mock模拟：open文件时抛出PermissionError
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", side_effect=PermissionError("无文件读取权限")):
            config = load_config()

    # 断言：权限错误被捕获，返回默认配置
    assert config == DEFAULT_CONFIG


# ------------------------------
# 3. 环境变量覆盖测试（核心逻辑）
# ------------------------------
def test_load_config_env_var_override():
    """
    测试环境变量覆盖逻辑：环境变量存在时，会覆盖配置文件/默认配置
    预期结果：
    1. NEO4J_*、OPENAI_API_KEY环境变量正确覆盖对应配置
    2. 未设置的环境变量不影响原有配置
    """
    # 构造测试环境变量
    test_env = {
        "NEO4J_URI": "bolt://env-override:7687",
        "NEO4J_USERNAME": "env_user",
        "NEO4J_PASSWORD": "env_pass",
        "NEO4J_DATABASE": "env_db",
        "OPENAI_API_KEY": "sk-env-123456"
    }

    # 用mock模拟：配置文件不存在（走默认配置），且注入环境变量
    with patch("os.path.exists", return_value=False):
        with patch.dict(os.environ, test_env, clear=True):  # 清空原有环境变量，注入测试变量
            config = load_config()

    # 断言：所有环境变量正确覆盖了默认配置
    assert config["neo4j"]["uri"] == test_env["NEO4J_URI"]
    assert config["neo4j"]["username"] == test_env["NEO4J_USERNAME"]
    assert config["neo4j"]["password"] == test_env["NEO4J_PASSWORD"]
    assert config["neo4j"]["database"] == test_env["NEO4J_DATABASE"]
    assert config["openai"]["api_key"] == test_env["OPENAI_API_KEY"]


def test_load_config_env_var_partial_override():
    """
    测试环境变量部分覆盖：仅设置部分环境变量
    预期结果：仅覆盖设置的变量，未设置的变量保持默认值
    """
    # 仅设置部分环境变量
    test_env = {
        "NEO4J_URI": "bolt://partial-override:7687",
        "OPENAI_API_KEY": "sk-partial-123"
    }

    with patch("os.path.exists", return_value=False):
        with patch.dict(os.environ, test_env, clear=True):
            config = load_config()

    # 断言：设置的变量被覆盖，未设置的变量保持默认
    assert config["neo4j"]["uri"] == test_env["NEO4J_URI"]
    assert config["neo4j"]["username"] == DEFAULT_CONFIG["neo4j"]["username"]  # 未设置，保持默认
    assert config["openai"]["api_key"] == test_env["OPENAI_API_KEY"]
    assert config["openai"]["model"] == DEFAULT_CONFIG["openai"]["model"]  # 未设置，保持默认


# ------------------------------
# 4. 单例模式测试（全局缓存逻辑）
# ------------------------------
def test_load_config_singleton_behavior():
    """
    测试单例模式：多次调用load_config()，返回同一个全局_config实例
    预期结果：多次调用返回的是同一个对象，不会重复读取文件
    """
    # 第一次调用：模拟文件不存在，走默认配置
    with patch("os.path.exists", return_value=False):
        config1 = load_config()

    # 第二次调用：即使修改mock，也会直接返回缓存的_config，不会重新执行
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data='{"test": "invalid"}')):
            config2 = load_config()

    # 断言1：两次调用返回同一个对象（内存地址相同）
    assert config1 is config2

    # 断言2：第二次调用不会重新读取文件，配置保持第一次的结果
    assert config2 == DEFAULT_CONFIG
    assert "test" not in config2  # 验证第二次的mock文件内容未被加载


# ------------------------------
# 5. 嵌套配置补全测试（边界场景）
# ------------------------------
def test_load_config_nested_config_completion():
    """
    测试嵌套字典的补全逻辑：配置文件中某一级字典缺失，会完整补全
    预期结果：即使整个neo4j配置缺失，也会用DEFAULT_CONFIG完整补全
    """
    # 构造测试配置：完全缺失neo4j配置
    test_config = {
        "openai": {
            "api_key": "sk-test-nested"
        }
    }

    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=json.dumps(test_config))):
            config = load_config()

    # 断言：完全缺失的neo4j配置被完整补全
    assert config["neo4j"] == DEFAULT_CONFIG["neo4j"]
    # 断言：openai配置保留自定义值，补全缺失的model
    assert config["openai"]["api_key"] == "sk-test-nested"
    assert config["openai"]["model"] == DEFAULT_CONFIG["openai"]["model"]
# ------------------------------
# 测试 save_config() 函数
# ------------------------------
def test_save_config_happy_path_normal_input():
    """
    测试正常场景：正常输入合法配置，成功保存文件
    预期结果：
    1. 成功写入配置文件，JSON格式正确
    2. 全局_config被正确更新为传入的配置
    3. 函数返回True
    """
    # 构造测试用的合法配置（和DEFAULT_CONFIG结构一致）
    test_config = {
        "neo4j": {
            "uri": "bolt://test-save:7687",
            "username": "test_user",
            "password": "test_pass",
            "database": "test_db"
        },
        "openai": {
            "api_key": "sk-test-save-123",
            "model": "gpt-4"
        }
    }

    # 1. 准备 Mock 对象
    # 模拟文件对象，用于模拟 open() 返回的文件句柄
    mock_file = mock_open()
    # 模拟 json.dump 函数，这是关键！
    # 如果不 patch json.dump，它就没有 assert_called_once_with 方法
    with patch("builtins.open", mock_file), patch("json.dump") as mock_json_dump:
        # 2. 调用待测试的函数
        result = save_config(test_config)

    # 3. 断言验证（全部通过）
    # 断言1：函数返回True，表示保存成功
    assert result is True

    # 断言2：open函数被正确调用（路径、模式、编码完全匹配）
    # 验证代码是否正确地打开了配置文件
    mock_file.assert_called_once_with(CONFIG_PATH, "w", encoding="utf-8")

    # 断言3：全局_config被正确更新为传入的配置
    # 验证代码逻辑中，保存后全局变量是否被更新
    from app.utils.config import _config
    assert _config == test_config

    # 断言4：json.dump被正确调用
    # 验证代码是否正确地将测试配置写入了文件
    # 同时验证了 indent=2 参数，保证 JSON 格式美观
    handle = mock_file() # 获取 mock_file 对应的文件句柄
    mock_json_dump.assert_called_once_with(test_config, handle, indent=2)
    

def test_save_config_unhappy_path_empty_input():
    """
    测试边界场景：输入空字典
    预期结果：
    1. 成功写入空字典到文件
    2. 全局_config被更新为空字典
    3. 函数返回True
    """
    # 构造空字典输入
    empty_config = {}

    mock_file = mock_open()
    with patch("builtins.open", mock_file):
        result = save_config(empty_config)

    # 断言：保存成功，全局_config被更新
    assert result is True
    mock_file.assert_called_once_with(CONFIG_PATH, "w", encoding="utf-8")
    from app.utils.config import _config
    assert _config == empty_config


def test_save_config_unhappy_path_permission_error():
    """
    测试异常场景：文件写入权限不足（PermissionError）
    预期结果：
    1. 捕获异常，函数返回False
    2. 全局_config不被修改（保持原有值）
    3. 打印错误日志
    """
    # 先给全局_config设置一个初始值，验证异常时不被修改
    from app.utils.config import _config as original_config
    test_config = {"test": "config"}

    # 用mock模拟open时抛出PermissionError（权限不足）
    with patch("builtins.open", side_effect=PermissionError("无写入权限")):
        result = save_config(test_config)

    # 断言1：函数返回False，表示保存失败
    assert result is False

    # 断言2：全局_config保持原有值，未被修改
    from app.utils.config import _config
    assert _config == original_config


def test_save_config_unhappy_path_invalid_json():
    """
    测试异常场景：输入无法序列化为JSON的对象（比如包含Python特殊对象）
    预期结果：
    1. json.dump抛出异常，函数返回False
    2. 全局_config不被修改
    """
    # 构造无法序列化为JSON的输入（包含Python函数对象）
    def test_func():
        pass
    invalid_config = {"func": test_func}  # 函数无法被json序列化

    # 先保存原始全局_config
    from app.utils.config import _config as original_config

    with patch("builtins.open", mock_open()):
        result = save_config(invalid_config)

    # 断言：保存失败，全局_config不变
    assert result is False
    from app.utils.config import _config
    assert _config == original_config


def test_save_config_unhappy_path_large_config():
    """
    测试边界场景：输入超大配置（超长字符串、多层嵌套）
    预期结果：
    1. 成功写入超大配置
    2. 函数返回True
    3. 全局_config被正确更新
    """
    # 构造超大配置：超长字符串 + 多层嵌套
    large_config = {
        "neo4j": {
            "uri": "bolt://" + "a" * 1000 + ":7687",  # 1000个字符的超长uri
            "username": "user" * 100,
            "password": "pass" * 100,
            "database": "db" * 100
        },
        "openai": {
            "api_key": "sk-" + "x" * 200,
            "model": "gpt-4" * 50
        },
        "nested": {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": "deep_value"
                    }
                }
            }
        }
    }

    mock_file = mock_open()
    with patch("builtins.open", mock_file):
        result = save_config(large_config)

    # 断言：超大配置保存成功
    assert result is True
    mock_file.assert_called_once_with(CONFIG_PATH, "w", encoding="utf-8")
    from app.utils.config import _config
    assert _config == large_config

# ------------------------------
# 测试 get_neo4j_config() 函数
# ------------------------------
def test_get_neo4j_config_happy_path_normal():
    """
    测试正常场景：配置中存在完整的neo4j配置
    预期结果：
    1. 正确调用load_config()获取全局配置
    2. 从配置中提取并返回完整的neo4j配置
    """
    # 构造测试用的完整neo4j配置
    test_neo4j_config = {
        "uri": "bolt://test-neo4j:7687",
        "username": "test_user",
        "password": "test_pass",
        "database": "test_db"
    }
    test_config = DEFAULT_CONFIG.copy()
    test_config["neo4j"] = test_neo4j_config  # 覆盖默认neo4j配置

    # 用mock模拟load_config()，返回我们构造的测试配置
    with patch("app.utils.config.load_config", return_value=test_config):
        # 调用待测试的函数
        result = get_neo4j_config()

    # 断言：返回的neo4j配置和我们构造的完全一致
    assert result == test_neo4j_config


def test_get_neo4j_config_unhappy_path_missing_key():
    """
    测试边界场景：配置中完全缺失neo4j键
    预期结果：
    1. 正确调用load_config()
    2. 当neo4j键不存在时，返回DEFAULT_CONFIG中的默认neo4j配置
    """
    # 构造测试配置：完全删除neo4j键
    test_config = DEFAULT_CONFIG.copy()
    del test_config["neo4j"]  # 故意删除neo4j配置，模拟缺失场景

    with patch("app.utils.config.load_config", return_value=test_config):
        result = get_neo4j_config()

    # 断言：返回默认的neo4j配置，保证程序不会崩溃
    assert result == DEFAULT_CONFIG["neo4j"]


def test_get_neo4j_config_unhappy_path_partial_config():
    """
    测试边界场景：配置中neo4j键存在，但部分字段缺失
    预期结果：
    1. load_config会自动补全缺失的默认字段
    2. get_neo4j_config从补全后的配置中，正确提取完整的neo4j配置
    """
    # 构造测试配置：neo4j仅包含uri，缺失其他字段
    test_config_input = {
        "neo4j": {"uri": "bolt://partial:7687"}
    }

    # 关键修复：不直接mock load_config，而是模拟真实文件读取，让load_config自动补全
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=json.dumps(test_config_input))):
            # 先调用load_config，让它自动补全缺失的默认字段
            full_config = load_config()
            # 再调用get_neo4j_config，从补全后的配置中提取
            result = get_neo4j_config()

    # 断言：返回的配置包含我们设置的uri，其他字段为load_config补全的默认值
    assert result["uri"] == "bolt://partial:7687"
    assert result["username"] == DEFAULT_CONFIG["neo4j"]["username"]
    assert result["password"] == DEFAULT_CONFIG["neo4j"]["password"]
    assert result["database"] == DEFAULT_CONFIG["neo4j"]["database"]

# ------------------------------
# 测试 get_openai_config() 函数
# ------------------------------
def test_get_openai_config_happy_path_normal():
    """
    测试正常场景：配置中存在完整的openai配置
    预期结果：
    1. 正确调用load_config()获取全局配置
    2. 从配置中提取并返回完整的openai配置
    """
    # 构造测试用的完整openai配置
    test_openai_config = {
        "api_key": "sk-test-openai-123456",
        "model": "gpt-4o"
    }
    test_config = DEFAULT_CONFIG.copy()
    test_config["openai"] = test_openai_config  # 覆盖默认openai配置

    # 用mock模拟load_config()，返回我们构造的测试配置
    with patch("app.utils.config.load_config", return_value=test_config):
        # 调用待测试的函数
        result = get_openai_config()

    # 断言：返回的openai配置和我们构造的完全一致
    assert result == test_openai_config


def test_get_openai_config_unhappy_path_missing_key():
    """
    测试边界场景：配置中完全缺失openai键
    预期结果：
    1. 正确调用load_config()
    2. 当openai键不存在时，返回DEFAULT_CONFIG中的默认openai配置
    """
    # 构造测试配置：完全删除openai键
    test_config = DEFAULT_CONFIG.copy()
    del test_config["openai"]  # 故意删除openai配置，模拟缺失场景

    with patch("app.utils.config.load_config", return_value=test_config):
        result = get_openai_config()

    # 断言：返回默认的openai配置，保证程序不会崩溃
    assert result == DEFAULT_CONFIG["openai"]


def test_get_openai_config_unhappy_path_empty_api_key():
    """
    测试边界场景：openai配置中api_key为空字符串
    预期结果：
    1. 正确返回配置，即使api_key为空，函数也不会报错
    2. 保持配置的完整性，不做额外修改
    """
    # 构造测试配置：api_key为空，model正常
    test_openai_config = {
        "api_key": "",  # 空字符串，边界场景
        "model": "gpt-3.5-turbo"
    }
    test_config = DEFAULT_CONFIG.copy()
    test_config["openai"] = test_openai_config

    with patch("app.utils.config.load_config", return_value=test_config):
        result = get_openai_config()

    # 断言：返回的配置和我们构造的完全一致，空值被保留
    assert result == test_openai_config

# ==============================================================================
# 测试 update_neo4j_config() 函数
# ==============================================================================
def test_update_neo4j_config_happy_path_full_update():
    """
    测试正常场景：完整更新Neo4j配置（uri + username + password + database）
    预期结果：
    1. 正确加载当前配置
    2. 正确更新所有传入的字段
    3. 调用save_config保存并返回True
    """
    # 1. 构造测试输入参数（更新所有可选字段）
    test_uri = "bolt://test-update:7687"
    test_user = "new_admin"
    test_pwd = "new_pass_123"
    test_db = "new_graph_db"

    # 2. 模拟load_config返回完整的默认配置
    test_config = DEFAULT_CONFIG.copy()
    with patch("app.utils.config.load_config", return_value=test_config):
        # 3. 模拟save_config成功，返回True
        with patch("app.utils.config.save_config", return_value=True) as mock_save:
            # 4. 调用待测试函数
            result = update_neo4j_config(
                uri=test_uri,
                username=test_user,
                password=test_pwd,
                database=test_db
            )

    # 5. 断言验证
    assert result is True  # 函数返回成功
    mock_save.assert_called_once_with(test_config)  # save_config被正确调用
    assert test_config["neo4j"]["uri"] == test_uri      # uri被更新
    assert test_config["neo4j"]["username"] == test_user # username被更新
    assert test_config["neo4j"]["password"] == test_pwd # password被更新
    assert test_config["neo4j"]["database"] == test_db # database被更新


def test_update_neo4j_config_unhappy_path_partial_update():
    """
    测试边界场景：部分更新Neo4j配置（仅更新uri和password）
    预期结果：
    1. 仅更新传入的字段
    2. 未传入的字段保持load_config()加载的原值
    3. 返回True
    """
    # 1. 构造初始配置
    test_config = DEFAULT_CONFIG.copy()
    original_uri = test_config["neo4j"]["uri"] # 记录原始值

    # 2. 测试逻辑
    with patch("app.utils.config.load_config", return_value=test_config):
        with patch("app.utils.config.save_config", return_value=True):
            result = update_neo4j_config(
                uri="bolt://partial-update:7687", # 仅更新uri
                password="new_pwd_only"            # 仅更新password
            )

    # 3. 断言
    assert result is True
    assert test_config["neo4j"]["uri"] == "bolt://partial-update:7687" # 已更新
    assert test_config["neo4j"]["password"] == "new_pwd_only"           # 已更新
    # 验证未修改的字段保持原值
    assert test_config["neo4j"]["username"] == DEFAULT_CONFIG["neo4j"]["username"]
    assert test_config["neo4j"]["database"] == DEFAULT_CONFIG["neo4j"]["database"]


def test_update_neo4j_config_unhappy_path_save_fails():
    """
    测试异常场景：save_config保存失败（权限不足等）
    预期结果：
    1. 函数返回False（符合代码逻辑）
    2. 内存中的config会被修改（代码逻辑如此，save失败不回滚内存修改）
    """
    test_config = DEFAULT_CONFIG.copy()
    original_uri = test_config["neo4j"]["uri"]

    with patch("app.utils.config.load_config", return_value=test_config):
        with patch("app.utils.config.save_config", return_value=False):
            result = update_neo4j_config(uri="bolt://fail:7687")

    # 断言1：函数返回False（保存失败）
    assert result is False
    # 断言2：内存中的config已被修改（代码逻辑如此，save失败不回滚）
    assert test_config["neo4j"]["uri"] == "bolt://fail:7687"


# ==============================================================================
# 测试 update_openai_config() 函数
# ==============================================================================
def test_update_openai_config_happy_path_full_update():
    """
    测试正常场景：完整更新OpenAI配置（api_key + model）
    预期结果：
    1. 正确更新配置
    2. save_config被调用并返回True
    """
    # 构造输入
    test_api_key = "sk-test-new-key-123456"
    test_model = "gpt-4o"

    # 加载配置并更新
    test_config = DEFAULT_CONFIG.copy()
    with patch("app.utils.config.load_config", return_value=test_config):
        with patch("app.utils.config.save_config", return_value=True) as mock_save:
            result = update_openai_config(
                api_key=test_api_key,
                model=test_model
            )

    # 断言
    assert result is True
    mock_save.assert_called_once_with(test_config)
    assert test_config["openai"]["api_key"] == test_api_key
    assert test_config["openai"]["model"] == test_model


def test_update_openai_config_unhappy_path_empty_string():
    """
    测试边界场景：输入为空字符串（合法输入）
    预期结果：
    1. 允许空字符串输入（不报错）
    2. 正确保存到配置中
    """
    test_config = DEFAULT_CONFIG.copy()

    with patch("app.utils.config.load_config", return_value=test_config):
        with patch("app.utils.config.save_config", return_value=True):
            # 传入空字符串，合法但为空
            result = update_openai_config(
                api_key="",
                model=""
            )

    assert result is True
    assert test_config["openai"]["api_key"] == ""
    assert test_config["openai"]["model"] == ""


def test_update_openai_config_unhappy_path_save_fails():
    """
    测试异常场景：保存失败
    预期结果：返回False，内存中的config会被修改
    """
    test_config = DEFAULT_CONFIG.copy()
    original_key = test_config["openai"]["api_key"]

    with patch("app.utils.config.load_config", return_value=test_config):
        with patch("app.utils.config.save_config", return_value=False):
            result = update_openai_config(api_key="sk-fail-key")

    # 断言1：函数返回False（保存失败）
    assert result is False
    # 断言2：内存中的config已被修改
    assert test_config["openai"]["api_key"] == "sk-fail-key"
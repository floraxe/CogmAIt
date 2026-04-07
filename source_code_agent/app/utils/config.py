"""
系统配置工具
"""
import os
import json
import logging
from typing import Dict, Any, Optional, Mapping, Callable

# 配置日志
logger = logging.getLogger(__name__)

# 配置文件路径
CONFIG_PATH = os.path.join(os.getcwd(), "config.json")

# 默认配置
DEFAULT_CONFIG = {
    "neo4j": {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password",
        "database": "neo4j"
    },
    "openai": {
        "api_key": "",
        "model": "gpt-4"
    }
}

# 全局配置对象
_config = None


class _ConfigLoader:
    """配置加载器：提取可测试的文件读取/默认值补全/环境覆盖逻辑。"""

    def __init__(
        self,
        config_path: str,
        default_config: Dict[str, Any],
        environ: Mapping[str, str],
        path_exists: Optional[Callable[[str], bool]] = None,
        open_fn: Optional[Callable] = None,
    ) -> None:
        self.config_path = config_path
        self.default_config = default_config
        self.environ = environ
        self.path_exists = path_exists or os.path.exists
        self.open_fn = open_fn or open

    def load(self) -> Dict[str, Any]:
        if self.path_exists(self.config_path):
            try:
                with self.open_fn(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self._merge_defaults(config)
                return config
            except Exception as e:
                logger.error(f"加载配置文件失败: {str(e)}")

        config = self.default_config.copy()
        self._apply_env_overrides(config)
        return config

    def _merge_defaults(self, config: Dict[str, Any]) -> None:
        for key, value in self.default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in config[key]:
                        config[key][sub_key] = sub_value

    def _apply_env_overrides(self, config: Dict[str, Any]) -> None:
        if "NEO4J_URI" in self.environ:
            config["neo4j"]["uri"] = self.environ["NEO4J_URI"]
        if "NEO4J_USERNAME" in self.environ:
            config["neo4j"]["username"] = self.environ["NEO4J_USERNAME"]
        if "NEO4J_PASSWORD" in self.environ:
            config["neo4j"]["password"] = self.environ["NEO4J_PASSWORD"]
        if "NEO4J_DATABASE" in self.environ:
            config["neo4j"]["database"] = self.environ["NEO4J_DATABASE"]
        if "OPENAI_API_KEY" in self.environ:
            config["openai"]["api_key"] = self.environ["OPENAI_API_KEY"]

def load_config() -> Dict[str, Any]:
    """
    加载配置文件
    
    Returns:
        Dict: 配置信息
    """
    global _config
    
    if _config is not None:
        return _config
    
    loader = _ConfigLoader(
        config_path=CONFIG_PATH,
        default_config=DEFAULT_CONFIG,
        environ=os.environ,
    )
    _config = loader.load()
    return _config

def save_config(config: Dict[str, Any]) -> bool:
    """
    保存配置文件
    
    Args:
        config: 配置信息
        
    Returns:
        bool: 是否保存成功
    """
    global _config
    
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        _config = config
        return True
    except Exception as e:
        logger.error(f"保存配置文件失败: {str(e)}")
        return False

def get_neo4j_config() -> Dict[str, Any]:
    """
    获取Neo4j配置
    
    Returns:
        Dict: Neo4j配置信息
    """
    config = load_config()
    return config.get("neo4j", DEFAULT_CONFIG["neo4j"])

def get_openai_config() -> Dict[str, Any]:
    """
    获取OpenAI配置
    
    Returns:
        Dict: OpenAI配置信息
    """
    config = load_config()
    return config.get("openai", DEFAULT_CONFIG["openai"])

def update_neo4j_config(
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None
) -> bool:
    """
    更新Neo4j配置
    
    Args:
        uri: Neo4j URI
        username: 用户名
        password: 密码
        database: 数据库名
        
    Returns:
        bool: 是否更新成功
    """
    config = load_config()
    
    if uri is not None:
        config["neo4j"]["uri"] = uri
    if username is not None:
        config["neo4j"]["username"] = username
    if password is not None:
        config["neo4j"]["password"] = password
    if database is not None:
        config["neo4j"]["database"] = database
    
    return save_config(config)

def update_openai_config(
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> bool:
    """
    更新OpenAI配置
    
    Args:
        api_key: API密钥
        model: 模型名称
        
    Returns:
        bool: 是否更新成功
    """
    config = load_config()
    
    if api_key is not None:
        config["openai"]["api_key"] = api_key
    if model is not None:
        config["openai"]["model"] = model
    
    return save_config(config) 
"""
系统配置工具
"""
import os
import json
import logging
from typing import Dict, Any, Optional

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

def load_config() -> Dict[str, Any]:
    """
    加载配置文件
    
    Returns:
        Dict: 配置信息
    """
    global _config
    
    if _config is not None:
        return _config
    
    # 检查配置文件是否存在
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                
            # 确保配置包含所有必要的键
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key not in config[key]:
                            config[key][sub_key] = sub_value
            
            _config = config
            return config
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
    
    # 如果配置文件不存在或者读取失败，使用默认配置
    _config = DEFAULT_CONFIG.copy()
    
    # 尝试从环境变量读取配置
    if "NEO4J_URI" in os.environ:
        _config["neo4j"]["uri"] = os.environ["NEO4J_URI"]
    if "NEO4J_USERNAME" in os.environ:
        _config["neo4j"]["username"] = os.environ["NEO4J_USERNAME"]
    if "NEO4J_PASSWORD" in os.environ:
        _config["neo4j"]["password"] = os.environ["NEO4J_PASSWORD"]
    if "NEO4J_DATABASE" in os.environ:
        _config["neo4j"]["database"] = os.environ["NEO4J_DATABASE"]
    
    if "OPENAI_API_KEY" in os.environ:
        _config["openai"]["api_key"] = os.environ["OPENAI_API_KEY"]
    
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
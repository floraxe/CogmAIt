#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()  # 自动加载项目根目录的 .env 文件
import os
import sys
import json
import uvicorn
import asyncio
import logging
import importlib.util
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resolve_config_path(config_file_env, base_dir):
    """解析配置文件路径：优先环境变量，不存在则回落到项目默认路径。"""
    if config_file_env and os.path.exists(config_file_env):
        return config_file_env
    return os.path.join(base_dir, "config.json")


def load_server_config(config_file):
    """读取配置文件，失败时返回空字典（保持原行为）。"""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"无法读取配置文件: {str(e)}")
        return {}


def get_uvicorn_options(config):
    """从配置中提取 uvicorn 运行参数（含默认值）。"""
    host = config.get("host", "0.0.0.0")
    port = config.get("port", 8000)
    reload = config.get("reload", True)
    return host, port, reload


_base_dir = os.path.dirname(os.path.abspath(__file__))
_config_file = resolve_config_path(os.environ.get("CONFIG_FILE"), _base_dir)
_config = load_server_config(_config_file)
host, port, reload = get_uvicorn_options(_config)

# 启动MCP服务
async def start_mcp_service():
    try:
        # 检查mcp_services模块是否存在
        spec = importlib.util.find_spec("mcp_services")
        if spec is None:
            logger.warning("未找到mcp_services模块，跳过MCP服务启动")
            return
        
        # 导入mcp_services模块
        mcp_services = importlib.import_module("mcp_services")
        logger.info("正在启动MCP服务...")
        
        # 在单独的线程中启动MCP服务
        with ThreadPoolExecutor() as executor:
            await asyncio.get_event_loop().run_in_executor(
                executor, 
                lambda: asyncio.run(mcp_services.start())
            )
    except Exception as e:
        logger.error(f"启动MCP服务失败: {str(e)}")

if __name__ == "__main__":
    # 启动MCP服务
    try:
        # 创建子进程启动MCP服务
        import subprocess
        mcp_process = subprocess.Popen(
            [sys.executable, "mcp_services.py"],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        logger.info(f"MCP服务启动中 (PID: {mcp_process.pid})")
    except Exception as e:
        logger.error(f"启动MCP服务失败: {str(e)}")
    
    # 启动FastAPI应用
    logger.info(f"启动API服务 - 监听 {host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=reload) 
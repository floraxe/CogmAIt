#!/usr/bin/env python3
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

# 读取配置文件
config_file = os.environ.get("CONFIG_FILE", "config.json")
if not os.path.exists(config_file):
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

try:
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
except Exception as e:
    logger.error(f"无法读取配置文件: {str(e)}")
    config = {}

# 获取配置
host = config.get("host", "0.0.0.0")
port = config.get("port", 8000)
reload = config.get("reload", True)

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
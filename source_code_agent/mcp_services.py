#!/usr/bin/env python3
"""
MCP 服务实现 - 提供标准工具服务

此模块实现了基于模型上下文协议（MCP）的工具服务，支持被大模型调用来完成特定任务。
启动时会自动检查数据库中是否存在这些服务，如不存在则自动添加。
"""
import os
import json
import uuid
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
from mysql.connector import Error

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 提取为纯函数，便于测试；默认行为与原实现一致
def build_mcp_db_config(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    source = env if env is not None else os.environ
    return {
        "host": source.get("DB_HOST", "localhost"),
        "user": source.get("DB_USER", "root"),
        "password": source.get("DB_PASSWORD", "xkkxkkxkk"),
        "database": source.get("DB_DATABASE", "cogmait"),
    }


def resolve_mcp_server_bind(env: Optional[Dict[str, str]] = None) -> tuple[str, int]:
    source = env if env is not None else os.environ
    host = source.get("MCP_HOST", "0.0.0.0")
    port = int(source.get("MCP_PORT", 8001))
    return host, port


# 数据库连接配置
DB_CONFIG = build_mcp_db_config()

# 创建FastAPI应用
app = FastAPI(title="CogmaitMCP", description="Cogmait模型上下文协议服务")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 工具函数注册表
tools = {}

# MCP服务配置
MCP_SERVICES = [
    {
        "name": "WebSearch",
        "description": "网络搜索工具，允许模型获取实时互联网信息。\n\n参数:\n  query: 搜索查询字符串\n  num_results: 返回的结果数量，默认为5\n\n返回:\n  包含搜索结果的字典，包括状态、查询和结果列表",
        "type": "search",
        "provider": "CogmaitMCP",
        "icon": "",
        "deployment_type": "local",
        "is_official": True,
        "tags": ["search", "web", "information", "internet"],
        "config_template": {
            "search_engine": "google",
            "max_results": 5,
            "function": "web_search"
        },
        "pricing": {
            "free": True
        },
        "statistics": {
            "users": 128,
            "calls": 1520,
            "avgResponseTime": "320ms",
            "availability": "99.8%"
        },
        "api_endpoints": [
            {
                "name": "执行网络搜索",
                "method": "POST",
                "path": "/tools/web_search",
                "description": "执行网络搜索并返回结果",
                "parameters": [
                    {
                        "name": "query",
                        "type": "string",
                        "required": True,
                        "description": "搜索查询字符串"
                    },
                    {
                        "name": "num_results",
                        "type": "integer",
                        "required": False,
                        "description": "返回的结果数量，默认为5"
                    }
                ]
            }
        ],
        "examples": [
            {
                "title": "基本搜索示例",
                "description": "执行简单的网络搜索",
                "code": "import requests\n\nresponse = requests.post(\n    \"http://localhost:8001/tools/web_search\",\n    json={\n        \"query\": \"人工智能最新进展\",\n        \"num_results\": 3\n    }\n)\n\nresults = response.json()\nprint(results)"
            },
            {
                "title": "在对话中使用",
                "description": "在智能体对话中使用网络搜索",
                "code": "// 在智能体系统提示中添加\n\"当用户询问最新信息时，你可以使用WebSearch工具获取实时互联网数据。\"\n\n// 用户提问: \"最近有哪些AI领域的重大突破？\"\n// 系统将自动调用WebSearch工具搜索相关信息"
            }
        ],
        "usage_docs": "# WebSearch 使用指南\n\n## 简介\nWebSearch工具允许模型获取实时互联网信息，帮助回答用户关于最新事件、数据或信息的问题。\n\n## 使用场景\n- 回答关于最新新闻的问题\n- 查找最新的产品信息\n- 获取实时数据和统计信息\n\n## 参数说明\n- `query`: 搜索查询字符串，应该明确具体\n- `num_results`: 返回的结果数量，默认为5，最大为10\n\n## 注意事项\n- 搜索结果可能不总是完全准确\n- 建议在关键信息上交叉验证多个来源\n- 当前版本不支持图片搜索"
    },
    {
        "name": "Calculator",
        "description": "计算工具，执行数学计算并返回结果。\n\n参数:\n  expression: 数学表达式字符串，如 \"1 + 2 * 3\"\n\n返回:\n  包含计算结果的字典，包括状态、结果和原始表达式",
        "type": "tool",
        "provider": "CogmaitMCP",
        "icon": "",
        "deployment_type": "local",
        "is_official": True,
        "tags": ["calculator", "math", "computation", "arithmetic"],
        "config_template": {
            "precision": 10,
            "function": "calculator"
        },
        "pricing": {
            "free": True
        },
        "statistics": {
            "users": 256,
            "calls": 3840,
            "avgResponseTime": "50ms",
            "availability": "99.9%"
        },
        "api_endpoints": [
            {
                "name": "执行计算",
                "method": "POST",
                "path": "/tools/calculator",
                "description": "执行数学表达式计算并返回结果",
                "parameters": [
                    {
                        "name": "expression",
                        "type": "string",
                        "required": True,
                        "description": "数学表达式字符串，如 \"1 + 2 * 3\""
                    }
                ]
            }
        ],
        "examples": [
            {
                "title": "基本计算示例",
                "description": "执行简单的数学计算",
                "code": "import requests\n\nresponse = requests.post(\n    \"http://localhost:8001/tools/calculator\",\n    json={\n        \"expression\": \"(15 + 3) * 2 / 6\"\n    }\n)\n\nresult = response.json()\nprint(result)"
            },
            {
                "title": "在对话中使用",
                "description": "在智能体对话中使用计算器",
                "code": "// 在智能体系统提示中添加\n\"当用户需要进行数学计算时，你可以使用Calculator工具执行准确的计算。\"\n\n// 用户提问: \"计算(15 + 3) * 2 / 6的结果是多少？\"\n// 系统将自动调用Calculator工具执行计算"
            }
        ],
        "usage_docs": "# Calculator 使用指南\n\n## 简介\nCalculator工具提供准确的数学计算功能，支持基本的算术运算和一些常用数学函数。\n\n## 使用场景\n- 执行复杂的数学计算\n- 验证计算结果\n- 在对话中需要精确计算时使用\n\n## 支持的运算\n- 基本运算: 加(+)、减(-)、乘(*)、除(/)\n- 幂运算: **\n- 内置函数: abs(), round(), min(), max()\n\n## 注意事项\n- 表达式必须是有效的Python数学表达式\n- 出于安全考虑，不支持导入模块或执行其他Python代码\n- 计算精度默认为10位小数"
    },
    {
        "name": "WeatherInfo",
        "description": "天气信息查询工具，获取指定城市的天气数据。\n\n参数:\n  city: 城市名称\n  country_code: 国家代码，默认为CN(中国)\n\n返回:\n  包含天气信息的字典，包括位置、天气状况、温度、湿度和风速",
        "type": "api",
        "provider": "CogmaitMCP",
        "icon": "",
        "deployment_type": "local",
        "is_official": True,
        "tags": ["weather", "api", "geo", "forecast"],
        "config_template": {
            "api_key": "",
            "units": "metric",
            "function": "weather_info"
        },
        "pricing": {
            "free": True
        },
        "statistics": {
            "users": 187,
            "calls": 2340,
            "avgResponseTime": "280ms",
            "availability": "99.5%"
        },
        "api_endpoints": [
            {
                "name": "获取天气信息",
                "method": "POST",
                "path": "/tools/weather_info",
                "description": "获取指定城市的天气信息",
                "parameters": [
                    {
                        "name": "city",
                        "type": "string",
                        "required": True,
                        "description": "城市名称"
                    },
                    {
                        "name": "country_code",
                        "type": "string",
                        "required": False,
                        "description": "国家代码，默认为CN(中国)"
                    }
                ]
            }
        ],
        "examples": [
            {
                "title": "查询城市天气",
                "description": "查询特定城市的天气信息",
                "code": "import requests\n\nresponse = requests.post(\n    \"http://localhost:8001/tools/weather_info\",\n    json={\n        \"city\": \"北京\",\n        \"country_code\": \"CN\"\n    }\n)\n\nweather_data = response.json()\nprint(weather_data)"
            },
            {
                "title": "在对话中使用",
                "description": "在智能体对话中使用天气查询",
                "code": "// 在智能体系统提示中添加\n\"当用户询问天气情况时，你可以使用WeatherInfo工具获取实时天气数据。\"\n\n// 用户提问: \"北京今天天气怎么样？\"\n// 系统将自动调用WeatherInfo工具获取北京的天气信息"
            }
        ],
        "usage_docs": "# WeatherInfo 使用指南\n\n## 简介\nWeatherInfo工具提供全球城市的实时天气信息，包括温度、湿度、风速和天气状况。\n\n## 使用场景\n- 回答用户关于天气的问题\n- 提供旅行建议\n- 在对话中需要天气信息时使用\n\n## 参数说明\n- `city`: 城市名称，支持中文和英文\n- `country_code`: 国家代码，使用ISO 3166-1 alpha-2标准，默认为CN(中国)\n\n## 返回数据\n- 位置信息: 城市和国家\n- 天气状况: 晴朗、多云、小雨等\n- 温度: 摄氏度\n- 湿度: 百分比\n- 风速: 米/秒\n\n## 注意事项\n- 当前版本使用模拟数据，实际部署时将连接真实天气API\n- 某些小城市可能无法获取准确数据"
    },
    {
        "name": "DatabaseQuery",
        "description": "数据库查询工具，允许模型执行SQL查询获取数据。\n\n参数:\n  query: SQL查询语句\n  db_name: 数据库名称，默认为\"default\"\n\n返回:\n  包含查询结果的字典，包括状态、结果集和查询统计信息",
        "type": "database",
        "provider": "CogmaitMCP",
        "icon": "",
        "deployment_type": "local",
        "is_official": True,
        "tags": ["database", "sql", "query", "data"],
        "config_template": {
            "db_type": "mysql",
            "max_rows": 100,
            "function": "database_query"
        },
        "pricing": {
            "free": True
        },
        "statistics": {
            "users": 92,
            "calls": 1240,
            "avgResponseTime": "450ms",
            "availability": "99.2%"
        },
        "api_endpoints": [
            {
                "name": "执行数据库查询",
                "method": "POST",
                "path": "/tools/database_query",
                "description": "执行SQL查询并返回结果",
                "parameters": [
                    {
                        "name": "query",
                        "type": "string",
                        "required": True,
                        "description": "SQL查询语句"
                    },
                    {
                        "name": "db_name",
                        "type": "string",
                        "required": False,
                        "description": "数据库名称，默认为\"default\""
                    }
                ]
            }
        ],
        "examples": [
            {
                "title": "基本SQL查询",
                "description": "执行简单的SQL查询",
                "code": "import requests\n\nresponse = requests.post(\n    \"http://localhost:8001/tools/database_query\",\n    json={\n        \"query\": \"SELECT * FROM users LIMIT 5\",\n        \"db_name\": \"default\"\n    }\n)\n\nresults = response.json()\nprint(results)"
            },
            {
                "title": "在对话中使用",
                "description": "在智能体对话中使用数据库查询",
                "code": "// 在智能体系统提示中添加\n\"当用户需要查询数据库信息时，你可以使用DatabaseQuery工具执行SQL查询。\"\n\n// 用户提问: \"查询最近注册的5个用户\"\n// 系统将自动调用DatabaseQuery工具执行相应的SQL查询"
            }
        ],
        "usage_docs": "# DatabaseQuery 使用指南\n\n## 简介\nDatabaseQuery工具允许模型执行SQL查询获取数据库信息，支持多种SQL操作和数据库类型。\n\n## 使用场景\n- 查询数据库中的记录\n- 生成数据报表\n- 验证数据存在性\n\n## 参数说明\n- `query`: SQL查询语句，必须是有效的SQL\n- `db_name`: 数据库名称，默认为\"default\"\n\n## 支持的SQL操作\n- SELECT: 查询数据\n- INSERT: 插入数据 (需要权限)\n- UPDATE: 更新数据 (需要权限)\n- DELETE: 删除数据 (需要权限)\n\n## 注意事项\n- 查询结果默认限制为100行\n- 出于安全考虑，某些操作可能被限制\n- 当前版本使用模拟数据，实际部署时将连接真实数据库"
    },
    {
        "name": "ImageAnalysis",
        "description": "图像分析工具，识别和描述图像内容。\n\n参数:\n  image_url: 图像URL地址\n  analysis_type: 分析类型，默认为\"general\"，可选值包括\"general\"、\"ocr\"、\"face\"、\"object\"\n\n返回:\n  包含图像分析结果的字典，包括状态、分析结果和标签",
        "type": "vision",
        "provider": "CogmaitMCP",
        "icon": "",
        "deployment_type": "local",
        "is_official": True,
        "tags": ["vision", "image", "analysis", "ai"],
        "config_template": {
            "model": "default",
            "detail_level": "high",
            "function": "image_analysis"
        },
        "pricing": {
            "free": True
        },
        "statistics": {
            "users": 142,
            "calls": 1860,
            "avgResponseTime": "850ms",
            "availability": "98.7%"
        },
        "api_endpoints": [
            {
                "name": "分析图像",
                "method": "POST",
                "path": "/tools/image_analysis",
                "description": "分析图像内容并返回描述",
                "parameters": [
                    {
                        "name": "image_url",
                        "type": "string",
                        "required": True,
                        "description": "图像URL地址"
                    },
                    {
                        "name": "analysis_type",
                        "type": "string",
                        "required": False,
                        "description": "分析类型，默认为\"general\"，可选值包括\"general\"、\"ocr\"、\"face\"、\"object\""
                    }
                ]
            }
        ],
        "examples": [
            {
                "title": "基本图像分析",
                "description": "分析图像内容",
                "code": "import requests\n\nresponse = requests.post(\n    \"http://localhost:8001/tools/image_analysis\",\n    json={\n        \"image_url\": \"https://example.com/image.jpg\",\n        \"analysis_type\": \"general\"\n    }\n)\n\nanalysis_result = response.json()\nprint(analysis_result)"
            },
            {
                "title": "在对话中使用",
                "description": "在智能体对话中使用图像分析",
                "code": "// 在智能体系统提示中添加\n\"当用户上传图片或提供图片URL时，你可以使用ImageAnalysis工具分析图像内容。\"\n\n// 用户提问: \"这张图片里有什么？[图片URL]\"\n// 系统将自动调用ImageAnalysis工具分析图像内容"
            }
        ],
        "usage_docs": "# ImageAnalysis 使用指南\n\n## 简介\nImageAnalysis工具提供强大的图像分析功能，可以识别和描述图像内容，包括物体、场景、文字和人脸等。\n\n## 使用场景\n- 识别图像中的物体和场景\n- 提取图像中的文字(OCR)\n- 分析人脸表情和特征\n- 检测特定物体\n\n## 分析类型\n- `general`: 通用图像分析，识别主要内容和场景\n- `ocr`: 光学字符识别，提取图像中的文字\n- `face`: 人脸分析，检测人脸并分析表情和特征\n- `object`: 物体检测，识别图像中的特定物体\n\n## 参数说明\n- `image_url`: 图像URL地址，必须是可公开访问的URL\n- `analysis_type`: 分析类型，默认为\"general\"\n\n## 注意事项\n- 图像URL必须是可公开访问的\n- 图像大小建议不超过10MB\n- 当前版本使用模拟数据，实际部署时将连接真实视觉AI模型"
    }
]

# 工具装饰器
def tool(name=None, description=None):
    """工具函数装饰器"""
    def decorator(func):
        nonlocal name, description
        func_name = name or func.__name__
        func_desc = description or func.__doc__ or "无描述"
        
        # 注册工具
        tools[func_name] = {
            "name": func_name,
            "description": func_desc,
            "function": func
        }
        logger.info(f"注册工具: {func_name}")
        return func
    return decorator

# 工具函数定义
@tool(name="calculator", description="执行数学表达式计算并返回结果")
async def calculator(expression: str) -> Dict[str, Any]:
    """
    执行数学表达式计算并返回结果

    参数:
        expression: 数学表达式字符串，如 "1 + 2 * 3"

    返回:
        包含计算结果的字典
    """
    try:
        # 安全执行计算
        # 注意：在实际环境中应当使用更安全的计算方法，避免执行任意代码
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round, "min": min, "max": max})
        return {
            "status": "success",
            "result": result,
            "expression": expression
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "expression": expression
        }

@tool(name="web_search", description="执行网络搜索并返回结果")
async def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    执行网络搜索并返回结果

    参数:
        query: 搜索查询字符串
        num_results: 返回的结果数量，默认为5

    返回:
        包含搜索结果的字典
    """
    try:
        # 模拟搜索结果
        # 在实际实现中，这里应该调用真实的搜索API，如Google、Bing或Tavily等
        mock_results = [
            {
                "title": f"搜索结果 {i} - {query}",
                "url": f"https://example.com/result/{i}",
                "snippet": f"这是关于 '{query}' 的第 {i} 条搜索结果的摘要内容。",
                "source": "模拟搜索引擎"
            } for i in range(1, min(num_results + 1, 10))
        ]
        
        return {
            "status": "success",
            "query": query,
            "results": mock_results,
            "total_results": len(mock_results)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "query": query
        }

@tool(name="weather_info", description="获取指定城市的天气信息")
async def weather_info(city: str, country_code: str = "CN") -> Dict[str, Any]:
    """
    获取指定城市的天气信息

    参数:
        city: 城市名称
        country_code: 国家代码，默认为CN(中国)

    返回:
        包含天气信息的字典
    """
    try:
        # 模拟天气API响应
        # 实际实现中应使用真实的天气API
        import random
        weather_conditions = ["晴朗", "多云", "小雨", "大雨", "阴天", "雷阵雨", "雪"]
        temperature = round(random.uniform(10, 35), 1)
        humidity = random.randint(30, 95)
        wind_speed = round(random.uniform(0, 30), 1)
        
        return {
            "status": "success",
            "location": {
                "city": city,
                "country": country_code
            },
            "weather": {
                "condition": random.choice(weather_conditions),
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "units": "metric"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "city": city
        }

@tool(name="database_query", description="执行数据库查询并返回结果")
async def database_query(query: str, db_name: str = "default") -> Dict[str, Any]:
    """
    执行数据库查询并返回结果

    参数:
        query: SQL查询语句
        db_name: 数据库名称，默认为"default"

    返回:
        包含查询结果的字典
    """
    try:
        # 模拟数据库查询结果
        # 实际实现中应当执行真实的SQL查询，并注意安全性
        if "SELECT" not in query.upper():
            return {
                "status": "error",
                "error": "只支持SELECT查询",
                "query": query
            }
        
        # 模拟一些随机数据
        columns = ["id", "name", "value", "created_at"]
        rows = [
            {"id": i, "name": f"Item {i}", "value": i * 10, "created_at": datetime.now().isoformat()}
            for i in range(1, 6)
        ]
        
        return {
            "status": "success",
            "query": query,
            "db_name": db_name,
            "result": {
                "columns": columns,
                "rows": rows,
                "total_rows": len(rows)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "query": query
        }

@tool(name="image_analysis", description="分析图像内容并返回描述")
async def image_analysis(image_url: str, analysis_type: str = "general") -> Dict[str, Any]:
    """
    分析图像内容并返回描述

    参数:
        image_url: 图像URL
        analysis_type: 分析类型，可选值："general", "objects", "text", "faces"

    返回:
        包含图像分析结果的字典
    """
    try:
        # 模拟图像分析结果
        # 实际实现中应当调用计算机视觉API
        analysis_results = {
            "general": "这是一张自然风景图，展示了山脉、森林和湖泊的美丽景色。",
            "objects": ["树木", "山脉", "湖泊", "岩石", "天空"],
            "text": "图像中未检测到文本内容",
            "faces": "图像中未检测到人脸"
        }
        
        return {
            "status": "success",
            "image_url": image_url,
            "analysis_type": analysis_type,
            "result": analysis_results.get(analysis_type, "未支持的分析类型"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "image_url": image_url
        }

# API路由
@app.get("/")
async def root():
    """根路由，返回API信息"""
    return {
        "name": "CogmaitMCP",
        "description": "Cogmait模型上下文协议服务",
        "version": "1.0.0",
        "tools_count": len(tools),
        "available_tools": list(tools.keys())
    }

@app.get("/tools")
async def list_tools():
    """列出所有可用工具"""
    tool_list = []
    for name, tool_info in tools.items():
        tool_list.append({
            "name": name,
            "description": tool_info["description"]
        })
    
    return {
        "tools": tool_list,
        "count": len(tool_list)
    }

@app.post("/tools/{tool_name}")
async def call_tool(tool_name: str, params: Dict[str, Any] = Body(...)):
    """调用特定工具"""
    if tool_name not in tools:
        raise HTTPException(status_code=404, detail=f"找不到工具: {tool_name}")
    
    try:
        # 获取工具函数
        tool_func = tools[tool_name]["function"]
        
        # 调用工具函数
        if asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(**params)
        else:
            result = tool_func(**params)
        
        return result
    except Exception as e:
        logger.error(f"调用工具 {tool_name} 失败: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "tool": tool_name,
                "params": params
            }
        )

async def sync_mcp_services_to_db():
    """将MCP服务同步到数据库"""
    try:
        # 连接到MySQL数据库
        connection = mysql.connector.connect(**DB_CONFIG)
        
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            
            for service_data in MCP_SERVICES:
                # 检查服务是否已存在
                cursor.execute(
                    "SELECT id FROM mcp_services WHERE name = %s AND provider = %s",
                    (service_data["name"], service_data["provider"])
                )
                existing = cursor.fetchone()
                
                service_id = None
                if existing:
                    logger.info(f"MCP服务已存在: {service_data['name']}, 更新服务信息")
                    service_id = existing["id"]
                else:
                    # 生成唯一ID
                    service_id = str(uuid.uuid4())
                    logger.info(f"添加新MCP服务: {service_data['name']}, ID: {service_id}")
                
                now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                
                # 确保JSON字段是字符串格式
                tags_json = json.dumps(service_data.get("tags")) if service_data.get("tags") else None
                config_template_json = json.dumps(service_data.get("config_template")) if service_data.get("config_template") else None
                pricing_json = json.dumps(service_data.get("pricing")) if service_data.get("pricing") else None
                statistics_json = json.dumps(service_data.get("statistics")) if service_data.get("statistics") else None
                api_endpoints_json = json.dumps(service_data.get("api_endpoints")) if service_data.get("api_endpoints") else None
                examples_json = json.dumps(service_data.get("examples")) if service_data.get("examples") else None
                
                if existing:
                    # 更新现有服务
                    update_query = """
                    UPDATE mcp_services SET 
                        name = %s, 
                        description = %s, 
                        type = %s, 
                        provider = %s, 
                        icon = %s, 
                        tags = %s, 
                        deployment_type = %s, 
                        updated_at = %s, 
                        is_official = %s, 
                        config_template = %s,
                        pricing = %s,
                        statistics = %s,
                        api_endpoints = %s,
                        examples = %s,
                        usage_docs = %s,
                        owner_id = %s,
                        github_url = %s
                    WHERE id = %s
                    """
                    
                    data = (
                        service_data["name"],
                        service_data["description"],
                        service_data["type"],
                        service_data["provider"],
                        service_data.get("icon", ""),
                        tags_json,
                        service_data.get("deployment_type", "local"),
                        now,
                        service_data.get("is_official", False),
                        config_template_json,
                        pricing_json,
                        statistics_json,
                        api_endpoints_json,
                        examples_json,
                        service_data.get("usage_docs", ""),
                        service_data.get("owner_id", "admin"),  # 默认为admin
                        service_data.get("github_url", ""),
                        service_id
                    )
                    
                    cursor.execute(update_query, data)
                else:
                    # 插入新服务
                    insert_query = """
                    INSERT INTO mcp_services (
                        id, name, description, type, provider, icon, 
                        tags, deployment_type, created_at, updated_at, 
                        is_official, config_template, pricing, statistics,
                        api_endpoints, examples, usage_docs, owner_id, github_url
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    data = (
                        service_id,
                        service_data["name"],
                        service_data["description"],
                        service_data["type"],
                        service_data["provider"],
                        service_data.get("icon", ""),
                        tags_json,
                        service_data.get("deployment_type", "local"),
                        now,
                        now,
                        service_data.get("is_official", False),
                        config_template_json,
                        pricing_json,
                        statistics_json,
                        api_endpoints_json,
                        examples_json,
                        service_data.get("usage_docs", ""),
                        service_data.get("owner_id", "admin"),  # 默认为admin
                        service_data.get("github_url", "")
                    )
                    
                    cursor.execute(insert_query, data)
            
            # 提交事务
            connection.commit()
            logger.info(f"MCP服务同步完成")
            
            cursor.close()
            connection.close()
    
    except Error as e:
        logger.error(f"数据库操作失败: {str(e)}")
    except Exception as e:
        logger.error(f"同步MCP服务失败: {str(e)}")
        logger.exception(e)

def register_tools_for_ollama():
    """为Ollama注册工具定义，以便Ollama模型可以调用MCP工具"""
    # 获取工具定义
    tool_schemas = []
    
    for name, tool_info in tools.items():
        # 从函数获取参数信息
        import inspect
        func = tool_info["function"]
        sig = inspect.signature(func)
        
        # 构建参数模式
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name != "self":  # 跳过self参数
                param_type = "string"  # 默认类型
                param_desc = ""
                
                # 尝试从类型注解获取类型
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                
                # 参数说明
                param_desc = f"{param_name}参数"
                
                properties[param_name] = {
                    "type": param_type,
                    "description": param_desc
                }
                
                # 如果没有默认值，则为必需参数
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
        
        # 构建完整的工具模式
        tool_schema = {
            "name": name,
            "description": tool_info["description"],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
        
        tool_schemas.append(tool_schema)
    
    return tool_schemas

async def start():
    """启动MCP服务器并同步服务到数据库"""
    # 同步服务到数据库
    await sync_mcp_services_to_db()
    
    # 注册Ollama工具
    tool_schemas = register_tools_for_ollama()
    logger.info(f"为Ollama注册了 {len(tool_schemas)} 个工具")
    
    # 启动FastAPI服务器
    import uvicorn
    host, port = resolve_mcp_server_bind()
    logger.info(f"启动MCP服务器 - 监听 {host}:{port}")
    
    # 使用uvicorn启动服务
    config = uvicorn.Config(app=app, host=host, port=port)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(start()) 
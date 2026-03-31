from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
import httpx
import json
import logging
import traceback
import time
import importlib

from app.models.mcp import MCPService

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MCP服务的默认URL
DEFAULT_MCP_SERVICE_URL = "http://localhost:8001"

def get_mcp_service(db: Session, service_id: str) -> Optional[MCPService]:
    """
    通过ID获取MCP服务
    
    参数:
        db (Session): 数据库会话
        service_id (str): MCP服务ID
        
    返回:
        Optional[MCPService]: MCP服务对象或None
    """
    return db.query(MCPService).filter(MCPService.id == service_id).first()


def get_mcp_services(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    name: Optional[str] = None,
    provider: Optional[str] = None,
    type: Optional[str] = None,
    status: Optional[str] = None
) -> List[MCPService]:
    """
    获取MCP服务列表，支持过滤
    
    参数:
        db (Session): 数据库会话
        skip (int): 跳过的记录数
        limit (int): 限制返回的记录数
        name (Optional[str]): 按名称过滤
        provider (Optional[str]): 按提供商过滤
        type (Optional[str]): 按类型过滤
        status (Optional[str]): 按状态过滤
        
    返回:
        List[MCPService]: MCP服务列表
    """
    query = db.query(MCPService)
    
    # 应用过滤条件
    if name:
        query = query.filter(MCPService.name.ilike(f"%{name}%"))
    if provider:
        query = query.filter(MCPService.provider == provider)
    if type:
        query = query.filter(MCPService.type == type)
    if status:
        query = query.filter(MCPService.status == status)
    
    # 应用分页
    return query.offset(skip).limit(limit).all()


def count_mcp_services(
    db: Session,
    name: Optional[str] = None,
    provider: Optional[str] = None,
    type: Optional[str] = None,
    status: Optional[str] = None
) -> int:
    """
    计算符合条件的MCP服务总数
    
    参数:
        db (Session): 数据库会话
        name (Optional[str]): 按名称过滤
        provider (Optional[str]): 按提供商过滤
        type (Optional[str]): 按类型过滤
        status (Optional[str]): 按状态过滤
        
    返回:
        int: 符合条件的MCP服务总数
    """
    query = db.query(MCPService)
    
    # 应用过滤条件
    if name:
        query = query.filter(MCPService.name.ilike(f"%{name}%"))
    if provider:
        query = query.filter(MCPService.provider == provider)
    if type:
        query = query.filter(MCPService.type == type)
    if status:
        query = query.filter(MCPService.status == status)
    
    return query.count()


async def call_mcp_service(
    db: Session, 
    service_id: str, 
    function_name: str, 
    params: Dict[str, Any],
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    调用MCP服务的指定功能
    
    参数:
        db: 数据库会话
        service_id: 服务ID
        function_name: 功能函数名称
        params: 调用参数
        user_id: 用户ID，可选
    
    返回:
        Dict[str, Any]: 调用结果
    """
    logger.info(f"开始调用MCP服务: service_id={service_id}, function_name={function_name}")
    logger.info(f"调用参数: {json.dumps(params, ensure_ascii=False)}")
    
    try:
        # 获取服务信息
        service = get_mcp_service(db, service_id)
        if not service:
            error_msg = f"找不到MCP服务: {service_id}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "service_id": service_id,
                "function_name": function_name,
                "params": params
            }
        
        # 构建API请求URL
        url = f"{DEFAULT_MCP_SERVICE_URL}/tools/{function_name}"
        logger.info(f"调用MCP服务URL: {url}")
        
        # 发送HTTP请求
        start_time = time.time()
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=params, timeout=30.0)
                response_time = time.time() - start_time
                logger.info(f"MCP服务响应时间: {response_time:.2f}秒")
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"MCP服务调用成功: {json.dumps(result, ensure_ascii=False)}")
                    
                    # 添加调用信息到结果中
                    result["mcp_call_info"] = {
                        "service_id": service_id,
                        "service_name": service.name,
                        "function_name": function_name,
                        "params": params,
                        "response_time_ms": int(response_time * 1000)
                    }
                    
                    # 更新服务统计信息
                    if user_id:
                        # 这里可以添加代码更新用户的服务使用统计
                        pass
                    
                    return result
                else:
                    error_msg = f"MCP服务返回错误: HTTP {response.status_code}, {response.text}"
                    logger.error(error_msg)
                    return {
                        "status": "error",
                        "error": error_msg,
                        "http_status": response.status_code,
                        "service_id": service_id,
                        "function_name": function_name,
                        "params": params
                    }
            except httpx.RequestError as e:
                error_msg = f"请求MCP服务失败: {str(e)}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "service_id": service_id,
                    "function_name": function_name,
                    "params": params
                }
    except Exception as e:
        error_msg = f"调用MCP服务时发生错误: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        
        # 尝试从mcp_services.py导入函数
        try:
            logger.info(f"尝试直接导入并调用函数: {function_name}")
            mcp_module = importlib.import_module("mcp_services")
            
            # 检查模块中是否有该函数
            if hasattr(mcp_module, "tools") and function_name in mcp_module.tools:
                logger.info(f"在mcp_services模块中找到函数: {function_name}")
                tool_info = mcp_module.tools[function_name]
                tool_func = tool_info["function"]
                
                # 调用函数
                logger.info(f"直接调用函数: {function_name}({json.dumps(params, ensure_ascii=False)})")
                start_time = time.time()
                result = await tool_func(**params)
                response_time = time.time() - start_time
                logger.info(f"函数调用成功，耗时: {response_time:.2f}秒")
                
                # 添加调用信息到结果中
                result["mcp_call_info"] = {
                    "service_id": service_id,
                    "service_name": service.name if service else "未知服务",
                    "function_name": function_name,
                    "params": params,
                    "response_time_ms": int(response_time * 1000),
                    "direct_call": True
                }
                
                return result
            else:
                logger.error(f"在mcp_services模块中找不到函数: {function_name}")
                return {
                    "status": "error",
                    "error": f"在mcp_services模块中找不到函数: {function_name}",
                    "service_id": service_id,
                    "function_name": function_name,
                    "params": params
                }
        except ImportError:
            logger.error("无法导入mcp_services模块")
            return {"error": "无法导入mcp_services模块"}
        except Exception as inner_e:
            logger.error(f"直接调用函数时发生错误: {str(inner_e)}")
            logger.exception(inner_e)
            return {
                "status": "error",
                "error": f"直接调用函数时发生错误: {str(inner_e)}",
                "service_id": service_id,
                "function_name": function_name,
                "params": params
            }

async def call_local_function(function_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    在无法连接到MCP服务时，尝试直接调用本地函数
    
    参数:
        function_name (str): 要调用的函数名称
        params (Dict[str, Any]): 调用参数
        
    返回:
        Dict[str, Any]: 函数调用结果
    """
    try:
        logger.info(f"尝试直接调用本地函数: {function_name}, 参数: {params}")
        
        # 添加项目根目录到Python路径
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # 尝试从mcp_services.py导入函数
        try:
            mcp_module = importlib.import_module("mcp_services")
            
            # 检查函数是否存在
            if hasattr(mcp_module, function_name):
                func = getattr(mcp_module, function_name)
                logger.info(f"找到本地MCP函数: {function_name}")
                
                # 判断函数是否是协程函数
                if asyncio.iscoroutinefunction(func):
                    logger.info(f"调用异步函数: {function_name}")
                    result = await func(**params)
                else:
                    logger.info(f"调用同步函数: {function_name}")
                    result = func(**params)
                
                logger.info(f"本地函数调用成功: {function_name}")
                return {"result": result}
            else:
                # 尝试从mcp.tool装饰器注册的工具中查找
                if hasattr(mcp_module, "mcp") and hasattr(mcp_module.mcp, "tools"):
                    tools = mcp_module.mcp.tools
                    for tool in tools:
                        if tool.name == function_name:
                            logger.info(f"找到MCP工具: {function_name}")
                            # 直接调用工具函数
                            result = await tool.func(**params)
                            logger.info(f"MCP工具调用成功: {function_name}")
                            return {"result": result}
                
                logger.error(f"本地函数 {function_name} 不存在")
                return {"error": f"本地函数 {function_name} 不存在"}
        except ImportError:
            logger.error("无法导入mcp_services模块")
            return {"error": "无法导入mcp_services模块"}
        except Exception as e:
            logger.error(f"调用本地函数时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"调用本地函数失败: {str(e)}"}
    except Exception as e:
        logger.error(f"尝试调用本地函数时发生未知错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"尝试调用本地函数失败: {str(e)}"} 

async def analyze_mcp_service_needs(db, model_id, user_message, available_services):
    """
    分析用户消息，判断是否需要调用MCP服务
    
    参数:
        db: 数据库会话
        model_id: 模型ID
        user_message: 用户消息
        available_services: 可用的MCP服务列表
    
    返回:
        Dict[str, Any]: 分析结果
    """
    from app.utils.model import execute_model_inference
    
    try:
        # 如果没有可用服务，直接返回不需要调用
        if not available_services or len(available_services) == 0:
            logger.warning("没有可用的MCP服务")
            return {"call_mcp": False, "reason": "没有可用的MCP服务"}
        
        # 获取模型
        from app.models.model import Model
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            logger.warning(f"找不到模型: {model_id}")
            return {"call_mcp": False, "reason": f"找不到模型: {model_id}"}
        
        # 判断模型是否支持MCP
        if not model.mcp_support:
            logger.warning(f"模型 {model.name} 不支持MCP")
            return {"call_mcp": False, "reason": f"模型 {model.name} 不支持MCP"}
        
        # 构建可用服务的描述
        available_services_desc = []
        for service in available_services:
            # 处理不同类型的服务对象
            if hasattr(service, 'to_dict'):
                # 如果是SQLAlchemy模型对象
                service_dict = service.to_dict()
            elif hasattr(service, 'name') and hasattr(service, 'description') and hasattr(service, 'type'):
                # 如果是具有必要属性的对象
                service_dict = {
                    "id": service.id,
                    "name": service.name,
                    "description": service.description,
                    "type": service.type
                }
            else:
                # 假设是字典
                service_dict = service
            
            # 添加服务描述
            service_desc = {
                "id": service_dict.get("id"),
                "name": service_dict.get("name"),
                "description": service_dict.get("description"),
                "type": service_dict.get("type"),
                "functions": get_service_functions(service_dict.get("type"))
            }
            available_services_desc.append(service_desc)
        logger.info(f"可用的MCP服务列表: {json.dumps(available_services_desc, ensure_ascii=False, indent=2)}")
        # 构建系统消息，让模型判断是否需要调用MCP服务
        mcp_detection_messages = [
            {"role": "system", "content": f"""你是一个AI助手，需要判断用户的问题是否需要调用外部工具或服务来回答。
以下是可用的MCP服务列表:
{json.dumps(available_services_desc, ensure_ascii=False, indent=2)}

请分析用户的问题，并返回JSON格式的结果，表明是否需要调用MCP服务，以及调用哪个服务和使用什么参数。
格式为：{{"call_mcp": true|false, "service_id": "服务ID", "function_name": "函数名称", "params": {{"参数名": "参数值"}}}}

只有明确需要功能时才返回true。
如果用户只是在问一般性问题或聊天，请返回false。

请确保返回的是有效的JSON格式，不要包含任何额外的文本说明。只返回JSON对象。"""},
            {"role": "user", "content": user_message}
        ]
        
        # 调用模型判断是否需要调用MCP服务
        logger.info(f"分析用户消息是否需要调用MCP服务: {user_message[:100]}...")
        model_detection_response = await execute_model_inference(
            db,
            model_id,
            {
                "messages": mcp_detection_messages,
                "stream": False,
                "temperature": 0.1  # 低温度以获得更确定的回答
            }
        )
        model_detection_response = json.loads(model_detection_response)
        logger.info(f"模型返回的完整响应: {type(model_detection_response)}，{json.dumps(model_detection_response, ensure_ascii=False)}")
        
        # 直接从模型响应中提取content内容
        detection_content = ""
        if isinstance(model_detection_response, dict) and "choices" in model_detection_response:
            choices = model_detection_response.get("choices", [])
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                detection_content = message.get("content", "")
                logger.info(f"模型返回的content内容: {detection_content}")
        
        # 尝试直接解析JSON内容
        try:
            # 直接尝试解析整个content内容
            detection_json = json.loads(detection_content)
            logger.info(f"MCP服务需求分析结果: {json.dumps(detection_json, ensure_ascii=False)}")
            return detection_json
        except json.JSONDecodeError as e:
            # 如果直接解析失败，尝试使用正则表达式提取JSON部分
            logger.warning(f"直接解析JSON失败: {e}，尝试使用正则表达式提取")
            import re
            json_match = re.search(r'\{.*\}', detection_content, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    logger.info(f"通过正则表达式提取的JSON字符串: {json_str}")
                    detection_json = json.loads(json_str)
                    logger.info(f"通过正则表达式提取的MCP服务需求分析结果: {json.dumps(detection_json, ensure_ascii=False)}")
                    return detection_json
                except json.JSONDecodeError as e2:
                    logger.warning(f"通过正则表达式提取后解析JSON仍然失败: {e2}")
                    logger.warning(f"问题JSON字符串: {json_match.group(0)}")
                    return {
                        "call_mcp": False,
                        "reason": f"无法解析JSON: {e2}"
                    }
            else:
                logger.warning(f"无法从模型响应中提取JSON: {detection_content}")
                return {
                    "call_mcp": False,
                    "reason": "无法从模型响应中提取JSON"
                }
    
    except Exception as e:
        logger.error(f"分析MCP服务需求时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"call_mcp": False, "reason": f"分析出错: {str(e)}"}

def get_service_functions(service_type):
    """
    根据服务类型返回可用的函数列表
    
    参数:
        service_type: 服务类型
        
    返回:
        List[Dict]: 函数列表
    """
    functions = {
        "tool": [{
            "name": "calculator",
            "description": "执行数学表达式计算并返回结果",
            "parameters": {
                "expression": "要计算的数学表达式，如 '1 + 2 * 3'"
            }
        }],
        "search": [{
            "name": "web_search",
            "description": "执行网络搜索并返回结果",
            "parameters": {
                "query": "搜索查询字符串",
                "num_results": "返回的结果数量，默认为5"
            }
        }],
        "api": [{
            "name": "weather_info",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "city": "城市名称",
                "country_code": "国家代码，默认为CN(中国)"
            }
        }],
        "database": [{
            "name": "database_query",
            "description": "执行数据库查询并返回结果",
            "parameters": {
                "query": "SQL查询语句",
                "db_name": "数据库名称，默认为'default'"
            }
        }],
        "vision": [{
            "name": "image_analysis",
            "description": "分析图像内容并返回描述",
            "parameters": {
                "image_url": "图像URL",
                "analysis_type": "分析类型，可选值：'general', 'objects', 'text', 'faces'"
            }
        }]
    }
    
    return functions.get(service_type, [])

def extract_function_params(message, function_name):
    """
    从用户消息中提取函数调用参数
    
    参数:
        message: 用户消息
        function_name: 函数名称
        
    返回:
        Dict[str, Any]: 提取的参数
    """
    # 根据函数名称提取不同的参数
    if function_name == "calculator":
        # 提取数学表达式
        import re
        # 查找可能是数学表达式的部分
        expressions = re.findall(r'(\d+\s*[\+\-\*\/]\s*\d+(?:[\+\-\*\/]\s*\d+)*)', message)
        if expressions:
            return {"expression": expressions[0].strip()}
        else:
            # 如果找不到明确的表达式，尝试提取数字
            numbers = re.findall(r'\d+', message)
            if len(numbers) >= 2:
                # 假设是加法操作
                return {"expression": f"{numbers[0]} + {numbers[1]}"}
            return {"expression": "1 + 1"}  # 默认值
    
    elif function_name == "web_search":
        # 提取搜索查询
        import re
        # 尝试找出引号内的内容作为查询
        quoted = re.findall(r'["\'](.*?)["\']', message)
        if quoted:
            return {"query": quoted[0].strip(), "num_results": 5}
        
        # 尝试找出"搜索"或"查询"后面的内容
        search_pattern = re.findall(r'(?:搜索|查询)\s+(.*?)(?:\s|$)', message)
        if search_pattern:
            return {"query": search_pattern[0].strip(), "num_results": 5}
        
        # 默认使用整个消息作为查询
        return {"query": message.strip(), "num_results": 5}
    
    elif function_name == "weather_info":
        # 提取城市名称
        import re
        # 尝试找出城市名称
        cities = re.findall(r'(?:天气).{0,5}(\w+市|\w+县|\w+区|\w+)', message)
        if cities:
            return {"city": cities[0].strip(), "country_code": "CN"}
        
        return {"city": "北京", "country_code": "CN"}  # 默认值
    
    elif function_name == "database_query":
        # 提取SQL查询
        import re
        # 尝试找出SQL语句
        sql = re.findall(r'SQL[:\s]*(.*)', message, re.IGNORECASE)
        if sql:
            return {"query": sql[0].strip(), "db_name": "default"}
        
        # 默认简单查询
        return {"query": "SELECT * FROM users LIMIT 5", "db_name": "default"}
    
    elif function_name == "image_analysis":
        # 提取图像URL和分析类型
        import re
        # 尝试找出URL
        urls = re.findall(r'https?://\S+', message)
        
        # 尝试确定分析类型
        analysis_type = "general"
        if "人脸" in message or "面部" in message:
            analysis_type = "faces"
        elif "文本" in message or "文字" in message:
            analysis_type = "text"
        elif "物体" in message or "对象" in message:
            analysis_type = "objects"
        
        if urls:
            return {"image_url": urls[0], "analysis_type": analysis_type}
        
        # 默认值
        return {"image_url": "https://example.com/image.jpg", "analysis_type": analysis_type}
    
    # 默认返回空参数
    return {} 
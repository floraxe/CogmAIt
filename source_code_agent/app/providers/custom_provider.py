import time
import json
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
import httpx

from app.providers.base import ModelProvider


class CustomProvider(ModelProvider):
    """
    自定义模型提供商，允许用户连接任何与OpenAI兼容的API
    """
    
    @property
    def provider_id(self) -> str:
        return "custom"
    
    @property
    def provider_name(self) -> str:
        return "自定义API"
    
    @property
    def description(self) -> str:
        return "连接任何兼容OpenAI的API服务，或自托管的模型服务"
    
    @property
    def icon(self) -> Optional[str]:
        return "https://ollama.com/public/ollama.png"
    
    @property
    def default_base_url(self) -> Optional[str]:
        return None
    
    @property
    def supported_model_types(self) -> List[str]:
        return ["chat", "completion", "embedding"]
    
    @property
    def features(self) -> List[str]:
        return ["自定义终端", "灵活配置", "本地部署", "流式输出"]
    
    async def test_connection(self, api_key: str, base_url: Optional[str] = None) -> Dict[str, Any]:
        """测试与自定义API的连接"""
        if not base_url:
            return {
                "status": "failed",
                "message": "自定义API必须提供基础URL"
            }
        
        try:
            start_time = time.time()
            
            # 尝试获取模型列表，简单测试连接
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # 尝试列出模型或发送简单请求
                response = await client.get(
                    f"{base_url}/models",
                    headers=headers
                )
                
                if response.status_code != 200:
                    return {
                        "status": "failed",
                        "message": f"API返回错误: {response.status_code} - {response.text}"
                    }
                
                response_time = round((time.time() - start_time) * 1000)
                
                return {
                    "status": "success",
                    "message": "连接成功",
                    "response": {
                        "model": "自定义API",
                        "responseTime": f"{response_time}ms"
                    }
                }
        except Exception as e:
            return {
                "status": "failed",
                "message": f"连接失败: {str(e)}"
            }
    
    async def chat_completion(
        self, 
        api_key: str, 
        messages: List[Dict[str, str]], 
        model: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """发送聊天完成请求到自定义API，支持流式输出"""
        if not base_url:
            return {
                "status": "error",
                "message": "自定义API必须提供基础URL"
            }
        
        # 构建请求数据
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        # 添加可选参数
        if max_tokens:
            data["max_tokens"] = max_tokens
            
        # 添加其他参数
        for key, value in kwargs.items():
            if value is not None:
                data[key] = value
        
        # 准备请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        
        # 处理流式输出
        if stream:
            async def stream_generator():
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        async with client.stream(
                            "POST",
                            f"{base_url}/chat/completions",
                            headers=headers,
                            json=data,
                            timeout=None
                        ) as response:
                            if response.status_code != 200:
                                yield {
                                    "status": "error",
                                    "message": f"API返回错误: {response.status_code}"
                                }
                                return
                                
                            # 处理SSE流
                            buffer = ""
                            async for chunk in response.aiter_text():
                                buffer += chunk
                                
                                # 处理缓冲区中的SSE事件
                                while "\n\n" in buffer:
                                    event, buffer = buffer.split("\n\n", 1)
                                    
                                    for line in event.split("\n"):
                                        if line.startswith("data: "):
                                            data_str = line[6:]
                                            
                                            # 跳过[DONE]消息
                                            if data_str.strip() == "[DONE]":
                                                continue
                                                
                                            try:
                                                # 解析JSON数据
                                                data_json = json.loads(data_str)
                                                
                                                # 添加响应时间
                                                current_time = time.time()
                                                response_time = round((current_time - start_time) * 1000)
                                                data_json["response_time_ms"] = response_time
                                                
                                                yield data_json
                                            except json.JSONDecodeError:
                                                yield {
                                                    "status": "error",
                                                    "message": f"无法解析JSON: {data_str}"
                                                }
                except Exception as e:
                    yield {
                        "status": "error",
                        "message": f"流式响应失败: {str(e)}"
                    }
            
            return stream_generator()
        
        # 处理非流式输出
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                response_time = round((time.time() - start_time) * 1000)
                
                if response.status_code != 200:
                    return {
                        "status": "error",
                        "message": f"API返回错误: {response.status_code} - {response.text}"
                    }
                
                result = response.json()
                result["response_time_ms"] = response_time
                return result
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"聊天完成请求失败: {str(e)}"
            }
    
    async def text_completion(
        self,
        api_key: str,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """发送文本完成请求到自定义API，支持流式输出"""
        if not base_url:
            return {
                "status": "error",
                "message": "自定义API必须提供基础URL"
            }
        
        # 构建请求数据
        data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": stream
        }
        
        # 添加可选参数
        if max_tokens:
            data["max_tokens"] = max_tokens
            
        # 添加其他参数
        for key, value in kwargs.items():
            if value is not None:
                data[key] = value
        
        # 准备请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        
        # 处理流式输出
        if stream:
            async def stream_generator():
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        async with client.stream(
                            "POST",
                            f"{base_url}/completions",
                            headers=headers,
                            json=data,
                            timeout=None
                        ) as response:
                            if response.status_code != 200:
                                yield {
                                    "status": "error",
                                    "message": f"API返回错误: {response.status_code}"
                                }
                                return
                                
                            # 处理SSE流
                            buffer = ""
                            async for chunk in response.aiter_text():
                                buffer += chunk
                                
                                # 处理缓冲区中的SSE事件
                                while "\n\n" in buffer:
                                    event, buffer = buffer.split("\n\n", 1)
                                    
                                    for line in event.split("\n"):
                                        if line.startswith("data: "):
                                            data_str = line[6:]
                                            
                                            # 跳过[DONE]消息
                                            if data_str.strip() == "[DONE]":
                                                continue
                                                
                                            try:
                                                # 解析JSON数据
                                                data_json = json.loads(data_str)
                                                
                                                # 添加响应时间
                                                current_time = time.time()
                                                response_time = round((current_time - start_time) * 1000)
                                                data_json["response_time_ms"] = response_time
                                                
                                                yield data_json
                                            except json.JSONDecodeError:
                                                yield {
                                                    "status": "error",
                                                    "message": f"无法解析JSON: {data_str}"
                                                }
                except Exception as e:
                    yield {
                        "status": "error",
                        "message": f"流式响应失败: {str(e)}"
                    }
            
            return stream_generator()
        
        # 处理非流式输出
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{base_url}/completions",
                    headers=headers,
                    json=data
                )
                
                response_time = round((time.time() - start_time) * 1000)
                
                if response.status_code != 200:
                    return {
                        "status": "error",
                        "message": f"API返回错误: {response.status_code} - {response.text}"
                    }
                
                result = response.json()
                result["response_time_ms"] = response_time
                return result
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"文本完成请求失败: {str(e)}"
            }
    
    async def embedding(
        self,
        api_key: str,
        text: Union[str, List[str]],
        model: str,
        base_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """获取文本嵌入向量从自定义API"""
        if not base_url:
            return {
                "status": "error",
                "message": "自定义API必须提供基础URL"
            }
        
        try:
            start_time = time.time()
            
            # 构建请求数据
            data = {
                "model": model,
                "input": text
            }
                
            # 添加其他参数
            for key, value in kwargs.items():
                if value is not None:
                    data[key] = value
            
            # 发送请求
            async with httpx.AsyncClient(timeout=60.0) as client:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                response = await client.post(
                    f"{base_url}/embeddings",
                    headers=headers,
                    json=data
                )
                
                response_time = round((time.time() - start_time) * 1000)
                
                if response.status_code != 200:
                    return {
                        "status": "error",
                        "message": f"API返回错误: {response.status_code} - {response.text}"
                    }
                
                result = response.json()
                result["response_time_ms"] = response_time
                return result
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"嵌入请求失败: {str(e)}"
            } 
import asyncio
import time
import os
import base64
from typing import Dict, Any, Optional, List, Union, AsyncGenerator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.completion import Completion
from openai.types.create_embedding_response import CreateEmbeddingResponse

from app.providers.base import ModelProvider


class OpenAIProvider(ModelProvider):
    """
    OpenAI模型提供商实现
    """
    
    @property
    def provider_id(self) -> str:
        return "openai"
    
    @property
    def provider_name(self) -> str:
        return "OpenAI"
    
    @property
    def description(self) -> str:
        return "OpenAI提供的AI模型，包括GPT-3.5和GPT-4系列"
    
    @property
    def icon(self) -> Optional[str]:
        return "https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg"
    
    @property
    def default_base_url(self) -> Optional[str]:
        return "https://api.openai.com/v1"
    
    @property
    def supported_model_types(self) -> List[str]:
        return ["chat", "completion", "embedding"]
    
    @property
    def features(self) -> List[str]:
        return ["流式响应", "函数调用", "并发请求", "图片识别"]
    
    async def test_connection(self, api_key: str, base_url: Optional[str] = None) -> Dict[str, Any]:
        """测试与OpenAI API的连接"""
        try:
            start_time = time.time()
            
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url or self.default_base_url
            )
            
            # 尝试获取模型列表，这是一个轻量级操作
            models = await client.models.list()
            
            response_time = round((time.time() - start_time) * 1000)
            
            return {
                "status": "success",
                "message": "连接成功",
                "response": {
                    "model": "OpenAI API",
                    "available_models": [model.id for model in models.data[:5]],
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
        """发送聊天完成请求到OpenAI API，支持流式输出"""
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or self.default_base_url
        )
        
        # 设置请求参数
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        # 添加可选参数
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # 添加其他参数
        for key, value in kwargs.items():
            params[key] = value
        
        start_time = time.time()
        
        # 处理流式输出
        if stream:
            async def stream_generator():
                try:
                    stream_resp = await client.chat.completions.create(**params)
                    async for chunk in stream_resp:
                        # 计算当前响应时间
                        current_time = time.time()
                        response_time = round((current_time - start_time) * 1000)
                        
                        # 构建chunk响应
                        chunk_data = chunk.model_dump()
                        chunk_data["response_time_ms"] = response_time
                        yield chunk_data
                except Exception as e:
                    yield {
                        "status": "error",
                        "message": f"流式响应失败: {str(e)}"
                    }
            
            return stream_generator()
        
        # 处理非流式输出
        try:
            response: ChatCompletion = await client.chat.completions.create(**params)
            response_time = round((time.time() - start_time) * 1000)
            
            # 处理响应
            result = response.model_dump()
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
        """发送文本完成请求到OpenAI API，支持流式输出"""
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or self.default_base_url
        )
        
        # 设置请求参数
        params = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": stream
        }
        
        # 添加可选参数
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # 添加其他参数
        for key, value in kwargs.items():
            params[key] = value
        
        start_time = time.time()
        
        # 处理流式输出
        if stream:
            async def stream_generator():
                try:
                    stream_resp = await client.completions.create(**params)
                    async for chunk in stream_resp:
                        # 计算当前响应时间
                        current_time = time.time()
                        response_time = round((current_time - start_time) * 1000)
                        
                        # 构建chunk响应
                        chunk_data = chunk.model_dump()
                        chunk_data["response_time_ms"] = response_time
                        yield chunk_data
                except Exception as e:
                    yield {
                        "status": "error",
                        "message": f"流式响应失败: {str(e)}"
                    }
            
            return stream_generator()
        
        # 处理非流式输出
        try:
            response: Completion = await client.completions.create(**params)
            response_time = round((time.time() - start_time) * 1000)
            
            # 处理响应
            result = response.model_dump()
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
        """获取文本嵌入向量从OpenAI API"""
        try:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url or self.default_base_url
            )
            
            # 设置请求参数
            params = {
                "model": model,
                "input": text,
            }
            
            # 添加其他参数
            for key, value in kwargs.items():
                params[key] = value
            
            # 发送请求
            start_time = time.time()
            response: CreateEmbeddingResponse = await client.embeddings.create(**params)
            response_time = round((time.time() - start_time) * 1000)
            
            # 处理响应
            result = response.model_dump()
            result["response_time_ms"] = response_time
            return result
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"嵌入向量请求失败: {str(e)}"
            }
            
    async def image_analysis(
        self,
        api_key: str,
        image_path: str,
        prompt: str = "请描述这张图片的内容",
        model: str = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """分析图片内容"""
        try:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url or self.default_base_url
            )
            
            # 设置默认多模态模型
            if model is None:
                model = "gpt-4o"
                
            # 根据图片路径读取图片数据
            image_data = None
            if image_path.startswith(('http://', 'https://')):
                # 处理URL图片
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_path) as response:
                        if response.status == 200:
                            image_data = await response.read()
                        else:
                            raise Exception(f"获取图片失败，HTTP状态码：{response.status}")
            else:
                # 处理本地图片
                if not os.path.exists(image_path):
                    raise Exception(f"图片文件不存在：{image_path}")
                with open(image_path, "rb") as f:
                    image_data = f.read()

            # 转换图片为base64
            if image_data:
                base64_image = base64.b64encode(image_data).decode('utf-8')
            else:
                raise Exception("无法读取图片数据")
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # 设置请求参数
            params = {
                "model": model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }
            
            # 发送请求
            start_time = time.time()
            response = await client.chat.completions.create(**params)
            response_time = round((time.time() - start_time) * 1000)
            
            # 处理响应
            result = response.model_dump()
            result["response_time_ms"] = response_time
            return {
                "status": "success",
                "model": model,
                "analysis": result["choices"][0]["message"]["content"] if result["choices"] else "",
                "response_time_ms": response_time,
                "full_response": result
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"图片分析失败: {str(e)}"
            } 
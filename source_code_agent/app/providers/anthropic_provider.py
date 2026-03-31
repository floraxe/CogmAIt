import json
import time
import os
import base64
from typing import Dict, Any, Optional, List, Union, AsyncGenerator

import anthropic
from anthropic import Anthropic
from anthropic._types import NotGiven

from app.providers.base import ModelProvider


class AnthropicProvider(ModelProvider):
    """
    Anthropic模型提供商实现
    """
    
    @property
    def provider_id(self) -> str:
        return "anthropic"
    
    @property
    def provider_name(self) -> str:
        return "Anthropic"
    
    @property
    def description(self) -> str:
        return "Anthropic提供的AI模型，包括Claude系列"
    
    @property
    def icon(self) -> Optional[str]:
        return "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/680790a96fa4599db312dc7b_opengraph.png"
    
    @property
    def default_base_url(self) -> Optional[str]:
        return "https://api.anthropic.com"
    
    @property
    def supported_model_types(self) -> List[str]:
        return ["chat"]
    
    @property
    def features(self) -> List[str]:
        return ["流式响应", "工具调用", "高安全性", "图片识别"]
    
    async def test_connection(self, api_key: str, base_url: Optional[str] = None) -> Dict[str, Any]:
        """测试与Anthropic API的连接"""
        try:
            start_time = time.time()
            
            client = Anthropic(
                api_key=api_key,
                base_url=base_url or self.default_base_url
            )
            
            # 尝试发送一个简单的消息来测试连接
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=10,
                messages=[
                    {"role": "user", "content": "Hello, this is a test."}
                ]
            )
            
            response_time = round((time.time() - start_time) * 1000)
            
            return {
                "status": "success",
                "message": "连接成功",
                "response": {
                    "model": response.model,
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
        """发送聊天完成请求到Anthropic API，支持流式输出"""
        client = Anthropic(
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
                    with client.messages.stream(**params) as stream:
                        async for chunk in stream:
                            # 计算当前响应时间
                            current_time = time.time()
                            response_time = round((current_time - start_time) * 1000)
                            
                            # 构建流式响应片段
                            if chunk.type == "content_block_delta":
                                yield {
                                    "id": stream.message_id,
                                    "model": model,
                                    "delta": {
                                        "content": chunk.delta.text,
                                        "role": "assistant"
                                    },
                                    "finish_reason": None,
                                    "response_time_ms": response_time
                                }
                            elif chunk.type == "message_stop":
                                yield {
                                    "id": stream.message_id,
                                    "model": model,
                                    "delta": {
                                        "content": "",
                                        "role": "assistant"
                                    },
                                    "finish_reason": "stop",
                                    "response_time_ms": response_time
                                }
                except Exception as e:
                    yield {
                        "status": "error",
                        "message": f"流式响应失败: {str(e)}"
                    }
            
            return stream_generator()
        
        # 处理非流式输出
        try:
            response = client.messages.create(**params)
            response_time = round((time.time() - start_time) * 1000)
            
            # 处理响应
            result = {
                "id": response.id,
                "model": response.model,
                "content": response.content,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "response_time_ms": response_time
            }
            
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
        """发送文本完成请求到Anthropic API（将转换为聊天格式），支持流式输出"""
        # Anthropic不直接支持text_completion，转为chat_completion
        messages = [{"role": "user", "content": prompt}]
        
        return await self.chat_completion(
            api_key=api_key,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            stream=stream,
            **kwargs
        )
    
    async def embedding(
        self,
        api_key: str,
        text: Union[str, List[str]],
        model: str,
        base_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Anthropic目前不支持嵌入API
        注意：此方法是为兼容性保留，但会返回错误消息
        """
        return {
            "status": "error",
            "message": "Anthropic当前不提供嵌入API服务"
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
            client = Anthropic(
                api_key=api_key,
                base_url=base_url or self.default_base_url
            )
            
            # 设置默认多模态模型
            if model is None:
                model = "claude-3-opus-20240229"
            elif not model.startswith("claude-3"):
                model = "claude-3-opus-20240229"  # 确保使用支持图片的模型
                
            # 根据图片路径读取图片数据
            image_data = None
            media_type = "image/jpeg"  # 默认MIME类型
            
            if image_path.startswith(('http://', 'https://')):
                # 处理URL图片
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_path) as response:
                        if response.status == 200:
                            image_data = await response.read()
                            media_type = response.headers.get("Content-Type", "image/jpeg")
                        else:
                            raise Exception(f"获取图片失败，HTTP状态码：{response.status}")
            else:
                # 处理本地图片
                if not os.path.exists(image_path):
                    raise Exception(f"图片文件不存在：{image_path}")
                    
                # 根据文件扩展名确定媒体类型
                ext = os.path.splitext(image_path)[1].lower()
                if ext in ['.jpg', '.jpeg']:
                    media_type = "image/jpeg"
                elif ext == '.png':
                    media_type = "image/png"
                elif ext == '.gif':
                    media_type = "image/gif"
                elif ext == '.webp':
                    media_type = "image/webp"
                
                with open(image_path, "rb") as f:
                    image_data = f.read()

            # 转换图片为base64
            if not image_data:
                raise Exception("无法读取图片数据")
                
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
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
            response = client.messages.create(**params)
            response_time = round((time.time() - start_time) * 1000)
            
            # 处理响应
            analysis_text = ""
            for content_block in response.content:
                if content_block.type == "text":
                    analysis_text += content_block.text
            
            return {
                "status": "success",
                "model": model,
                "analysis": analysis_text,
                "response_time_ms": response_time,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"图片分析失败: {str(e)}"
            } 
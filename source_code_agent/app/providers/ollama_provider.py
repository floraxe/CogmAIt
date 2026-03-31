import asyncio
import json
import time
import os
import base64
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
import aiohttp
from aiohttp.client_exceptions import ClientError

from app.providers.base import ModelProvider


class OllamaProvider(ModelProvider):
    """
    Ollama模型提供商实现
    提供对本地或远程Ollama服务的访问
    """
    
    @property
    def provider_id(self) -> str:
        return "ollama"
    
    @property
    def provider_name(self) -> str:
        return "Ollama"
    
    @property
    def description(self) -> str:
        return "Ollama提供的本地或远程运行的开源大语言模型，支持Llama、Mistral等多种模型"
    
    @property
    def icon(self) -> Optional[str]:
        return "https://ollama.com/public/ollama.png"
    
    @property
    def default_base_url(self) -> Optional[str]:
        return "http://localhost:11434"
    
    @property
    def supported_model_types(self) -> List[str]:
        return ["chat", "completion", "embedding"]
    
    @property
    def features(self) -> List[str]:
        return ["本地模型", "流式响应", "自定义模型", "图片识别"]
    
    async def test_connection(self, api_key: str, base_url: Optional[str] = None) -> Dict[str, Any]:
        """测试与Ollama API的连接"""
        try:
            start_time = time.time()
            
            # Ollama不需要API密钥，但我们保留参数以兼容接口
            url = f"{base_url or self.default_base_url}/api/tags"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return {
                            "status": "failed",
                            "message": f"连接失败: HTTP {response.status}"
                        }
                    
                    data = await response.json()
                    
            response_time = round((time.time() - start_time) * 1000)
            
            # 提取可用模型列表
            models = []
            if "models" in data:
                models = [model["name"] for model in data["models"][:5]]
            
            return {
                "status": "success",
                "message": "连接成功",
                "response": {
                    "model": "Ollama API",
                    "available_models": models,
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
        """发送聊天完成请求到Ollama API，支持流式输出"""
        # Ollama的聊天API端点
        url = f"{base_url or self.default_base_url}/api/chat"
        
        # 处理系统消息和用户消息
        formatted_messages = []
        for msg in messages:
            if "role" in msg and "content" in msg:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # 设置请求参数
        params = {
            "model": model,
            "messages": formatted_messages,
            "stream": stream,
            "options": {
                "temperature": temperature
            }
        }
        
        # 处理max_tokens参数
        if max_tokens is not None:
            params["options"]["num_predict"] = max_tokens
        
        # 添加其他参数到options
        for key, value in kwargs.items():
            if key not in ["model", "messages", "stream", "options"]:
                params["options"][key] = value
        
        start_time = time.time()
        
        # 处理流式输出
        if stream:
            async def stream_generator():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, json=params) as response:
                            if response.status != 200:
                                yield {
                                    "status": "error",
                                    "message": f"请求失败: HTTP {response.status}"
                                }
                                return
                            
                            # 获取流式响应
                            buffer = ""
                            async for line in response.content:
                                if line:
                                    buffer += line.decode('utf-8')
                                    if buffer.endswith('\n'):
                                        # 处理完整的JSON对象
                                        try:
                                            chunk = json.loads(buffer.strip())
                                            current_time = time.time()
                                            response_time = round((current_time - start_time) * 1000)
                                            
                                            # 转换Ollama格式到标准格式
                                            result = {
                                                "id": f"chatcmpl-{time.time()}",
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": model,
                                                "choices": [
                                                    {
                                                        "index": 0,
                                                        "delta": {
                                                            "role": "assistant",
                                                            "content": chunk.get("message", {}).get("content", "")
                                                        },
                                                        "finish_reason": "stop" if chunk.get("done", False) else None
                                                    }
                                                ],
                                                "response_time_ms": response_time
                                            }
                                            # 将结果转换为JSON字符串
                                            yield json.dumps(result,ensure_ascii=False)
                                            
                                            if chunk.get("done", False):
                                                break
                                                
                                        except json.JSONDecodeError:
                                            # 处理不完整的JSON
                                            pass
                                        
                                        buffer = ""
                except Exception as e:
                    yield {
                        "status": "error",
                        "message": f"流式响应失败: {str(e)}"
                    }
            
            return stream_generator()
        
        # 处理非流式输出
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=params) as response:
                    if response.status != 200:
                        return {
                            "status": "error",
                            "message": f"聊天完成请求失败: HTTP {response.status}"
                        }
                    
                    data = await response.json()
            
            response_time = round((time.time() - start_time) * 1000)
            
            # 转换Ollama响应格式为类似OpenAI的格式
            result = {
                "id": f"chatcmpl-{time.time()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": data.get("message", {}).get("content", "")
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": -1,  # Ollama不提供token统计
                    "completion_tokens": -1,
                    "total_tokens": -1
                },
                "response_time_ms": response_time
            }
            # 将结果转换为JSON字符串
            return json.dumps(result,ensure_ascii=False)
        
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
        """发送文本完成请求到Ollama API，支持流式输出"""
        # Ollama的生成端点
        url = f"{base_url or self.default_base_url}/api/generate"
        
        # 设置请求参数
        params = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature
            }
        }
        
        # 处理max_tokens参数
        if max_tokens is not None:
            params["options"]["num_predict"] = max_tokens
        
        # 添加其他参数到options
        for key, value in kwargs.items():
            if key not in ["model", "prompt", "stream", "options"]:
                params["options"][key] = value
        
        start_time = time.time()
        
        # 处理流式输出
        if stream:
            async def stream_generator():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, json=params) as response:
                            if response.status != 200:
                                yield {
                                    "status": "error",
                                    "message": f"请求失败: HTTP {response.status}"
                                }
                                return
                            
                            # 获取流式响应
                            buffer = ""
                            async for line in response.content:
                                if line:
                                    buffer += line.decode('utf-8')
                                    if buffer.endswith('\n'):
                                        # 处理完整的JSON对象
                                        try:
                                            chunk = json.loads(buffer.strip())
                                            current_time = time.time()
                                            response_time = round((current_time - start_time) * 1000)
                                            
                                            # 转换Ollama格式到标准格式
                                            result = {
                                                "id": f"cmpl-{time.time()}",
                                                "object": "text_completion.chunk",
                                                "created": int(time.time()),
                                                "model": model,
                                                "choices": [
                                                    {
                                                        "text": chunk.get("response", ""),
                                                        "index": 0,
                                                        "finish_reason": "stop" if chunk.get("done", False) else None
                                                    }
                                                ],
                                                "response_time_ms": response_time
                                            }
                                            # 将结果转换为JSON字符串
                                            yield json.dumps(result,ensure_ascii=False)
                                            
                                            if chunk.get("done", False):
                                                break
                                                
                                        except json.JSONDecodeError:
                                            # 处理不完整的JSON
                                            pass
                                        
                                        buffer = ""
                except Exception as e:
                    yield {
                        "status": "error",
                        "message": f"流式响应失败: {str(e)}"
                    }
            
            return stream_generator()
        
        # 处理非流式输出
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=params) as response:
                    if response.status != 200:
                        return {
                            "status": "error",
                            "message": f"文本完成请求失败: HTTP {response.status}"
                        }
                    
                    data = await response.json()
            
            response_time = round((time.time() - start_time) * 1000)
            
            # 转换Ollama响应格式为类似OpenAI的格式
            result = {
                "id": f"cmpl-{time.time()}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "text": data.get("response", ""),
                        "index": 0,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": -1,  # Ollama不提供token统计
                    "completion_tokens": -1,
                    "total_tokens": -1
                },
                "response_time_ms": response_time
            }
            # 将结果转换为JSON字符串
            return json.dumps(result,ensure_ascii=False)
        
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
        """获取文本嵌入向量从Ollama API"""
        # Ollama的嵌入端点
        url = f"{base_url or self.default_base_url}/api/embeddings"
        # print('请求URL：：',url)
        # 处理输入文本
        input_texts = []
        if isinstance(text, str):
            input_texts = [text]
        else:
            input_texts = text
        
        # 由于Ollama的API只能一次处理一个文本，我们需要逐个处理
        embeddings_list = []
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                for input_text in input_texts:
                    params = {
                        "model": model,
                        "prompt": input_text
                    }
                    # print("请求参数::",params)
                    # 添加其他参数
                    for key, value in kwargs.items():
                        if key not in ["model", "prompt"]:
                            params[key] = value
                    
                    async with session.post(url, json=params) as response:
                        if response.status != 200:
                            return {
                                "status": "error",
                                "message": f"嵌入请求失败: HTTP {response.status}"
                            }
                        data = await response.json()
                        # print(data)
                        
                        if "embedding" in data:
                            embeddings_list.append(data["embedding"])
            
            response_time = round((time.time() - start_time) * 1000)
            
            # 转换为类似OpenAI的响应格式
            result = {
                "object": "embedding_list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": embedding,
                        "index": i
                    }
                    for i, embedding in enumerate(embeddings_list)
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": -1,  # Ollama不提供token统计
                    "total_tokens": -1
                },
                "embeddings": embeddings_list,  # 添加直接可用的嵌入列表
                "response_time_ms": response_time
            }
            # print(result)
            return result
        
        except Exception as e:
            print(e)
            return {
                "status": "error",
                "message": f"嵌入请求失败: {str(e)}"
            }
    
    async def image_analysis(
        self,
        api_key: str,
        image_path: str,
        prompt: str = "请详细描述这张图片的内容",
        model: str = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用Ollama多模态模型分析图片内容
        
        Args:
            api_key: API密钥（Ollama不需要但保留参数以兼容接口）
            image_path: 图片文件路径或URL
            prompt: 提示文本，引导模型如何分析图片
            model: 模型名称，默认为"llava"（多模态模型）
            base_url: API基础URL，可选
            **kwargs: 其他参数
            
        Returns:
            Dict包含图片分析结果
        """
        try:
            # 设置默认多模态模型
            if model is None or not model:
                model = "llava:latest"
                
            # 日志记录
            print(f"使用Ollama多模态模型 {model} 分析图片: {image_path}")
            
            # 读取图片数据并转换为base64
            image_data = None
            if image_path.startswith(('http://', 'https://')):
                # 处理URL图片
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_path) as response:
                        if response.status != 200:
                            raise Exception(f"获取图片失败，HTTP状态码：{response.status}")
                        image_data = await response.read()
            else:
                # 处理本地图片
                if not os.path.exists(image_path):
                    raise Exception(f"图片文件不存在：{image_path}")
                with open(image_path, "rb") as f:
                    image_data = f.read()
                    
            # 确认图片数据已获取
            if not image_data:
                raise Exception("无法读取图片数据")
                
            # 将图片编码为base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Ollama的聊天API端点
            url = f"{base_url or self.default_base_url}/api/chat"
            
            # 构建带图像的消息
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64_image]
                }
            ]
            
            # 设置请求参数
            params = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7)
                }
            }
            
            # 处理max_tokens参数
            if "max_tokens" in kwargs:
                params["options"]["num_predict"] = kwargs["max_tokens"]
                
            start_time = time.time()
            
            # 发送请求
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=params) as response:
                    if response.status != 200:
                        return {
                            "status": "error",
                            "message": f"图片分析请求失败: HTTP {response.status}",
                            "response_time_ms": round((time.time() - start_time) * 1000)
                        }
                    
                    data = await response.json()
                    
            response_time = round((time.time() - start_time) * 1000)
            print(f"图片分析完成，响应时间: {response_time}ms")
            
            # 从响应中提取文本内容
            analysis_text = ""
            if "message" in data and "content" in data["message"]:
                analysis_text = data["message"]["content"]
            
            return {
                "status": "success",
                "model": model,
                "analysis": analysis_text,
                "response_time_ms": response_time
            }
            
        except Exception as e:
            print(f"图片分析失败: {str(e)}")
            return {
                "status": "error",
                "message": f"图片分析失败: {str(e)}",
                "response_time_ms": -1
            } 
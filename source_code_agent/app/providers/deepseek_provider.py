import json
import time
from typing import Dict, Any, Optional, List, Union, AsyncGenerator

import httpx

from app.providers.base import ModelProvider


class DeepSeekProvider(ModelProvider):
    """
    DeepSeek 模型提供商实现（OpenAI 兼容接口）。
    使用 httpx 进行异步 HTTP 调用。
    """

    @property
    def provider_id(self) -> str:
        return "deepseek"

    @property
    def provider_name(self) -> str:
        return "DeepSeek"

    @property
    def description(self) -> str:
        return "DeepSeek 提供的模型服务，兼容 OpenAI 风格接口"

    @property
    def icon(self) -> Optional[str]:
        return "https://www.deepseek.com/favicon.ico"

    @property
    def default_base_url(self) -> Optional[str]:
        return "https://api.deepseek.com/v1"

    @property
    def supported_model_types(self) -> List[str]:
        return ["chat", "completion", "embedding"]

    @property
    def features(self) -> List[str]:
        return ["流式响应", "OpenAI兼容接口", "异步调用"]

    def _build_headers(self, api_key: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def test_connection(self, api_key: str, base_url: Optional[str] = None) -> Dict[str, Any]:
        """测试与 DeepSeek API 的连接。"""
        endpoint = f"{base_url or self.default_base_url}/models"
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(endpoint, headers=self._build_headers(api_key))
                resp.raise_for_status()
                data = resp.json()

            models = [m.get("id") for m in data.get("data", []) if m.get("id")]
            response_time = round((time.time() - start_time) * 1000)
            return {
                "status": "success",
                "message": "连接成功",
                "response": {
                    "model": "DeepSeek API",
                    "available_models": models[:5],
                    "responseTime": f"{response_time}ms",
                },
            }
        except Exception as e:
            return {
                "status": "failed",
                "message": f"连接失败: {str(e)}",
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
        **kwargs,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """发送聊天完成请求到 DeepSeek API。"""
        endpoint = f"{base_url or self.default_base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        start_time = time.time()
        headers = self._build_headers(api_key)

        if stream:
            async def stream_generator() -> AsyncGenerator[Dict[str, Any], None]:
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        async with client.stream("POST", endpoint, headers=headers, json=payload) as resp:
                            resp.raise_for_status()
                            async for line in resp.aiter_lines():
                                if not line or not line.startswith("data:"):
                                    continue
                                data_line = line[5:].strip()
                                if data_line == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_line)
                                except json.JSONDecodeError:
                                    continue
                                chunk["response_time_ms"] = round((time.time() - start_time) * 1000)
                                yield chunk
                except Exception as e:
                    yield {"status": "error", "message": f"流式响应失败: {str(e)}"}

            return stream_generator()

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(endpoint, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            data["response_time_ms"] = round((time.time() - start_time) * 1000)
            return data
        except Exception as e:
            return {"status": "error", "message": f"聊天完成请求失败: {str(e)}"}

    async def text_completion(
        self,
        api_key: str,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """发送文本完成请求到 DeepSeek API（OpenAI 兼容 completions 端点）。"""
        endpoint = f"{base_url or self.default_base_url}/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        start_time = time.time()
        headers = self._build_headers(api_key)

        if stream:
            async def stream_generator() -> AsyncGenerator[Dict[str, Any], None]:
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        async with client.stream("POST", endpoint, headers=headers, json=payload) as resp:
                            resp.raise_for_status()
                            async for line in resp.aiter_lines():
                                if not line or not line.startswith("data:"):
                                    continue
                                data_line = line[5:].strip()
                                if data_line == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_line)
                                except json.JSONDecodeError:
                                    continue
                                chunk["response_time_ms"] = round((time.time() - start_time) * 1000)
                                yield chunk
                except Exception as e:
                    yield {"status": "error", "message": f"流式响应失败: {str(e)}"}

            return stream_generator()

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(endpoint, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            data["response_time_ms"] = round((time.time() - start_time) * 1000)
            return data
        except Exception as e:
            return {"status": "error", "message": f"文本完成请求失败: {str(e)}"}

    async def embedding(
        self,
        api_key: str,
        text: Union[str, List[str]],
        model: str,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """发送嵌入请求到 DeepSeek API。"""
        endpoint = f"{base_url or self.default_base_url}/embeddings"
        payload: Dict[str, Any] = {
            "model": model,
            "input": text,
        }
        payload.update(kwargs)

        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    endpoint,
                    headers=self._build_headers(api_key),
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            data["response_time_ms"] = round((time.time() - start_time) * 1000)
            return data
        except Exception as e:
            return {"status": "error", "message": f"嵌入向量请求失败: {str(e)}"}
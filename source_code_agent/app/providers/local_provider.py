import time
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from app.providers.base import ModelProvider

class LocalProvider(ModelProvider):
    """
    本地模型提供商实现
    使用ModelScope框架加载本地或托管的开源模型
    """
    
    _model_cache = {}  # 类级别模型缓存
    _tokenizer_cache = {}  # 类级别分词器缓存

    @property
    def provider_id(self) -> str:
        return "local"
    
    @property
    def provider_name(self) -> str:
        return "本地模型"
    
    @property
    def description(self) -> str:
        return "使用ModelScope框架加载本地或托管的开源模型，支持聊天和文本生成"
    
    @property
    def icon(self) -> Optional[str]:
        return "//img.alicdn.com/imgextra/i4/O1CN01fvt4it25rEZU4Gjso_!!6000000007579-2-tps-128-128.png"
    
    @property
    def default_base_url(self) -> Optional[str]:
        return None  # 需要用户指定模型路径
    
    @property
    def supported_model_types(self) -> List[str]:
        return ["chat", "completion"]
    
    @property
    def features(self) -> List[str]:
        return ["本地推理", "自定义模型"]
    
    async def test_connection(self, api_key: str, base_url: Optional[str] = None) -> Dict[str, Any]:
        """测试模型加载能力"""
        try:
            start_time = time.time()
            model_name = base_url or ""
            
            # 尝试加载模型和分词器
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            
            # 更新缓存
            self.__class__._model_cache[model_name] = model
            self.__class__._tokenizer_cache[model_name] = tokenizer
            
            response_time = round((time.time() - start_time) * 1000)
            return {
                "status": "success",
                "message": "模型加载成功",
                "response": {
                    "model": model_name,
                    "available_models": [model_name],
                    "responseTime": f"{response_time}ms"
                }
            }
        except Exception as e:
            return {
                "status": "failed",
                "message": f"模型加载失败: {str(e)}"
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
        """聊天补全接口"""
        if stream:
            async def error_stream():
                yield {"status": "error", "message": "本地模型暂不支持流式输出"}
            return error_stream()

        start_time = time.time()
        try:
            # 获取或加载模型
            tokenizer = self._tokenizer_cache.get(model)
            model_inst = self._model_cache.get(model)
            if not tokenizer or not model_inst:
                tokenizer = AutoTokenizer.from_pretrained(model)
                model_inst = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype="auto",
                    device_map="auto"
                )
                self._model_cache[model] = model_inst
                self._tokenizer_cache[model] = tokenizer

            # 构建模型输入
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(model_inst.device)

            # 生成配置
            generate_kwargs = {
                "max_new_tokens": max_tokens or 512,
                "temperature": temperature,
                **kwargs
            }

            # 执行推理
            outputs = model_inst.generate(**inputs, **generate_kwargs)
            output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()

            # 解析输出
            try:
                split_idx = len(output_ids) - output_ids[::-1].index(151668)
                content = tokenizer.decode(output_ids[split_idx:], skip_special_tokens=True).strip()
            except ValueError:
                content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            response_time = round((time.time() - start_time) * 1000)
            
            return {
                "id": f"chatcmpl-{time.time()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1
                },
                "response_time_ms": response_time
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"聊天请求失败: {str(e)}"
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
        """文本补全接口"""
        if stream:
            async def error_stream():
                yield {"status": "error", "message": "本地模型暂不支持流式输出"}
            return error_stream()

        start_time = time.time()
        try:
            # 获取或加载模型
            tokenizer = self._tokenizer_cache.get(model)
            model_inst = self._model_cache.get(model)
            if not tokenizer or not model_inst:
                tokenizer = AutoTokenizer.from_pretrained(model)
                model_inst = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype="auto",
                    device_map="auto"
                )
                self._model_cache[model] = model_inst
                self._tokenizer_cache[model] = tokenizer

            # 构建输入
            inputs = tokenizer([prompt], return_tensors="pt").to(model_inst.device)

            # 生成配置
            generate_kwargs = {
                "max_new_tokens": max_tokens or 512,
                "temperature": temperature,
                **kwargs
            }

            # 执行推理
            outputs = model_inst.generate(**inputs, **generate_kwargs)
            output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            response_time = round((time.time() - start_time) * 1000)
            
            return {
                "id": f"cmpl-{time.time()}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "text": content,
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1
                },
                "response_time_ms": response_time
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"文本生成失败: {str(e)}"
            }
    
    async def embedding(
        self,
        api_key: str,
        text: Union[str, List[str]],
        model: str,
        base_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        return {
            "status": "error",
            "message": "本地模型暂不支持嵌入功能"
        }

    def to_provider_info(self) -> Dict[str, Any]:
        """转换为提供商信息字典"""
        return {
            "value": self.provider_id,
            "name": self.provider_name,
            "description": self.description,
            "icon": self.icon,
            "default_base_url": self.default_base_url,
            "supported_types": self.supported_model_types,
            "features": self.features
        }
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
import aiohttp
import json
import asyncio

from app.schemas.model import ProviderInfo

class ModelProvider(ABC):
    """
    模型提供商的基础抽象类。
    所有模型提供商必须继承此类并实现其方法。
    """
    
    @property
    @abstractmethod
    def provider_id(self) -> str:
        """提供商唯一ID"""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """提供商名称，用于显示"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """提供商描述"""
        pass
    
    @property
    @abstractmethod
    def icon(self) -> Optional[str]:
        """提供商图标URL，可选"""
        pass
    
    @property
    @abstractmethod
    def default_base_url(self) -> Optional[str]:
        """默认API基础URL，如无则返回None"""
        pass
    
    @property
    @abstractmethod
    def supported_model_types(self) -> List[str]:
        """支持的模型类型列表，如chat, completion, embedding等"""
        pass
    
    @property
    @abstractmethod
    def features(self) -> List[str]:
        """支持的特性列表"""
        pass
    
    @abstractmethod
    async def test_connection(self, api_key: str, base_url: Optional[str] = None) -> Dict[str, Any]:
        """
        测试与模型提供商的连接
        
        Args:
            api_key: API密钥
            base_url: API基础URL，可选
            
        Returns:
            Dict包含连接测试结果
        """
        pass
    
    @abstractmethod
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
        """
        发送聊天完成请求到模型提供商
        
        Args:
            api_key: API密钥
            messages: 消息列表，每个消息包含'role'和'content'
            model: 模型名称
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成token数
            base_url: API基础URL，可选
            stream: 是否使用流式输出
            **kwargs: 其他参数
            
        Returns:
            Dict包含完成结果，或在流式模式下返回AsyncGenerator
        """
        pass
    
    @abstractmethod
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
        """
        发送文本完成请求到模型提供商
        
        Args:
            api_key: API密钥
            prompt: 提示文本
            model: 模型名称
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成token数
            base_url: API基础URL，可选
            stream: 是否使用流式输出
            **kwargs: 其他参数
            
        Returns:
            Dict包含完成结果，或在流式模式下返回AsyncGenerator
        """
        pass
    
    @abstractmethod
    async def embedding(
        self,
        api_key: str,
        text: Union[str, List[str]],
        model: str,
        base_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        获取文本嵌入向量
        
        Args:
            api_key: API密钥
            text: 单个文本或文本列表
            model: 模型名称
            base_url: API基础URL，可选
            **kwargs: 其他参数
            
        Returns:
            Dict包含嵌入结果
        """
        pass
    
    def to_provider_info(self) -> Dict[str, Any]:
        """
        将提供商信息转换为字典
        
        返回:
            Dict[str, Any]: 提供商信息字典
        """
        return {
            "value": self.provider_id,
            "name": self.provider_name,
            "description": self.description,
            "icon": self.icon,
            "default_base_url": self.default_base_url,
            "supported_types": self.supported_model_types,
            "features": self.features
        } 

class MCPServiceProvider(ABC):
    """MCP服务提供商基类，所有MCP服务提供商实现都应该继承此类"""
    
    # 必须由子类定义
    provider_id: str       # 提供商唯一标识符
    provider_name: str     # 提供商显示名称
    
    def __init__(self):
        """初始化MCP服务提供商"""
        pass
    
    @abstractmethod
    async def call_function(self, service: Any, function_name: str, params: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        调用MCP服务功能
        
        参数:
            service (Any): 服务对象
            function_name (str): 功能名称
            params (Dict[str, Any]): 调用参数
            user_id (Optional[str]): 用户ID，用于记录使用统计
            
        返回:
            Dict[str, Any]: 调用结果
        """
        pass
    
    async def test_connection(self, service: Any) -> Dict[str, Any]:
        """
        测试连接，验证服务配置是否有效
        
        参数:
            service (Any): 服务对象
            
        返回:
            Dict[str, Any]: 测试结果
        """
        try:
            # 默认实现：尝试调用一个简单的测试功能
            result = await self.call_function(
                service=service,
                function_name="test",
                params={},
                user_id=None
            )
            
            return {
                "status": "success",
                "message": f"连接测试成功: {self.provider_name}",
                "response": result
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"连接测试失败: {str(e)}"
            }
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        获取提供商信息
        
        返回:
            Dict[str, Any]: 提供商信息
        """
        return {
            "value": self.provider_id,
            "name": self.provider_name
        }
    
    def get_supported_functions(self) -> List[Dict[str, Any]]:
        """
        获取支持的功能列表
        
        返回:
            List[Dict[str, Any]]: 功能列表
        """
        return [] 
import httpx
import json
from typing import Dict, List, Any, Optional
import os
from fastapi import HTTPException
from app.config.settings import TAVILY_API_KEY
from tavily import TavilyClient
class WebSearchClient:
    """
    网络搜索客户端，用于进行互联网搜索
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化搜索客户端
        
        参数:
            api_key: API密钥，如果未提供则使用环境变量中的API密钥
        """
        self.api_key = api_key or TAVILY_API_KEY
        if not self.api_key:
            raise ValueError("未配置Tavily API密钥，请在环境变量中设置TAVILY_API_KEY或在初始化时提供")
        print(self.api_key)
        self.base_url = "https://api.tavily.com/v1"
    
    async def search(self, query: str, search_depth: str = "basic", max_results: int = 5) -> Dict[str, Any]:
        """
        执行网络搜索
        
        参数:
            query: 搜索查询
            search_depth: 搜索深度，可以是"basic"或"advanced"
            max_results: 最大结果数量
            
        返回:
            搜索结果
        """
        try:
            

            # Step 1. Instantiating your TavilyClient
            tavily_client = TavilyClient(api_key= self.api_key)

            # Step 2. Executing a simple search query
            response = tavily_client.search(query)
            # async with httpx.AsyncClient(timeout=30.0) as client:
            #     response = await client.post(
            #         f"{self.base_url}/search",
            #         headers={"Content-Type": "application/json", "X-API-Key": self.api_key},
            #         json={
            #             "query": query,
            #             "search_depth": search_depth,
            #             "max_results": max_results
            #         }
            #     )
                
            # if response.status_code != 200:
            #     print(f"Tavily搜索请求失败: {response.status_code}, {response.text}")
            #     return {
            #         "success": False,
            #         "error": f"搜索请求失败: {response.status_code}",
            #         "results": []
            #     }
            print(response)
            return response
        except Exception as e:
            print(f"执行网络搜索时出错: {str(e)}")
            return {
                "success": False,
                "error": f"执行搜索时出错: {str(e)}",
                "results": []
            }
    
    def format_search_results(self, search_results: Dict[str, Any]) -> str:
        """
        将搜索结果格式化为可读文本，用于添加到提示中
        
        参数:
            search_results: 搜索结果
            
        返回:
            格式化的搜索结果文本
        """
        if not search_results.get("results"):
            return "未找到相关网络搜索结果。"
        
        formatted_text = "以下是来自互联网的相关信息：\n\n"
        
        for i, result in enumerate(search_results.get("results", []), 1):
            title = result.get("title", "无标题")
            content = result.get("content", "无内容")
            url = result.get("url", "无URL")
            
            formatted_text += f"[{i}] {title}\n"
            formatted_text += f"内容: {content}\n"
            formatted_text += f"来源: {url}\n\n"
        
        formatted_text += "请基于以上网络搜索结果和你自己的知识回答用户的问题。如果网络搜索结果与问题相关，请在回答中引用相关信息的来源编号(如[1],[2])。\n\n"
        
        return formatted_text


# 创建一个全局实例，方便调用
web_search_client = None

def get_web_search_client() -> WebSearchClient:
    """
    获取网络搜索客户端实例
    
    返回:
        WebSearchClient: 网络搜索客户端实例
    """
    global web_search_client
    
    if web_search_client is None:
        try:
            web_search_client = WebSearchClient()
        except ValueError as e:
            print(f"初始化网络搜索客户端失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"网络搜索服务不可用: {str(e)}")
    
    return web_search_client

async def search_web(query: str) -> Dict[str, Any]:
    """
    搜索网络
    
    参数:
        query: 搜索查询
        
    返回:
        搜索结果
    """
    client = get_web_search_client()
    results = await client.search(query)
    return results 
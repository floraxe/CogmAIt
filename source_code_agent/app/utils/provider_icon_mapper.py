"""
提供商图标映射工具

用于将提供商ID映射到对应的图标文件名
"""

from typing import Dict, Optional

# 提供商ID到图标文件名的映射
PROVIDER_ICON_MAP: Dict[str, str] = {
    "openai": "openai.svg",
    "anthropic": "claude.svg",
    "google": "gemini.svg",
    "ollama": "Ollama.svg",
    "cohere": "cohere.svg",
    "mistral": "mistral.svg",
    "local": "huggingface.svg",
    "custom": "custom.svg",
    "baidu": "ernie.svg",
    "zhipu": "chatglm.svg",
    "aliyun": "qwen.svg",
    "xunfei": "sparkDesk.svg",
    "360": "yi.svg",
    "hunyuan": "hunyuan.svg",
    "azure": "azure.svg",
    "meta": "meta.svg",
    "cloudflare": "cloudflare.svg",
    "jina": "jina.svg",
    "groq": "groq.svg",
    "deepseek": "deepseek.svg"
}

def get_icon_filename(provider_id: str) -> str:
    """
    根据提供商ID获取对应的图标文件名
    
    参数:
        provider_id (str): 提供商ID
    
    返回:
        str: 图标文件名，如果没有映射则使用provider_id.svg
    """
    return PROVIDER_ICON_MAP.get(provider_id, f"{provider_id}.svg")

def extract_icon_from_url(url: Optional[str], provider_id: str) -> str:
    """
    从URL中提取图标文件名，或返回默认图标
    
    参数:
        url (Optional[str]): 图标URL
        provider_id (str): 提供商ID，用于获取默认图标
    
    返回:
        str: 图标文件名
    """
    if not url:
        return get_icon_filename(provider_id)
        
    if not url.endswith('.svg') and not url.endswith('.png'):
        return get_icon_filename(provider_id)
        
    # 如果是URL，提取文件名
    if url.startswith('http://') or url.startswith('https://'):
        try:
            from urllib.parse import urlparse
            import os
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            
            # 如果文件扩展名不是.svg，使用默认图标
            if not filename.endswith('.svg'):
                return get_icon_filename(provider_id)
                
            return filename
        except Exception:
            # 如果解析失败，使用默认图标
            return get_icon_filename(provider_id)
    
    # 如果已经是文件名，检查是否是.svg
    if url.endswith('.svg'):
        return url
        
    # 默认返回映射图标
    return get_icon_filename(provider_id) 
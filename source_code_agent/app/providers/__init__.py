# 模型提供商包
# 导出提供商类，确保管理器能够自动发现和加载
from app.providers.openai_provider import OpenAIProvider
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.google_provider import GoogleProvider
from app.providers.custom_provider import CustomProvider
from app.providers.ollama_provider import OllamaProvider 
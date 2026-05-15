from app.models.model import Model
from app.providers.base import ModelProvider
from app.providers.manager import provider_manager


class ProviderFactory:
    """从模型记录解析 Provider 插件实例（工厂方法）。"""

    @staticmethod
    def get_for_model(model: Model) -> ModelProvider:
        return provider_manager.get_provider(model.provider)

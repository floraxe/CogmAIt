import logging
import traceback
from typing import Any, AsyncGenerator, Dict, Union

from sqlalchemy.orm import Session

from app.services.inference.context import InferenceContext
from app.services.inference.provider_factory import ProviderFactory
from app.services.inference.strategies.registry import get_inference_strategy
from app.utils.model import get_model

logger = logging.getLogger(__name__)


class ModelInferenceFacade:
    """
    模型推理统一入口（外观模式）。
    负责加载模型、解析 Provider，再按 model_type 分派到具体策略。
    """

    async def run(
        self,
        db: Session,
        model_id: str,
        payload: Dict[str, Any],
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        try:
            model = get_model(db, model_id)
            if not model:
                return {"error": f"找不到模型: {model_id}"}

            if model.status != "active":
                return {"error": f"模型状态不是活动状态: {model.status}"}

            provider = ProviderFactory.get_for_model(model)
            model_type = payload.get("model_type", model.type)
            strategy = get_inference_strategy(model_type)
            if strategy is None:
                return {"error": f"不支持的模型类型: {model_type}"}

            context = InferenceContext(
                model=model,
                provider=provider,
                payload=payload,
                model_type=model_type,
            )
            return await strategy.execute(context)
        except Exception as exc:
            logger.exception("model inference facade error model_id=%s", model_id)
            traceback.print_exc()
            return {"error": f"执行模型推理时出错: {str(exc)}"}


model_inference_facade = ModelInferenceFacade()

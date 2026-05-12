"""
联网搜索策略。

完全自包含：不依赖 RetrievalAugmentationService，
直接调用 search_web / get_web_search_client 工具函数。
"""
import logging
from typing import Any

from app.services.strategy_base import BaseRetrievalStrategy, StrategyContext, StrategyResult
from app.services import chat_events as ev
from app.utils.web_search import search_web, get_web_search_client

logger = logging.getLogger(__name__)


class WebSearchStrategy(BaseRetrievalStrategy):
    def is_active(self, agent: Any) -> bool:
        return bool(getattr(agent, "enable_web_search", False))

    async def execute(self, context: StrategyContext) -> StrategyResult:
        result = StrategyResult()
        result.events.append(ev.web_search_start().to_dict())

        try:
            search_results = await search_web(context.user_message)
            raw_results = search_results.get("results", [])

            if raw_results:
                context.memory.add_web_context(
                    get_web_search_client().format_search_results(search_results)
                )
                result.web_search_results = raw_results
                for item in raw_results:
                    result.sources.append({
                        "content": item.get("content", ""),
                        "score": 1.0,
                        "source_file": item.get("title", "网络搜索结果"),
                        "url": item.get("url", ""),
                        "type": "web_search",
                    })

            result.events.append(ev.web_search_complete(raw_results).to_dict())
        except Exception:
            logger.exception("联网搜索失败")
            result.events.append(ev.web_search_error().to_dict())

        return result

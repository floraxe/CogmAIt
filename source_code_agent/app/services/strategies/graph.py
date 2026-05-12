"""
图谱检索策略。

图谱检索的实现逻辑较重（Cypher 生成、Neo4j 连接管理等），
委托给专职的 GraphRetrievalService，策略层只负责激活判断和结果适配。
"""
from typing import Any

from app.services.strategy_base import BaseRetrievalStrategy, StrategyContext, StrategyResult


class GraphRetrievalStrategy(BaseRetrievalStrategy):
    def __init__(self, graph_service: Any) -> None:
        self._graph_service = graph_service

    def is_active(self, agent: Any) -> bool:
        return bool(getattr(agent, "graphs", None))

    async def execute(self, context: StrategyContext) -> StrategyResult:
        events = await self._graph_service.run(
            db=context.db,
            agent=context.agent,
            user_message=context.user_message,
            model_id=context.model_id,
        )
        return StrategyResult(events=events)

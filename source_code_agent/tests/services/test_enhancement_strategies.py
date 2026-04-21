from types import SimpleNamespace
import asyncio

from app.services.strategy_base import StrategyContext
from app.services.chat_orchestration_service import (
    StrategyResult,
    RetrievalAugmentationResult,
    WebSearchStrategy,
    KnowledgeRetrievalStrategy,
    GraphRetrievalStrategy,
)


class _FakeRetrievalService:
    async def run_web_search_only(self, memory, agent, user_message):
        return RetrievalAugmentationResult(
            events=[{"event": "web_search_complete", "data": {"ok": True}, "sleep": 0.1}],
            sources=[{"type": "web_search", "source_file": "doc"}],
            web_search_results=[{"title": "result"}],
        )

    async def run_knowledge_retrieval_only(self, memory, db, agent, user_message, config):
        return RetrievalAugmentationResult(
            events=[{"event": "vector_search_complete", "data": {"count": 1}, "sleep": 0.1}],
            sources=[{"type": "document", "source_file": "kb"}],
            web_search_results=[],
        )


class _FakeGraphService:
    async def run(self, db, agent, user_message, model_id):
        return [{"event": "graph_search_complete", "data": '{"ok": true}', "sleep": 0.1}]


def _build_context():
    return StrategyContext(
        memory=SimpleNamespace(),
        db=SimpleNamespace(),
        agent=SimpleNamespace(),
        user_message="hello",
        model_id="m1",
        config={"top_k": 3},
    )


def test_web_search_strategy_execute_returns_unified_result():
    strategy = WebSearchStrategy(_FakeRetrievalService())
    result = asyncio.run(strategy.execute(_build_context()))
    assert isinstance(result, StrategyResult)
    assert result.events and result.events[0]["event"] == "web_search_complete"
    assert result.sources and result.sources[0]["type"] == "web_search"
    assert result.web_search_results and result.web_search_results[0]["title"] == "result"


def test_knowledge_strategy_execute_returns_unified_result():
    strategy = KnowledgeRetrievalStrategy(_FakeRetrievalService())
    result = asyncio.run(strategy.execute(_build_context()))
    assert isinstance(result, StrategyResult)
    assert result.events and result.events[0]["event"] == "vector_search_complete"
    assert result.sources and result.sources[0]["type"] == "document"
    assert result.web_search_results == []


def test_graph_strategy_execute_returns_unified_result():
    strategy = GraphRetrievalStrategy(_FakeGraphService())
    result = asyncio.run(strategy.execute(_build_context()))
    assert isinstance(result, StrategyResult)
    assert result.events and result.events[0]["event"] == "graph_search_complete"
    assert result.sources == []
    assert result.web_search_results == []

"""
增强策略单元测试。

策略重构后各自拥有完整业务逻辑（不再委托给 RetrievalAugmentationService）。
此处通过 monkeypatch 替换策略所调用的底层工具函数，避免真实外部依赖。
"""
import asyncio
from types import SimpleNamespace

import pytest

from app.services.strategy_base import StrategyContext, StrategyResult
from app.services.strategies import (
    WebSearchStrategy,
    KnowledgeRetrievalStrategy,
    GraphRetrievalStrategy,
)


def _build_context():
    return StrategyContext(
        memory=SimpleNamespace(
            add_web_context=lambda ctx: None,
            add_knowledge_context=lambda ctx: None,
        ),
        db=SimpleNamespace(),
        agent=SimpleNamespace(
            enable_web_search=True,
            knowledge_bases=[SimpleNamespace(id="kb-1")],
            graphs=[SimpleNamespace(id="g-1")],
        ),
        user_message="hello",
        model_id="m1",
        config={"top_k": 3, "similarity_threshold": 0.5},
    )


# ── WebSearchStrategy ─────────────────────────────────────────────────────────

def test_web_search_strategy_is_active():
    agent_on = SimpleNamespace(enable_web_search=True)
    agent_off = SimpleNamespace(enable_web_search=False)
    s = WebSearchStrategy()
    assert s.is_active(agent_on) is True
    assert s.is_active(agent_off) is False


def test_web_search_strategy_execute_returns_unified_result(monkeypatch):
    async def _fake_search(query):
        return {"results": [{"title": "T", "url": "http://x", "content": "C"}]}

    monkeypatch.setattr("app.services.strategies.web_search.search_web", _fake_search)
    monkeypatch.setattr(
        "app.services.strategies.web_search.get_web_search_client",
        lambda: SimpleNamespace(format_search_results=lambda r: "ctx"),
    )

    result = asyncio.run(WebSearchStrategy().execute(_build_context()))

    assert isinstance(result, StrategyResult)
    assert any(e["event"] == "web_search_complete" for e in result.events)
    assert result.sources and result.sources[0]["type"] == "web_search"
    assert result.web_search_results and result.web_search_results[0]["title"] == "T"


def test_web_search_strategy_handles_empty_results(monkeypatch):
    async def _fake_search(query):
        return {"results": []}

    monkeypatch.setattr("app.services.strategies.web_search.search_web", _fake_search)
    monkeypatch.setattr(
        "app.services.strategies.web_search.get_web_search_client",
        lambda: SimpleNamespace(format_search_results=lambda r: ""),
    )

    result = asyncio.run(WebSearchStrategy().execute(_build_context()))
    assert any(e["event"] == "web_search_complete" for e in result.events)
    assert result.sources == []


# ── KnowledgeRetrievalStrategy ────────────────────────────────────────────────

def test_knowledge_strategy_is_active():
    s = KnowledgeRetrievalStrategy()
    assert s.is_active(SimpleNamespace(knowledge_bases=[SimpleNamespace(id="kb")])) is True
    assert s.is_active(SimpleNamespace(knowledge_bases=[])) is False


async def _fake_execute_model_inference(db, model, payload):
    """与真实 `execute_model_inference` 同为 async，避免在 asyncio.run 内嵌套 run_until_complete。"""
    return {"embeddings": [[0.1, 0.2]]}


def test_knowledge_strategy_execute_returns_unified_result(monkeypatch):
    monkeypatch.setattr(
        "app.services.strategies.knowledge_retrieval.get_knowledge",
        lambda db, kb_id: SimpleNamespace(embedding_model="emb-1", name="TestKB"),
    )
    monkeypatch.setattr(
        "app.services.strategies.knowledge_retrieval.execute_model_inference",
        _fake_execute_model_inference,
    )

    fake_store = SimpleNamespace(
        client=object(),
        search_similar=lambda **kw: [{"score": 0.9, "text": "content", "file_id": "f1", "chunk_index": 0}],
    )
    monkeypatch.setattr(
        "app.services.strategies.knowledge_retrieval.EmbeddingManager",
        SimpleNamespace(get_vector_store=lambda: fake_store),
    )
    monkeypatch.setattr(
        "app.services.strategies.knowledge_retrieval.get_knowledge_file",
        lambda db, fid: SimpleNamespace(original_filename="doc.txt"),
    )

    result = asyncio.run(KnowledgeRetrievalStrategy().execute(_build_context()))

    assert isinstance(result, StrategyResult)
    assert any(e["event"] == "vector_search_complete" for e in result.events)
    assert result.sources and result.sources[0]["type"] == "document"


# ── GraphRetrievalStrategy ────────────────────────────────────────────────────

def test_graph_strategy_is_active():
    s = GraphRetrievalStrategy(graph_service=None)
    assert s.is_active(SimpleNamespace(graphs=[SimpleNamespace(id="g")])) is True
    assert s.is_active(SimpleNamespace(graphs=[])) is False


def test_graph_strategy_execute_delegates_to_service(monkeypatch):
    import json

    class _FakeGraphService:
        async def run(self, db, agent, user_message, model_id):
            return [{"event": "graph_search_complete", "data": json.dumps({"ok": True})}]

    result = asyncio.run(GraphRetrievalStrategy(_FakeGraphService()).execute(_build_context()))
    assert isinstance(result, StrategyResult)
    assert result.events and result.events[0]["event"] == "graph_search_complete"
    assert result.sources == []
    assert result.web_search_results == []

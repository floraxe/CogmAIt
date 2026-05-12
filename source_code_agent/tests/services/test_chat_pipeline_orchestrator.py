"""
流水线编排器集成测试。

使用 fake 对象替换所有外部依赖（DB、模型、MCP 等），
验证四阶段 Pipeline 的事件序列与历史持久化行为。
"""
import asyncio
import json
from types import SimpleNamespace

from app.services.chat_pipeline import ChatPipelineOrchestrator, ChatPipelineRequest
from app.services.document_context_service import DocumentContextResult
from app.services.mcp_service import McpOrchestrationResult
from app.services.strategy_base import StrategyResult
from app.schemas.agent import AgentChatMessage


class _FakeDocumentService:
    async def process_files(self, db, file_ids):
        return DocumentContextResult(
            formatted_contexts=["--- 文件: demo.txt ---\n\nhello\n\n"],
            processed_messages=["已处理文件: demo.txt"],
            error_messages=[],
        )

    def build_system_context(self, formatted_contexts):
        if not formatted_contexts:
            return None
        return "以下是用户上传的文件内容\n" + "\n".join(formatted_contexts)


class _FakeRetrievalStrategy:
    def is_active(self, agent):
        return True

    async def execute(self, context):
        return StrategyResult(
            events=[{"event": "web_search_complete", "data": json.dumps({"ok": True})}],
            sources=[{"type": "web_search", "source_file": "demo"}],
            web_search_results=[{"title": "demo"}],
        )


class _FakeMcpService:
    async def run(self, db, agent, user_message, model_id, current_user_id):
        return McpOrchestrationResult(
            events=[{"event": "mcp_result", "data": json.dumps({"ok": True})}],
            tool_result_prompt="tool result",
        )


class _FakeResponseService:
    async def ensure_file_guidance(self, memory, final_messages, file_ids, db, document_service, user_message):
        return final_messages, True

    def build_extra_data(self, response_time, used_tokens, sources, web_search_results, has_file_content):
        return {
            "response_time_ms": response_time,
            "tokens_used": used_tokens,
            "sources_count": len(sources),
            "web_count": len(web_search_results),
            "has_file_content": has_file_content,
        }


class _FakeInferenceService:
    def build_stream_payload(self, messages, config):
        return {"messages": messages, "stream": True, **config}

    async def run_stream(self, db, model_id, payload):
        async def _gen():
            yield {"choices": [{"delta": {"content": "你好"}}]}
            yield {"choices": [{"delta": {"content": "世界"}}]}

        return _gen()

    def normalize_stream_chunk(self, chunk):
        content = chunk["choices"][0]["delta"].get("content", "")
        return "message_chunk", chunk, content


def _build_request():
    return ChatPipelineRequest(
        db=SimpleNamespace(),
        agent_id="agent-1",
        messages=[AgentChatMessage(role="user", content="你好")],
        session_id="session-1",
        config_override={"temperature": 0.1},
        file_ids=["file-1"],
        current_user_id="user-1",
        access_type="user",
    )


def _build_orchestrator(fake_agent, monkeypatch, *, capture: dict = None):
    """
    工厂函数：通过构造函数注入所有 fake 服务，展示 DI 的正确用法。
    monkeypatch 只用于替换 chat_pipeline 内 agent_utils 别名指向模块上的 DB/history 调用，
    不再通过属性赋值覆盖已构造好的对象。
    """
    monkeypatch.setattr("app.services.chat_pipeline.agent_utils.get_agent", lambda db, aid: fake_agent)
    monkeypatch.setattr("app.services.chat_pipeline.agent_utils.get_model", lambda db, mid: {"id": mid})
    if capture is not None:
        monkeypatch.setattr(
            "app.services.chat_pipeline.agent_utils.create_chat_history",
            lambda **kw: capture.update(kw),
        )

    return ChatPipelineOrchestrator(
        document_service=_FakeDocumentService(),
        mcp_service=_FakeMcpService(),
        response_service=_FakeResponseService(),
        inference_service=_FakeInferenceService(),
        strategy_registry=[_FakeRetrievalStrategy()],
    )


def test_chat_pipeline_stream_contains_four_stage_key_events(monkeypatch):
    fake_agent = SimpleNamespace(
        model_id="model-1",
        config={"top_p": 0.9},
        system_prompt="system",
        enable_web_search=True,
        knowledge_bases=[],
        graphs=[],
    )
    orchestrator = _build_orchestrator(fake_agent, monkeypatch)

    events = asyncio.run(_collect_events(orchestrator.stream(_build_request())))
    event_names = [item["event"] for item in events]

    assert event_names[0] == "status"
    assert "file_processing" in event_names
    assert "think" in event_names
    assert "web_search_complete" in event_names
    assert "reasoning" in event_names
    assert "answer" in event_names
    assert event_names[-1] == "done"


def test_chat_pipeline_filter_persists_chat_history(monkeypatch):
    fake_agent = SimpleNamespace(
        model_id="model-2",
        config={},
        system_prompt="system",
        enable_web_search=True,
        knowledge_bases=[],
        graphs=[],
    )
    captured: dict = {}
    orchestrator = _build_orchestrator(fake_agent, monkeypatch, capture=captured)

    asyncio.run(_collect_events(orchestrator.stream(_build_request())))

    assert captured["agent_id"] == "agent-1"
    assert captured["session_id"] == "session-1"
    assert captured["user_id"] == "user-1"
    assert captured["access_type"] == "user"
    assert captured["model_id"] == "model-2"
    assert "你好世界" in captured["agent_response"]
    assert captured["extra_data"]["has_file_content"] is True


async def _collect_events(generator):
    items = []
    async for event in generator:
        if isinstance(event.get("data"), str):
            try:
                json.loads(event["data"])
            except Exception:
                pass
        items.append(event)
    return items

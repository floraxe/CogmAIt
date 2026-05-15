import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.inference.facade import ModelInferenceFacade
from app.services.inference.stream_adapter import StreamChunkNormalizer


@pytest.mark.asyncio
async def test_facade_delegates_to_chat_strategy():
    facade = ModelInferenceFacade()
    model = SimpleNamespace(
        id="m1",
        status="active",
        provider="openai",
        type="chat",
        name="gpt-test",
        api_key="k",
        base_url="http://localhost",
    )
    mock_provider = MagicMock()
    mock_provider.chat_completion = AsyncMock(return_value={"choices": []})

    with patch("app.services.inference.facade.get_model", return_value=model), patch(
        "app.services.inference.facade.ProviderFactory.get_for_model",
        return_value=mock_provider,
    ):
        result = await facade.run(
            MagicMock(),
            "m1",
            {"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )

    assert result == {"choices": []}
    mock_provider.chat_completion.assert_awaited_once()


def test_stream_chunk_normalizer_openai_delta():
    normalizer = StreamChunkNormalizer()
    event, data, text = normalizer.normalize(
        {"choices": [{"delta": {"content": "你好"}, "finish_reason": None}]}
    )
    assert event == "message_chunk"
    assert text == "你好"


def test_stream_chunk_normalizer_tool_calls():
    normalizer = StreamChunkNormalizer()
    event, _, text = normalizer.normalize(
        {"choices": [{"delta": {"tool_calls": [{"id": "1"}]}, "finish_reason": None}]}
    )
    assert event == "tool_calls"
    assert text == ""

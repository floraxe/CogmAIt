"""
本地 OpenAI 兼容假服务：用于在无真实厂商 Key 时验证 CogmAIt 的 Model/Agent/对话链路。

提供：
  GET  /v1/models
  POST /v1/chat/completions  （支持 stream=true 的 SSE）

运行（在 source_code_agent 目录下）:
  poetry run python scripts/openai_compat_mock.py

默认监听 0.0.0.0:18080（与平台模型 base_url 推荐值 http://127.0.0.1:18080/v1 对齐）。
默认模型 id 为 DeepSeek-Test-Model（与平台里模型的 name 字段默认对齐）；可用环境变量 MOCK_OPENAI_MODEL 覆盖。
"""
from __future__ import annotations

import json
import os
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="OpenAI-compat mock", version="0.1.0")

MOCK_MODEL = os.environ.get("MOCK_OPENAI_MODEL", "DeepSeek-Test-Model")


@app.get("/v1/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": MOCK_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "mock",
            }
        ],
    }


def _last_user_text(messages: list) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content", "")
            return c if isinstance(c, str) else str(c)
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = bool(body.get("stream"))
    messages = body.get("messages") or []
    user_text = _last_user_text(messages)
    reply = (
        f"[本地假服务] 已收到你的消息：{user_text[:500]}"
        if user_text
        else "[本地假服务] 你好，这是用于联调的固定回复。"
    )

    model_name = body.get("model") or MOCK_MODEL

    if not stream:
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": max(1, len(reply) // 4),
                "total_tokens": 2,
            },
        }

    async def event_stream() -> AsyncIterator[str]:
        # 按字符切分，避免截断中文
        step = 12
        for i in range(0, len(reply), step):
            part = reply[i : i + step]
            chunk = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": part},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        final = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("MOCK_OPENAI_HOST", "0.0.0.0")
    port = int(os.environ.get("MOCK_OPENAI_PORT", "18080"))
    print(
        f"[openai_compat_mock] 即将启动: http://127.0.0.1:{port}/v1 "
        f"(本机也可用 0.0.0.0:{port})  默认模型 id={MOCK_MODEL!r}"
    )
    uvicorn.run(app, host=host, port=port)

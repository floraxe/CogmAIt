import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.core.constants import AGENT_TYPES
from app.db.session import get_db
from app.models.user import User
from app.schemas.agent import (
    AgentChatMessage,
    AgentChatRequest,
    AgentCreate,
    AgentListResponse,
    AgentResponse,
    AgentUpdate,
    ShareChatRequestBody,
)
from app.services import agent_service
from app.services.chat_application_service import get_chat_application_service
from app.utils import format_datetime
from app.utils.deps import get_current_active_user

logger = logging.getLogger(__name__)

_ERR_SHARE_TOKEN = "生成分享令牌失败，请稍后重试"
_ERR_API_KEY = "生成 API 密钥失败，请稍后重试"

# OpenAPI：流式对话实际返回 text/event-stream（SSE），非 JSON 包装的 AgentChatResponse。
_AGENT_CHAT_STREAM_OPENAPI = {
    200: {
        "description": "对话事件流（SSE，`text/event-stream`，每条事件的 data 为 JSON）",
        "content": {
            "text/event-stream": {
                "schema": {
                    "type": "string",
                    "example": 'event: answer_delta\ndata: {"text":"..."}\n\n',
                }
            }
        },
    },
    400: {"description": "当前仅支持流式请求（stream=true）"},
}

router = APIRouter()


def _agent_share_public_card(agent) -> Dict[str, Any]:
    """分享场景下对外暴露的智能体卡片字段（与 share_id/agent_id 校验无关的纯展示）。"""
    return {
        "agent_id": agent.id,
        "name": agent.name,
        "description": agent.description,
        "avatar": agent.avatar,
        "welcome_message": agent.welcome_message,
        "type": agent.type,
    }


def _agent_or_404(db: Session, agent_id: str):
    agent = agent_service.get_agent(db=db, agent_id=agent_id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="智能体不存在")
    return agent


def _build_streaming_response(chat_request: AgentChatRequest, generator) -> EventSourceResponse:
    if not chat_request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前仅支持流式请求",
        )
    return EventSourceResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/", response_model=AgentListResponse)
async def get_agents(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    name: Optional[str] = None,
    type: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
):
    """获取智能体列表。"""
    skip = (page - 1) * limit
    agents = agent_service.get_agents(
        db,
        skip=skip,
        limit=limit,
        name=name,
        type=type,
        status=status,
    )
    agent_dicts = [agent.to_dict() for agent in agents]
    total = agent_service.count_agents(db, name=name, type=type, status=status)
    return {"total": total, "items": agent_dicts}


@router.get("/types")
async def get_agent_types(
    current_user: User = Depends(get_current_active_user),
):
    """获取智能体类型列表。"""
    return AGENT_TYPES


@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent_in: AgentCreate = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """创建智能体。"""
    agent = agent_service.create_agent(db=db, agent_in=agent_in, owner=current_user)
    return agent.to_dict()


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent_detail(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """获取智能体详情。"""
    agent = _agent_or_404(db, agent_id)
    return agent.to_dict()


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    agent_in: AgentUpdate = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """更新智能体。"""
    agent = _agent_or_404(db, agent_id)
    updated_agent = agent_service.update_agent(db=db, agent=agent, agent_in=agent_in)
    return updated_agent.to_dict()


@router.post("/{agent_id}/avatar", response_model=AgentResponse)
async def update_agent_avatar(
    agent_id: str,
    avatar_data: Dict[str, str] = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """更新智能体头像（base64）。"""
    agent = _agent_or_404(db, agent_id)
    if "avatar" not in avatar_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请求缺少avatar字段",
        )
    avatar_update = AgentUpdate(avatar=avatar_data["avatar"])
    updated_agent = agent_service.update_agent(db=db, agent=agent, agent_in=avatar_update)
    return updated_agent.to_dict()


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """删除智能体。"""
    _agent_or_404(db, agent_id)
    agent_service.delete_agent(db=db, agent_id=agent_id)


@router.post(
    "/{agent_id}/chat",
    response_class=EventSourceResponse,
    responses=_AGENT_CHAT_STREAM_OPENAPI,
)
async def chat_with_agent(
    agent_id: str,
    chat_request: AgentChatRequest = Body(...),
    db: Session = Depends(get_db),
    token: Optional[str] = Query(None, description="分享令牌，用于免登录窗口访问"),
    user_id: Optional[str] = Query("000000", description="用户ID，默认为000000表示游客"),
):
    """与智能体对话（支持分享令牌免登录）。"""
    logger.debug(
        "chat_with_agent: agent_id=%s user_id=%s file_ids=%s",
        agent_id,
        user_id,
        chat_request.file_ids,
    )
    return _build_streaming_response(
        chat_request,
        get_chat_application_service().stream_chat(
            db,
            agent_id,
            chat_request,
            share_token=token,
            current_user_id=user_id or "000000",
        ),
    )


@router.post("/{agent_id}/generate-share-token", response_model=Dict[str, str])
async def generate_agent_share_token(
    agent_id: str,
    name: Optional[str] = Query(None, description="Token名称/描述"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """为智能体生成分享令牌。"""
    _agent_or_404(db, agent_id)
    try:
        token_id, token = agent_service.generate_share_token(db, agent_id, name)
        return {"id": token_id, "share_token": token}
    except ValueError as exc:
        logger.warning("generate_share_token: %s", exc)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception:
        logger.exception("generate_share_token failed agent_id=%s", agent_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_ERR_SHARE_TOKEN,
        ) from None


@router.post("/{agent_id}/generate-api-key", response_model=Dict[str, str])
async def generate_agent_api_key(
    agent_id: str,
    name: Optional[str] = Query(None, description="密钥名称/描述"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """为智能体生成 API 密钥。"""
    _agent_or_404(db, agent_id)
    try:
        key_id, api_key = agent_service.generate_api_key(db, agent_id, name)
        return {"id": key_id, "api_key": api_key}
    except ValueError as exc:
        logger.warning("generate_api_key: %s", exc)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception:
        logger.exception("generate_api_key failed agent_id=%s", agent_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_ERR_API_KEY,
        ) from None


@router.get("/{agent_id}/api-keys", response_model=Dict[str, Any])
async def get_agent_api_keys(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """获取智能体的所有 API 密钥。"""
    agent = _agent_or_404(db, agent_id)
    api_keys = agent_service.get_agent_api_keys(db, agent_id)
    return {
        "enabled": agent.api_enabled,
        "items": [api_key.to_dict() for api_key in api_keys],
    }


@router.get("/{agent_id}/share-tokens", response_model=Dict[str, Any])
async def get_agent_share_tokens(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """获取智能体的所有分享 Token。"""
    agent = _agent_or_404(db, agent_id)
    share_tokens = agent_service.get_agent_share_tokens(db, agent_id)
    return {
        "enabled": agent.share_enabled,
        "items": [share_token.to_dict() for share_token in share_tokens],
    }


@router.delete("/{agent_id}/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_api_key(
    agent_id: str,
    key_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """删除智能体的 API 密钥。"""
    _agent_or_404(db, agent_id)
    if not agent_service.delete_api_key(db, agent_id, key_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API密钥不存在")


@router.delete("/{agent_id}/share-tokens/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_share_token(
    agent_id: str,
    token_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """删除智能体的分享 Token。"""
    _agent_or_404(db, agent_id)
    if not agent_service.delete_share_token(db, agent_id, token_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="分享Token不存在")


@router.post("/{agent_id}/toggle-share", response_model=Dict[str, bool])
async def toggle_agent_share(
    agent_id: str,
    enabled: bool = Query(..., description="是否启用分享"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """切换智能体分享状态。"""
    if not agent_service.toggle_share_status(db, agent_id, enabled):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="智能体不存在")
    return {"success": True, "share_enabled": enabled}


@router.post("/{agent_id}/toggle-api", response_model=Dict[str, bool])
async def toggle_agent_api(
    agent_id: str,
    enabled: bool = Query(..., description="是否启用API访问"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """切换智能体 API 访问状态。"""
    if not agent_service.toggle_api_status(db, agent_id, enabled):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="智能体不存在")
    return {"success": True, "api_enabled": enabled}


@router.get("/share/{token}", response_model=Dict[str, Any])
async def get_agent_by_share_token(
    token: str,
    db: Session = Depends(get_db),
):
    """通过分享令牌获取智能体公开信息。"""
    agent = agent_service.get_agent_by_share_token(db, token)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在或分享链接无效",
        )
    return _agent_share_public_card(agent)


@router.post(
    "/chat-with-api-key",
    response_class=EventSourceResponse,
    responses=_AGENT_CHAT_STREAM_OPENAPI,
)
async def chat_with_agent_api(
    chat_request: AgentChatRequest = Body(...),
    api_key: str = Header(..., description="智能体API密钥"),
    db: Session = Depends(get_db),
    user_id: Optional[str] = Query("000000", description="用户ID，默认为000000表示游客"),
):
    """使用 API Key 与智能体对话。"""
    result = agent_service.get_agent_by_api_key(db, api_key)
    if not result:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API密钥无效或已禁用")

    agent, api_key_id = result
    if not agent.api_enabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="该智能体未启用API访问")

    return _build_streaming_response(
        chat_request,
        get_chat_application_service().stream_chat_for_api_key(
            db,
            agent.id,
            chat_request,
            api_key_id=api_key_id,
            current_user_id=user_id or "000000",
        ),
    )


@router.get("/{agent_id}/logs", response_model=Dict[str, Any])
async def get_agent_chat_logs(
    agent_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """获取智能体会话列表（分页与按会话/用户筛选）。"""
    _agent_or_404(db, agent_id)
    raw_sessions, total = agent_service.get_chat_sessions(
        db,
        agent_id,
        session_id=session_id,
        user_id=user_id,
        skip=skip,
        limit=limit,
    )
    sessions = [
        {
            **s,
            "lastMessage": format_datetime(s["lastMessage"]),
            "firstMessage": format_datetime(s["firstMessage"]),
        }
        for s in raw_sessions
    ]
    return {"total": total, "items": sessions}


@router.get("/{agent_id}/share-chat/{share_id}", response_model=Dict[str, Any])
async def get_agent_share_info(
    agent_id: str,
    share_id: str,
    db: Session = Depends(get_db),
):
    """通过分享 ID 获取智能体信息（含完整 `agent_info`，供分享页初始化）。"""
    agent = agent_service.get_agent_by_share_token(db, share_id)
    if not agent or agent.id != agent_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在或分享链接无效",
        )
    return {**_agent_share_public_card(agent), "agent_info": agent.to_dict()}


@router.post(
    "/{agent_id}/share-chat/{share_id}/chat",
    response_class=EventSourceResponse,
    responses=_AGENT_CHAT_STREAM_OPENAPI,
)
async def share_chat_with_agent(
    agent_id: str,
    share_id: str,
    body: ShareChatRequestBody = Body(...),
    db: Session = Depends(get_db),
):
    """通过分享链接与智能体对话。"""
    agent = agent_service.get_agent_by_share_token(db, share_id)
    if not agent or agent.id != agent_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在或分享链接无效",
        )

    session_id = body.session_id or f"share_{share_id}_{int(time.time())}"
    logger.debug("share_chat: agent_id=%s message_len=%d", agent_id, len(body.message))

    chat_request = AgentChatRequest(
        messages=[AgentChatMessage(role="user", content=body.message)] if body.message else [],
        stream=True,
        session_id=session_id,
        config={},
        type="web",
        file_ids=body.file_ids or None,
    )

    return await chat_with_agent(
        agent_id=agent_id,
        chat_request=chat_request,
        db=db,
        token=share_id,
    )

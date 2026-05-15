import time
from typing import List, Optional

from sqlalchemy.orm import Session

from app.schemas.agent import AgentChatRequest
from app.services.chat_pipeline import ChatPipelineRequest


class ChatPipelineRequestBuilder:
    """将 HTTP 层对话参数组装为流水线请求（工厂方法）。"""

    @staticmethod
    def build(
        *,
        db: Session,
        agent_id: str,
        messages: List,
        session_id: Optional[str],
        config_override: Optional[dict],
        file_ids: Optional[List[str]],
        current_user_id: str,
        access_type: str,
        share_token: Optional[str] = None,
        api_key_id: Optional[str] = None,
    ) -> ChatPipelineRequest:
        return ChatPipelineRequest(
            db=db,
            agent_id=agent_id,
            messages=messages,
            session_id=session_id or f"session_{int(time.time())}",
            config_override=config_override or {},
            file_ids=file_ids or [],
            current_user_id=current_user_id or "000000",
            access_type=access_type,
            share_token=share_token,
            api_key_id=api_key_id,
        )

    @classmethod
    def from_chat_request(
        cls,
        *,
        db: Session,
        agent_id: str,
        chat_request: AgentChatRequest,
        access_type: str,
        current_user_id: str = "000000",
        share_token: Optional[str] = None,
        api_key_id: Optional[str] = None,
    ) -> ChatPipelineRequest:
        return cls.build(
            db=db,
            agent_id=agent_id,
            messages=chat_request.messages,
            session_id=chat_request.session_id,
            config_override=chat_request.config,
            file_ids=chat_request.file_ids,
            current_user_id=current_user_id,
            access_type=access_type,
            share_token=share_token,
            api_key_id=api_key_id,
        )

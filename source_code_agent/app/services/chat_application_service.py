from typing import AsyncGenerator, Dict, Optional

from sqlalchemy.orm import Session

from app.schemas.agent import AgentChatRequest
from app.services.agent_access_service import resolve_agent_access
from app.services.chat_pipeline import ChatPipelineOrchestrator, ChatPipelineRequest
from app.services.chat_pipeline_request_builder import ChatPipelineRequestBuilder


class ChatApplicationService:
    """
    智能体对话应用服务（外观模式）。
    API 层仅依赖本服务，不直接组装 ChatPipelineRequest / Orchestrator。
    """

    def __init__(self, orchestrator: Optional[ChatPipelineOrchestrator] = None) -> None:
        self._orchestrator = orchestrator or ChatPipelineOrchestrator()

    def stream_chat(
        self,
        db: Session,
        agent_id: str,
        chat_request: AgentChatRequest,
        *,
        share_token: Optional[str] = None,
        current_user_id: str = "000000",
    ) -> AsyncGenerator[Dict, None]:
        _, resolved_agent_id, access_type = resolve_agent_access(db, agent_id, share_token)
        pipeline_request = ChatPipelineRequestBuilder.from_chat_request(
            db=db,
            agent_id=resolved_agent_id,
            chat_request=chat_request,
            access_type=access_type,
            current_user_id=current_user_id,
            share_token=share_token if access_type == "share" else None,
        )
        return self._orchestrator.stream(pipeline_request)

    def stream_chat_with_pipeline_request(
        self, pipeline_request: ChatPipelineRequest
    ) -> AsyncGenerator[Dict, None]:
        return self._orchestrator.stream(pipeline_request)

    def stream_chat_for_api_key(
        self,
        db: Session,
        agent_id: str,
        chat_request: AgentChatRequest,
        *,
        api_key_id: str,
        current_user_id: str = "000000",
    ) -> AsyncGenerator[Dict, None]:
        pipeline_request = ChatPipelineRequestBuilder.from_chat_request(
            db=db,
            agent_id=agent_id,
            chat_request=chat_request,
            access_type="api",
            current_user_id=current_user_id,
            api_key_id=api_key_id,
        )
        return self._orchestrator.stream(pipeline_request)


_default_chat_application_service = ChatApplicationService()


def get_chat_application_service() -> ChatApplicationService:
    return _default_chat_application_service

"""
智能体访问解析服务。

将"根据 token/agent_id 确定访问目标"的逻辑从 API 路由层抽离，
路由层只做"调用 + HTTP 异常映射"，不包含业务判断。
"""
from typing import Any, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.services import agent_service


def resolve_agent_access(
    db: Session,
    agent_id: str,
    token: Optional[str],
) -> Tuple[Any, str, str]:
    """
    解析聊天请求的访问目标与访问类型。

    Returns:
        (agent, resolved_agent_id, access_type)
        access_type: "share" | "user"
    Raises:
        HTTPException 401 — token 无效
        HTTPException 404 — agent 不存在
    """
    if token:
        agent = agent_service.get_agent_by_share_token(db, token)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="分享令牌无效或已禁用",
            )
        if agent.id != agent_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="智能体不存在或分享链接无效",
            )
        return agent, agent.id, "share"

    agent = agent_service.get_agent(db=db, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="智能体不存在",
        )
    return agent, agent_id, "user"

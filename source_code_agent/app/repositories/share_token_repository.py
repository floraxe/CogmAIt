from typing import Optional

from sqlalchemy.orm import Session

from app.models.agent import AgentShareToken


class ShareTokenRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def find_id_by_token(self, token: str) -> Optional[str]:
        row = (
            self._db.query(AgentShareToken)
            .filter(AgentShareToken.token == token)
            .first()
        )
        return row.id if row else None

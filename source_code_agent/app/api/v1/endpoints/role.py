from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.utils.deps import get_current_active_user
from app.utils.user import get_roles
from app.schemas.user import RoleResponse

router = APIRouter()

@router.get("/", response_model=List[RoleResponse])
async def read_roles(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取所有角色列表
    """
    roles = get_roles(db)
    return [role.to_dict() for role in roles] 
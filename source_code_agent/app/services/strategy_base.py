from dataclasses import dataclass, field
from typing import Any, Dict, List
from abc import ABC, abstractmethod

from sqlalchemy.orm import Session


@dataclass
class StrategyContext:
    memory: Any
    db: Session
    agent: Any
    user_message: str
    model_id: str
    config: Dict[str, Any]


@dataclass
class StrategyResult:
    events: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    web_search_results: List[Dict[str, Any]] = field(default_factory=list)


class BaseRetrievalStrategy(ABC):
    @abstractmethod
    def is_active(self, agent: Any) -> bool:
        """根据智能体配置决定该策略是否应当激活，满足开闭原则：新增策略只需实现此方法。"""
        raise NotImplementedError

    @abstractmethod
    async def execute(self, context: StrategyContext) -> StrategyResult:
        raise NotImplementedError

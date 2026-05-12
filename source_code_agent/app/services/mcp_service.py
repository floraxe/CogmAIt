import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.services import chat_events as ev

logger = logging.getLogger(__name__)


@dataclass
class McpOrchestrationResult:
    events: List[Dict[str, Any]] = field(default_factory=list)
    tool_result_prompt: Optional[str] = None


class McpOrchestrationService:
    async def run(
        self,
        db: Session,
        agent: Any,
        user_message: str,
        model_id: str,
        current_user_id: str,
    ) -> McpOrchestrationResult:
        result = McpOrchestrationResult()
        mcp_service_list = list(getattr(agent, "mcp_services", []) or [])
        if not mcp_service_list:
            return result

        result.events.append(ev.mcp_processing().to_dict())

        try:
            from app.utils.mcp import call_mcp_service, analyze_mcp_service_needs

            detection_result = await analyze_mcp_service_needs(
                db=db,
                model_id=model_id,
                user_message=user_message,
                available_services=mcp_service_list,
            )
            if not detection_result or not detection_result.get("call_mcp", False):
                reason = (detection_result or {}).get("reason", "无原因")
                if any(key in reason for key in ["无法解析", "解析错误", "解析失败", "解析JSON"]):
                    result.events.append(ev.mcp_error(f"解析MCP服务需求失败: {reason}").to_dict())
                return result

            service_id = detection_result.get("service_id")
            function_name = detection_result.get("function_name")
            params = detection_result.get("params", {})
            if not service_id or not function_name:
                result.events.append(ev.mcp_error("MCP服务ID或函数名称为空").to_dict())
                return result

            result.events.append(ev.mcp_call(service_id, function_name, params).to_dict())

            service = next((svc for svc in mcp_service_list if svc.id == service_id), None)
            if not service:
                result.events.append(ev.mcp_error(f"服务ID {service_id} 不在当前智能体关联的服务列表中").to_dict())
                return result

            mcp_call_result = await call_mcp_service(
                db=db,
                service_id=service_id,
                function_name=function_name,
                params=params,
                user_id=current_user_id,
            )
            service_name = service.name if hasattr(service, "name") else service.get("name", "Unknown")
            result.events.append(ev.mcp_result(service_name, function_name, mcp_call_result).to_dict())
            result.tool_result_prompt = (
                f"以下是调用MCP服务 '{service_name}' 的函数 '{function_name}' 的结果，"
                f"请使用这些结果回答用户的问题:\n\n"
                f"```json\n{json.dumps(mcp_call_result, ensure_ascii=False, indent=2)}\n```"
            )
        except Exception:
            logger.exception("MCP 服务执行异常")
            result.events.append(ev.mcp_error().to_dict())

        return result

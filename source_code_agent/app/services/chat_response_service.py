from typing import Any, Dict, List, Tuple

from sqlalchemy.orm import Session


class ChatResponseService:
    @staticmethod
    async def ensure_file_guidance(
        memory: Any,
        final_messages: List[Dict[str, Any]],
        file_ids: List[str],
        db: Session,
        document_service: Any,
        user_message: str,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        has_file_content = any(
            "以下是用户上传的文件内容" in msg.get("content", "")
            for msg in final_messages
            if msg.get("role") == "system"
        )
        if not has_file_content and file_ids:
            fallback_contexts = await document_service.load_plain_text_contexts(db, file_ids)
            fallback_system_context = document_service.build_system_context(fallback_contexts)
            if fallback_system_context:
                memory.prepend_context(fallback_system_context)
                has_file_content = True
                final_messages = memory.messages()

        if has_file_content and any(
            key in user_message
            for key in ["文档说的什么", "文档说了什么", "文件内容是什么", "文件说了什么", "文件说的什么"]
        ):
            memory.add_system_prompt(
                "用户正在询问文件内容。请直接回答文件的内容是什么，不要回避或者说找不到相关信息。"
                "文件内容已经在之前的系统消息中提供。"
            )
            final_messages = memory.messages()

        return final_messages, has_file_content

    @staticmethod
    def build_extra_data(
        response_time: int,
        used_tokens: int,
        sources: List[Dict[str, Any]],
        web_search_results: List[Dict[str, Any]],
        has_file_content: bool,
    ) -> Dict[str, Any]:
        extra_data: Dict[str, Any] = {
            "response_time_ms": response_time,
            "tokens_used": used_tokens,
        }
        if sources:
            extra_data["sources"] = sources
        if web_search_results:
            extra_data["web_results"] = web_search_results
        if has_file_content:
            extra_data["has_file_content"] = True
        return extra_data

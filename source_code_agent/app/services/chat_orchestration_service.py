from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os
import tempfile

from sqlalchemy.orm import Session

from app.models.file import File as FileModel
from app.utils.file_processor import extract_text_from_file_path
from app.core.minio_client import get_file_stream
from app.utils.model import execute_model_inference


@dataclass
class DocumentContextResult:
    formatted_contexts: List[str] = field(default_factory=list)
    processed_messages: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)


class DocumentContextService:
    def __init__(self, max_content_length: int = 3000):
        self.max_content_length = max_content_length

    async def process_files(self, db: Session, file_ids: List[str]) -> DocumentContextResult:
        result = DocumentContextResult()
        for file_id in file_ids:
            try:
                file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
                if not file_record:
                    result.error_messages.append(f"文件不存在: {file_id}")
                    continue

                if file_record.status != "processed":
                    if file_record.status == "processing":
                        result.error_messages.append(f"文件 {file_record.original_filename} 正在处理中，请稍后再试")
                    else:
                        result.error_messages.append(
                            f"文件 {file_record.original_filename} 未处理完成，状态: {file_record.status}"
                        )
                    continue

                file_content = await self._get_file_content(file_record)
                if not file_content:
                    result.error_messages.append(f"获取文件 {file_record.original_filename} 内容时出错")
                    continue

                if len(file_content) > self.max_content_length:
                    file_content = (
                        file_content[: self.max_content_length]
                        + f"\n\n[内容过长，已截断。原文共 {len(file_content)} 字符]"
                    )

                formatted_content = (
                    f"--- 文件: {file_record.original_filename} ({file_record.file_type}) ---\n\n"
                    f"{file_content}\n\n"
                )
                result.formatted_contexts.append(formatted_content)
                result.processed_messages.append(f"已处理文件: {file_record.original_filename}")
            except Exception as exc:
                result.error_messages.append(f"处理文件ID {file_id} 时出错: {str(exc)}")

        return result

    async def load_plain_text_contexts(self, db: Session, file_ids: List[str]) -> List[str]:
        contexts: List[str] = []
        for file_id in file_ids:
            file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
            if not file_record or not file_record.text_content:
                continue
            contexts.append(
                f"--- 文件: {file_record.original_filename} ({file_record.file_type}) ---\n\n"
                f"{file_record.text_content}\n\n"
            )
        return contexts

    @staticmethod
    def build_system_context(formatted_contexts: List[str]) -> Optional[str]:
        if not formatted_contexts:
            return None
        combined_content = "\n".join(formatted_contexts)
        return (
            "以下是用户上传的文件内容，这是非常重要的上下文信息，请务必仔细阅读并在回答问题时参考这些内容。"
            "如果用户询问关于文件内容的问题，请直接基于这些内容回答：\n\n"
            f"{combined_content}"
        )

    async def _get_file_content(self, file_record: FileModel) -> str:
        if file_record.text_content:
            return file_record.text_content

        if not file_record.path or "/" not in file_record.path:
            return ""

        bucket, object_name = file_record.path.split("/", 1)
        response = get_file_stream(bucket, object_name)
        if not response:
            return ""

        temp_file_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_record.file_type}") as temp_file:
                temp_file.write(response.read())
                temp_file_path = temp_file.name

            content = await extract_text_from_file_path(temp_file_path)
            if content and not content.startswith("提取文件文本内容时出错"):
                return content

            try:
                with open(temp_file_path, "r", encoding="utf-8") as file_obj:
                    return file_obj.read()
            except UnicodeDecodeError:
                with open(temp_file_path, "r", encoding="latin-1") as file_obj:
                    return file_obj.read()
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

        return ""


class ModelInferenceService:
    @staticmethod
    def build_stream_payload(messages: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "messages": messages,
            "stream": True,
            **config,
        }

    @staticmethod
    async def run_stream(db: Session, model_id: str, payload: Dict[str, Any]):
        return await execute_model_inference(db, model_id, payload)

    @staticmethod
    def normalize_stream_chunk(chunk: Any) -> Tuple[str, Any, str]:
        if isinstance(chunk, dict) and chunk.get("choices"):
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            if delta.get("tool_calls"):
                return "tool_calls", chunk, ""
            if choice.get("finish_reason") == "tool_calls":
                return "tool_call_result", chunk, ""
            content = delta.get("content", "") or ""
            return "message_chunk", chunk, content

        content = ""
        if isinstance(chunk, str):
            try:
                import json

                content = json.loads(chunk)["choices"][0]["delta"].get("content", "")
            except Exception:
                content = ""
        return "message_chunk", chunk, content

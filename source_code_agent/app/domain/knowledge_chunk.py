from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    """
    不可变知识切块对象。

    AF:
        表示知识处理流水线中的一个“文本块 + 元数据”最小单元。

    RI:
        1) text 必须是非空字符串。
        2) metadata 在构造后不可被外部修改（冻结为 tuple）。
        3) chunk_index >= 0, total_chunks >= 1, 且 chunk_index < total_chunks。
    """

    text: str
    chunk_index: int
    total_chunks: int
    _metadata_items: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("KnowledgeChunk.text 必须是非空字符串")
        if self.chunk_index < 0:
            raise ValueError("KnowledgeChunk.chunk_index 不能小于 0")
        if self.total_chunks < 1:
            raise ValueError("KnowledgeChunk.total_chunks 不能小于 1")
        if self.chunk_index >= self.total_chunks:
            raise ValueError("KnowledgeChunk.chunk_index 必须小于 total_chunks")

    @classmethod
    def from_parts(
        cls,
        text: str,
        chunk_index: int,
        total_chunks: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> "KnowledgeChunk":
        frozen_items = tuple(sorted((metadata or {}).items(), key=lambda item: item[0]))
        return cls(
            text=text,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            _metadata_items=frozen_items,
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        # 返回新 dict，防止外部修改影响对象内部状态。
        return dict(self._metadata_items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata,
        }

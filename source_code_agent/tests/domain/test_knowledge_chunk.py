from dataclasses import FrozenInstanceError

import pytest

from app.domain.knowledge_chunk import KnowledgeChunk


def test_knowledge_chunk_is_immutable():
    chunk = KnowledgeChunk.from_parts(
        text="hello",
        chunk_index=0,
        total_chunks=1,
        metadata={"source": "unit"},
    )
    with pytest.raises(FrozenInstanceError):
        chunk.text = "mutated"


def test_knowledge_chunk_metadata_isolation():
    raw_metadata = {"k": "v"}
    chunk = KnowledgeChunk.from_parts(
        text="hello",
        chunk_index=0,
        total_chunks=1,
        metadata=raw_metadata,
    )
    raw_metadata["k"] = "changed"
    external = chunk.metadata
    external["k"] = "outside-changed"

    assert chunk.metadata["k"] == "v"


def test_knowledge_chunk_rejects_invalid_state():
    with pytest.raises(ValueError):
        KnowledgeChunk.from_parts(text="", chunk_index=0, total_chunks=1, metadata={})
    with pytest.raises(ValueError):
        KnowledgeChunk.from_parts(text="ok", chunk_index=1, total_chunks=1, metadata={})

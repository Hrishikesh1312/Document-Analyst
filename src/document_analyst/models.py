from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PageSpan:
    page_number: int
    text: str


@dataclass(slots=True)
class DocumentRecord:
    source_path: str
    name: str
    extension: str
    size_bytes: int
    text: str
    page_spans: list[PageSpan] = field(default_factory=list)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    source_path: str
    document_name: str
    text: str
    sequence: int
    char_count: int
    approx_page: int


@dataclass(slots=True)
class SourceRecord:
    source_id: str
    document_name: str
    source_path: str
    text: str
    score: float
    approx_page: int

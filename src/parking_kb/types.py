from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

BlockKind = Literal["heading", "paragraph", "table", "list", "ocr_text"]
DocStatus = Literal["现行", "失效", "未知"]


@dataclass
class FileInfo:
    path: Path
    ext: str
    size: int
    sha256: str
    page_count: int | None = None
    has_text_layer: bool | None = None
    text_char_count: int | None = None
    is_scanned: bool = False
    error: str | None = None


@dataclass
class Block:
    kind: BlockKind
    text: str
    level: int = 0
    page: int | None = None


@dataclass
class ParsedDoc:
    doc_id: str
    path: Path
    title: str
    blocks: list[Block]
    page_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class Section:
    path: list[str]
    blocks: list[Block]


@dataclass
class DocumentMeta:
    doc_id: str
    title: str
    doc_type: str
    status: DocStatus
    doc_no: str | None = None
    issuer: str | None = None
    effective_date: str | None = None
    version: str | None = None
    source_url: str | None = None
    page_count: int = 0


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section_path: str
    page_start: int | None
    page_end: int | None
    meta: DocumentMeta


@dataclass
class Candidate:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass
class Citation:
    n: int
    chunk_id: str
    doc_id: str
    title: str
    effective_date: str | None
    section_path: str
    page_start: int | None


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    abstained: bool
    missing: list[str] = field(default_factory=list)

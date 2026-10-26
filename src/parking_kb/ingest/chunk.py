import hashlib
import re
from dataclasses import dataclass

from parking_kb.types import Block, Chunk, DocumentMeta, ParsedDoc, Section

_SENTENCE = re.compile(r"[^。！？；\n]*[。！？；\n]|[^。！？；\n]+$")
TABLE_GROUP_ROWS = 200


@dataclass(frozen=True)
class ChunkConfig:
    target: int = 500
    max: int = 800
    min: int = 100
    overlap: int = 100


def make_chunk_id(doc_id: str, section_path: str, index: int) -> str:
    raw = f"{doc_id}|{section_path}|{index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_sections(blocks: list[Block]) -> list[Section]:
    sections: list[Section] = []
    stack: list[Block] = []
    current = Section(path=[], blocks=[])
    for block in blocks:
        if block.kind == "heading":
            if current.blocks:
                sections.append(current)
            level = block.level or 1
            while stack and (stack[-1].level or 1) >= level:
                stack.pop()
            stack.append(block)
            current = Section(path=[item.text for item in stack], blocks=[])
        else:
            current.blocks.append(block)
    if current.blocks:
        sections.append(current)
    return sections


def _hard_split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def split_text(text: str, config: ChunkConfig) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= config.max:
        return [text]

    pieces: list[str] = []
    for sentence in _SENTENCE.findall(text):
        if sentence.strip():
            pieces.extend(_hard_split(sentence, config.max))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) >= config.target:
            chunks.append(current)
            overlap = min(config.overlap, config.max - len(piece))
            current = (current[-overlap:] if overlap > 0 else "") + piece
        elif current and len(current) + len(piece) > config.max:
            chunks.append(current)
            overlap = min(config.overlap, config.max - len(piece))
            current = (current[-overlap:] if overlap > 0 else "") + piece
        else:
            current += piece
    if current:
        chunks.append(current)

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < config.min and len(merged[-1]) + len(chunk) <= config.max:
            merged[-1] += chunk
        else:
            merged.append(chunk)
    return merged


def _split_table(text: str) -> list[str]:
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) <= TABLE_GROUP_ROWS + 2:
        return ["\n".join(lines)]
    header, body = lines[:2], lines[2:]
    return [
        "\n".join(header + body[i : i + TABLE_GROUP_ROWS])
        for i in range(0, len(body), TABLE_GROUP_ROWS)
    ]


def chunk_document(
    doc: ParsedDoc, meta: DocumentMeta, config: ChunkConfig = ChunkConfig()
) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0

    for section in build_sections(doc.blocks):
        path = " > ".join(section.path)
        buffer: list[Block] = []

        def flush() -> None:
            nonlocal index
            if not buffer:
                return
            text = "\n".join(block.text for block in buffer)
            pages = [block.page for block in buffer if block.page is not None]
            for part in split_text(text, config):
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(doc.doc_id, path, index),
                        doc_id=doc.doc_id,
                        text=part,
                        section_path=path,
                        page_start=pages[0] if pages else None,
                        page_end=pages[-1] if pages else None,
                        meta=meta,
                    )
                )
                index += 1
            buffer.clear()

        for block in section.blocks:
            if block.kind == "table":
                flush()
                for part in _split_table(block.text):
                    chunks.append(
                        Chunk(
                            chunk_id=make_chunk_id(doc.doc_id, path, index),
                            doc_id=doc.doc_id,
                            text=part,
                            section_path=path,
                            page_start=block.page,
                            page_end=block.page,
                            meta=meta,
                        )
                    )
                    index += 1
            else:
                buffer.append(block)
        flush()

    return chunks

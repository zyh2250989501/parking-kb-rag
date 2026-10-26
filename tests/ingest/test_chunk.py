from parking_kb.ingest.chunk import ChunkConfig, build_sections, chunk_document
from parking_kb.types import Block
from tests.factories import (
    meta,
    parsed_doc_with_headings,
    parsed_doc_with_table,
    parsed_doc_with_text,
)


def test_chunk_carries_section_path_and_size_limit():
    doc = parsed_doc_with_headings()
    chunks = chunk_document(doc, meta(), ChunkConfig())
    assert chunks[0].section_path == "第一章 总则"
    assert all(len(c.text) <= 800 for c in chunks)


def test_long_section_splits_with_overlap():
    doc = parsed_doc_with_text("第一段。" * 400)
    chunks = chunk_document(doc, meta(), ChunkConfig())
    assert len(chunks) > 1
    assert chunks[0].text[-100:] in chunks[1].text


def test_table_is_atomic_and_keeps_header():
    doc = parsed_doc_with_table()
    chunk = next(c for c in chunk_document(doc, meta()) if "|" in c.text)
    assert "| 车型 |" in chunk.text


def test_build_sections_tracks_nested_headings():
    blocks = [
        Block("heading", "第一章 总则", level=1),
        Block("paragraph", "正文一。"),
        Block("heading", "第一节 范围", level=2),
        Block("paragraph", "正文二。"),
    ]
    sections = build_sections(blocks)
    assert sections[1].path == ["第一章 总则", "第一节 范围"]
    assert [b.text for b in sections[1].blocks] == ["正文二。"]

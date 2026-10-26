from pathlib import Path

from parking_kb.types import Block, Candidate, Chunk, DocumentMeta, FileInfo, ParsedDoc


def file_info(**overrides) -> FileInfo:
    values = dict(
        path=Path("doc.pdf"),
        ext=".pdf",
        size=100,
        sha256="abc123",
        page_count=3,
        has_text_layer=True,
        text_char_count=1000,
        is_scanned=False,
    )
    values.update(overrides)
    return FileInfo(**values)


def meta(**overrides) -> DocumentMeta:
    values = dict(
        doc_id="d1",
        title="测试文档",
        doc_type="标准",
        status="现行",
        page_count=3,
    )
    values.update(overrides)
    return DocumentMeta(**values)


def chunk(chunk_id="c1", doc_id="d1", text="测试内容", **overrides) -> Chunk:
    meta_obj = overrides.pop("meta", None) or meta(doc_id=doc_id)
    values = dict(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        section_path="第一章 总则",
        page_start=1,
        page_end=1,
        meta=meta_obj,
    )
    values.update(overrides)
    return Chunk(**values)


def cand(chunk_id="c1", status="现行", text="测试内容", score=1.0, **overrides) -> Candidate:
    doc_id = overrides.pop("doc_id", "d1")
    title = overrides.pop("title", "测试文档")
    effective_date = overrides.pop("effective_date", None)
    doc_no = overrides.pop("doc_no", None)
    version = overrides.pop("version", None)
    page = overrides.pop("page", 1)
    section_path = overrides.pop("section_path", "第一章 总则")
    meta_obj = overrides.pop("meta", None) or meta(
        doc_id=doc_id,
        title=title,
        status=status,
        effective_date=effective_date,
        doc_no=doc_no,
        version=version,
    )
    c = chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        meta=meta_obj,
        section_path=section_path,
        page_start=page,
        page_end=page,
        **overrides,
    )
    return Candidate(chunk=c, score=score)


def parsed_doc_with_text(text: str, doc_id: str = "d1") -> ParsedDoc:
    return ParsedDoc(
        doc_id=doc_id,
        path=Path("doc.pdf"),
        title="测试文档",
        blocks=[Block("paragraph", text, page=1)],
        page_count=1,
    )


def parsed_doc_with_headings(doc_id: str = "d1") -> ParsedDoc:
    return ParsedDoc(
        doc_id=doc_id,
        path=Path("doc.pdf"),
        title="测试文档",
        blocks=[
            Block("heading", "第一章 总则", level=1, page=1),
            Block("paragraph", "停车收费按小时计算。", page=1),
            Block("heading", "第二章 收费", level=1, page=2),
            Block("paragraph", "月租车另行办理。", page=2),
        ],
        page_count=2,
    )


def parsed_doc_with_table(doc_id: str = "d1") -> ParsedDoc:
    return ParsedDoc(
        doc_id=doc_id,
        path=Path("doc.xlsx"),
        title="价目表",
        blocks=[
            Block("heading", "价目", level=1, page=1),
            Block("table", "| 车型 | 费率 |\n| --- | --- |\n| 小型车 | 2 元/小时 |", page=1),
        ],
        page_count=1,
    )

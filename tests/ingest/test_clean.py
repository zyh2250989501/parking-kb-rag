from pathlib import Path

from parking_kb.ingest.clean import (
    clean_document,
    dedupe_blocks,
    join_wrapped_lines,
    normalize_text,
    remove_repeated_lines,
)
from parking_kb.types import Block, ParsedDoc


def test_remove_repeated_header_and_footer():
    blocks = []
    for page in (1, 2, 3):
        blocks += [
            Block("paragraph", "内部资料 严禁外传", page=page),
            Block("paragraph", f"第 {page} 页", page=page),
            Block("paragraph", f"正文{page}", page=page),
        ]
    cleaned = remove_repeated_lines(blocks, page_count=3)
    assert [b.text for b in cleaned] == ["正文1", "正文2", "正文3"]


def test_join_wrapped_chinese_lines():
    raw = "停车收费标准按照车型\n和停放时长计算。"
    assert join_wrapped_lines(raw) == "停车收费标准按照车型和停放时长计算。"


def test_join_wrapped_lines_keeps_sentence_boundaries():
    raw = "第一句。\n第二句。"
    assert join_wrapped_lines(raw) == raw


def test_dedupe_near_duplicate_paragraphs():
    blocks = [
        Block("paragraph", "同一段话。" * 20),
        Block("paragraph", "同一段话。" * 20),
    ]
    assert len(dedupe_blocks(blocks)) == 1


def test_normalize_text_collapses_spaces_and_newlines():
    assert normalize_text("停车　 收费\r\n\r\n按小时") == "停车 收费\n\n按小时"


def test_repeated_lines_not_removed_from_single_page():
    blocks = [Block("paragraph", "内部资料 严禁外传", page=1), Block("paragraph", "正文", page=1)]
    assert len(remove_repeated_lines(blocks, page_count=1)) == 2


def test_clean_document_combines_passes():
    doc = ParsedDoc(
        doc_id="d1",
        path=Path("doc.pdf"),
        title="测试",
        blocks=[
            Block("paragraph", "内部资料 严禁外传", page=1),
            Block("paragraph", "停车收费标准按照车型\n和停放时长计算。", page=1),
            Block("paragraph", "内部资料 严禁外传", page=2),
            Block("paragraph", "停车收费标准按照车型和停放时长计算。", page=2),
            Block("paragraph", "   ", page=2),
        ],
        page_count=2,
    )
    cleaned = clean_document(doc)
    assert [b.text for b in cleaned.blocks] == ["停车收费标准按照车型和停放时长计算。"]

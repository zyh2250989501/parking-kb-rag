from parking_kb.ingest.parse.docx import parse_docx
from tests.fixtures.builders import make_docx_with_headings


def test_parse_docx_uses_heading_styles(tmp_path):
    p = make_docx_with_headings(tmp_path / "policy.docx")
    doc = parse_docx(p, "w1")
    assert doc.blocks[0].kind == "heading"
    assert doc.blocks[0].level == 1
    assert doc.blocks[0].text == "第一章 总则"
    assert any(b.kind == "paragraph" and "停车收费" in b.text for b in doc.blocks)

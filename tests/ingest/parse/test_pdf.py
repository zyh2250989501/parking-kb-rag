import pytest

from parking_kb.ingest.parse.pdf import ParseError, parse_pdf
from parking_kb.ingest.parse.tables import table_to_markdown
from tests.fixtures.builders import make_mixed_pdf


def test_parse_pdf_extracts_heading_and_table(tmp_path):
    p = make_mixed_pdf(tmp_path / "doc.pdf")
    doc = parse_pdf(p, "d1")
    assert doc.blocks[0].kind == "heading"
    assert doc.blocks[0].text == "第一章 总则"
    assert any(b.kind == "table" and "| 车型 | 费率 |" in b.text for b in doc.blocks)


def test_parse_pdf_raises_parse_error_on_corrupt(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"not a pdf")
    with pytest.raises(ParseError):
        parse_pdf(p, "bad")


def test_table_to_markdown_pads_missing_cells():
    md = table_to_markdown([["车型", "费率"], ["小型车", None]])
    assert md.splitlines()[-1] == "| 小型车 |  |"


def test_drop_degenerate_columns_handles_none_cells():
    from parking_kb.ingest.parse.pdf import _drop_degenerate_columns

    rows = [["a", None], ["b", "c"]]
    cells = [None, (0, 0, 10, 10)]
    assert _drop_degenerate_columns(rows, cells) == rows

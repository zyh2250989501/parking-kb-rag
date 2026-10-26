from parking_kb.ingest.parse.xlsx import parse_xlsx
from tests.fixtures.builders import make_xlsx_with_merged_header


def test_parse_xlsx_propagates_merged_cells(tmp_path):
    p = make_xlsx_with_merged_header(tmp_path / "prices.xlsx")
    doc = parse_xlsx(p, "x1")
    table = next(b for b in doc.blocks if b.kind == "table")
    assert "小型车" in table.text
    assert table.text.count("小型车") >= 2
    assert any(b.kind == "heading" and "价目" in b.text for b in doc.blocks)

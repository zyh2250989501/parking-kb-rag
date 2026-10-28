import pytest

from parking_kb.ingest.parse.pdf import ParseError, parse_pdf
from parking_kb.ingest.parse.tables import table_rows, table_to_markdown
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


class _StubPage:
    """页面桩：按 clip 矩形的左上角返回文字。"""

    def __init__(self, texts: dict[tuple[float, float], str]):
        self.texts = texts

    def get_text(self, mode: str, clip=None) -> str:
        return self.texts.get((round(clip.x0, 1), round(clip.y0, 1)), "")


class _StubRow:
    def __init__(self, cells):
        self.cells = cells


class _StubTable:
    """复刻大华手册 p.8 的几何：表头是 6 个窄列（含双线边框留下的空列），
    数据行是两个跨列合并的单元格。`extract()` 返回的正是手册里真实的坏结果。
    """

    HEADER = [
        (99.3, 495.0, 104.7, 510.7),
        (104.7, 495.0, 193.1, 510.7),
        (193.1, 495.0, 198.4, 510.7),
        (198.4, 495.0, 203.9, 510.7),
        (203.9, 495.0, 533.4, 510.7),
        (533.4, 495.0, 538.8, 510.7),
    ]
    BODY = [
        (99.3, 510.7, 198.4, 527.1),
        None,
        None,
        (198.4, 510.7, 538.8, 527.1),
        None,
        None,
    ]

    def __init__(self, raw):
        self.rows = [_StubRow(self.HEADER), _StubRow(self.BODY)]
        self._raw = raw

    def extract(self):
        return self._raw


def _dahua_page() -> _StubPage:
    return _StubPage(
        {
            (99.3, 495.0): "",
            (104.7, 495.0): "Parameter \n",
            (193.1, 495.0): "",
            (198.4, 495.0): "",
            (203.9, 495.0): "Description \n",
            (533.4, 495.0): "",
            (99.3, 510.7): "Processor \n",
            (198.4, 510.7): "Core i3-4160 \n",
        }
    )


def test_table_rows_realign_columns_and_keep_spacing():
    table = _StubTable(
        raw=[
            ["", "Parameter", "", "", "Description", ""],
            ["Processor", None, None, "Corei3 -4160", None, None],
        ]
    )
    rows = table_rows(_dahua_page(), table)
    assert rows == [["Parameter", "Description"], ["Processor", "Core i3-4160"]]


def test_table_rows_fall_back_to_extract_without_a_stable_grid():
    """没有重复出现的列边界（单行表）时退回 extract()，不能把表读空。"""
    table = _StubTable(raw=[["Processor", "Corei3 -4160"]])
    table.rows = [_StubRow([(0, 0, 50, 10), (50, 0, 100, 10)])]
    assert table_rows(_dahua_page(), table) == [["Processor", "Corei3 -4160"]]

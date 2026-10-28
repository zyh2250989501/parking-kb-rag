"""英文手册的表格块不能丢词间空格，也不能错列。

`Table.extract()` 把单元格文本拼成 `Corei3 -4160`，双线边框还会在真实列之间
留一批 5pt 宽的空列。后果是表格题 Recall@5 在 dense/RRF/重排四种配置里
一直停在 0.667——问题在入库的表示层，融合和重排救不回来。

用例跑真实语料，语料不入库，缺失时跳过：
`pytest -m slow tests/integration/test_table_text_quality.py`
"""

import re
from pathlib import Path

import pytest

from parking_kb.ingest.parse.pdf import parse_pdf

pytestmark = pytest.mark.slow

MANUAL = Path("corpus/dahua_pms_manual.pdf")

# 正常的英文单元格不会有 18 个字母连着；这样连着的都是没补空格的拼接结果。
GLUED = re.compile(r"[A-Za-z]{18,}")


@pytest.fixture(scope="module")
def manual():
    """整本手册解析一次要一分多钟，模块内共用。"""
    if not MANUAL.exists():
        pytest.skip("语料未下载")
    return parse_pdf(MANUAL, "d1")


def test_english_table_cells_keep_word_spacing(manual):
    tables = [block for block in manual.blocks if block.kind == "table"]
    assert tables, "手册里有表格，一个都没解析出来说明表格识别失效"
    glued = [block.text[:80] for block in tables if GLUED.search(block.text)]
    assert not glued, f"{len(glued)}/{len(tables)} 个表格块仍有粘连词，例如 {glued[0]!r}"


def test_english_table_cells_align_to_two_columns(manual):
    """p.8 的系统需求表是两列（Parameter / Description），不该读成六列。"""
    requirements = [
        block
        for block in manual.blocks
        if block.kind == "table" and "Processor" in block.text
    ]
    assert requirements, "没找到系统需求表，表格识别可能失效"
    header = requirements[0].text.splitlines()[0]
    assert header == "| Parameter | Description |", f"表头读成了 {header!r}"

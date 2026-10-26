import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from parking_kb.ingest.parse.tables import table_to_markdown
from parking_kb.types import Block, ParsedDoc

_HEADING_RE = re.compile(r"^(?:Heading|标题)\s*(\d)")


def _iter_body(document):
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document._body)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document._body)


def _heading_level(paragraph: Paragraph) -> int | None:
    style = paragraph.style.name if paragraph.style is not None else ""
    match = _HEADING_RE.match(style)
    if match:
        return min(int(match.group(1)), 3)
    if style == "Title":
        return 1
    return None


def _table_markdown(table: Table) -> str:
    rows = [[cell.text for cell in row.cells] for row in table.rows]
    return table_to_markdown(rows).strip()


def parse_docx(path: Path, doc_id: str) -> ParsedDoc:
    document = Document(path)
    blocks: list[Block] = []
    for item in _iter_body(document):
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            level = _heading_level(item)
            blocks.append(
                Block(kind="heading", text=text, level=level)
                if level is not None
                else Block(kind="paragraph", text=text)
            )
        else:
            markdown = _table_markdown(item)
            if markdown:
                blocks.append(Block(kind="table", text=markdown))
    return ParsedDoc(
        doc_id=doc_id,
        path=path,
        title=path.stem,
        blocks=blocks,
        page_count=0,
    )

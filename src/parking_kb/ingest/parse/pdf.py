from pathlib import Path

import pymupdf as fitz

from parking_kb.ingest.parse.tables import table_to_markdown
from parking_kb.types import Block, ParsedDoc


class ParseError(Exception):
    def __init__(self, path: Path, reason: str):
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


def _is_bold(span: dict) -> bool:
    return bool(span.get("flags", 0) & 16) or "bold" in span.get("font", "").lower()


def _body_size(spans: list[tuple[dict, str]]) -> float:
    weighted: list[float] = []
    for span, text in spans:
        weighted.extend([float(span.get("size", 0))] * len(text.strip()))
    if not weighted:
        return 0.0
    weighted.sort()
    return weighted[len(weighted) // 2]


def _drop_degenerate_columns(rows: list[list], cells: list) -> list[list]:
    """Drop empty columns narrower than a few points (double-drawn cell borders)."""
    if not rows or not cells:
        return rows
    drop = set()
    for i, cell in enumerate(cells):
        if cell is None:
            continue
        narrow = (cell[2] - cell[0]) < 3
        empty = all(i >= len(row) or not str(row[i] or "").strip() for row in rows)
        if narrow and empty:
            drop.add(i)
    if not drop:
        return rows
    return [[value for i, value in enumerate(row) if i not in drop] for row in rows]


def _overlaps_table(rect: fitz.Rect, tables: list[fitz.Rect]) -> bool:
    area = rect.get_area()
    if area <= 0:
        return False
    for table in tables:
        inter = fitz.Rect(rect) & table
        if inter.width <= 0 or inter.height <= 0:
            continue
        if inter.get_area() / area >= 0.5:
            return True
    return False


def _heading_level(size: float, body: float) -> int:
    if body <= 0:
        return 1
    ratio = size / body
    if ratio >= 1.6:
        return 1
    if ratio >= 1.3:
        return 2
    return 3


def _page_blocks(page, page_no: int) -> list[tuple[float, Block]]:
    tables = list(page.find_tables().tables)
    table_rects = [fitz.Rect(t.bbox) for t in tables]

    items: list[tuple[float, Block]] = []
    for table in tables:
        rows = table.extract()
        cells = table.rows[0].cells if table.rows else []
        markdown = table_to_markdown(_drop_degenerate_columns(rows, cells))
        if markdown:
            items.append((table.bbox[1], Block(kind="table", text=markdown, page=page_no)))

    raw_blocks = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        rect = fitz.Rect(block["bbox"])
        if _overlaps_table(rect, table_rects):
            continue
        spans: list[tuple[dict, str]] = []
        lines: list[str] = []
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", []))
            if line_text.strip():
                lines.append(line_text.strip())
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    spans.append((span, span["text"]))
        if not spans:
            continue
        raw_blocks.append((rect, spans, lines))

    all_spans = [pair for _, spans, _ in raw_blocks for pair in spans]
    body = _body_size(all_spans)

    for rect, spans, lines in raw_blocks:
        text = "\n".join(lines).strip()
        if not text:
            continue
        max_size = max(float(span.get("size", 0)) for span, _ in spans)
        bold_chars = sum(len(t.strip()) for span, t in spans if _is_bold(span))
        total_chars = sum(len(t.strip()) for _, t in spans)
        bold = total_chars > 0 and bold_chars / total_chars > 0.5
        is_heading = text.count("\n") <= 1 and (
            max_size >= body * 1.25 or (bold and max_size >= body * 1.1)
        )
        if is_heading:
            block = Block(
                kind="heading",
                text=text.replace("\n", " "),
                level=_heading_level(max_size, body),
                page=page_no,
            )
        else:
            block = Block(kind="paragraph", text=text, page=page_no)
        items.append((rect.y0, block))

    items.sort(key=lambda item: item[0])
    return items


def parse_pdf(path: Path, doc_id: str) -> ParsedDoc:
    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001 - normalize library errors
        raise ParseError(path, str(exc)) from exc

    try:
        if doc.needs_pass:
            raise ParseError(path, "PDF is password protected")
        blocks: list[Block] = []
        for page_no, page in enumerate(doc, start=1):
            blocks.extend(block for _, block in _page_blocks(page, page_no))
        return ParsedDoc(
            doc_id=doc_id,
            path=path,
            title=path.stem,
            blocks=blocks,
            page_count=len(doc),
        )
    except ParseError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize library errors
        raise ParseError(path, str(exc)) from exc
    finally:
        doc.close()

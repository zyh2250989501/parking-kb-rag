from pathlib import Path

from openpyxl import load_workbook

from parking_kb.ingest.parse.tables import table_to_markdown
from parking_kb.types import Block, ParsedDoc

MAX_ROWS = 5000
MAX_COLS = 100


def _sheet_rows(ws) -> list[list[str | None]]:
    max_row = min(ws.max_row or 1, MAX_ROWS)
    max_col = min(ws.max_column or 1, MAX_COLS)
    grid: list[list[str | None]] = [
        [None for _ in range(max_col)] for _ in range(max_row)
    ]

    for row in ws.iter_rows(min_row=1, max_row=max_row, min_col=1, max_col=max_col):
        for cell in row:
            if cell.value is not None:
                grid[cell.row - 1][cell.column - 1] = str(cell.value)

    for merged in ws.merged_cells.ranges:
        top_left = grid[merged.min_row - 1][merged.min_col - 1]
        for r in range(merged.min_row, min(merged.max_row, max_row) + 1):
            for c in range(merged.min_col, min(merged.max_col, max_col) + 1):
                grid[r - 1][c - 1] = top_left

    rows = [row for row in grid if any(value is not None for value in row)]
    if not rows:
        return []
    while rows and all(value is None for value in rows[0]):
        rows.pop(0)
    used_cols = [
        c for c in range(len(rows[0])) if any(row[c] is not None for row in rows)
    ]
    if not used_cols:
        return []
    first, last = used_cols[0], used_cols[-1] + 1
    return [row[first:last] for row in rows]


def parse_xlsx(path: Path, doc_id: str) -> ParsedDoc:
    workbook = load_workbook(path, data_only=True, read_only=False)
    blocks: list[Block] = []
    try:
        for ws in workbook.worksheets:
            blocks.append(Block(kind="heading", text=ws.title, level=1))
            rows = _sheet_rows(ws)
            if rows:
                blocks.append(Block(kind="table", text=table_to_markdown(rows)))
    finally:
        workbook.close()
    return ParsedDoc(
        doc_id=doc_id,
        path=path,
        title=path.stem,
        blocks=blocks,
        page_count=0,
    )

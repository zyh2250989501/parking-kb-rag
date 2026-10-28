"""表格转 markdown。

PyMuPDF 的 `Table.extract()` 有两个坑，这里都绕开：
1. 单元格文本会丢词间空格（大华手册 `Core i3` → `Corei3`），
   而按矩形从页面重新取文字是完整的；
2. 双线边框会产生一批 5pt 宽的空列，表头的列和数据行的合并单元格因此对不上，
   按 `extract()` 的行列直接拼出来是错位的。
"""

from collections import Counter

import pymupdf as fitz

# 窄于这个宽度的单元格是双线边框留下的，不是数据列。
MIN_CELL_WIDTH = 3.0


def _cell(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("\n", " ")


def _cell_text(page, rect) -> str:
    """按矩形从页面取单元格文字，空白归一。"""
    return " ".join(page.get_text("text", clip=fitz.Rect(rect)).split())


def _column_bounds(table) -> list[float]:
    """列边界。至少两行都用到才算数——只在一行里出现的边界是那一行自己的排版噪音。"""
    counts: Counter[float] = Counter()
    for row in table.rows:
        used: set[float] = set()
        for cell in row.cells:
            if cell is None or cell[2] - cell[0] < MIN_CELL_WIDTH:
                continue
            used.add(round(cell[0], 1))
            used.add(round(cell[2], 1))
        counts.update(used)
    return sorted(bound for bound, seen in counts.items() if seen >= 2)


def _column_of(cell, bounds: list[float]) -> int | None:
    center = (cell[0] + cell[2]) / 2
    for index in range(len(bounds) - 1):
        if bounds[index] <= center < bounds[index + 1]:
            return index
    return None


def _drop_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    """双线边框还会在真实列之间留下一个 2pt 宽的空列，所有行都空就丢掉。"""
    if not rows:
        return rows
    width = max(len(row) for row in rows)
    keep = [
        index
        for index in range(width)
        if any(index < len(row) and row[index] for row in rows)
    ]
    return [[row[index] if index < len(row) else "" for index in keep] for row in rows]


def table_rows(page, table) -> list[list[str]]:
    """表格内容按页面文字重取，并对齐到稳定的列网格。"""
    bounds = _column_bounds(table)
    if len(bounds) < 2:
        return _drop_empty_columns(
            [[_cell(value) for value in row] for row in table.extract()]
        )

    rows: list[list[str]] = []
    for row in table.rows:
        columns = [""] * (len(bounds) - 1)
        for cell in row.cells:
            if cell is None:
                continue
            text = _cell_text(page, cell)
            index = _column_of(cell, bounds)
            if text and index is not None:
                columns[index] = f"{columns[index]} {text}".strip()
        rows.append(columns)

    if not any(value for row in rows for value in row):
        # 矩形取不到文字（无文字层等），退回库的抽取结果总比空表强。
        return _drop_empty_columns(
            [[_cell(value) for value in row] for row in table.extract()]
        )
    return _drop_empty_columns(rows)


def table_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned = [[_cell(value) for value in row] for row in rows]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    padded = [row + [""] * (width - len(row)) for row in cleaned]
    header, *body = padded
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)

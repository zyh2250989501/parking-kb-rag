def _cell(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("\n", " ")


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

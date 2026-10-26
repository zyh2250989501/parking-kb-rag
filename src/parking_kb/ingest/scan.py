import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import pymupdf as fitz

from parking_kb.types import FileInfo

SCANNED_CHAR_THRESHOLD = 50
# Corpus bookkeeping files, not documents to ingest.
SUPPORT_FILES = {"sources.yaml", "sources.md", "metadata_overrides.yaml", "jieba_userdict.txt"}


def probe_pdf(path: Path) -> tuple[int, bool, int]:
    """Return (page_count, has_text_layer, total_text_chars)."""
    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("PDF is password protected")
        counts = [len(page.get_text("text").strip()) for page in doc]
    total = sum(counts)
    median = statistics.median(counts) if counts else 0
    return len(counts), median >= SCANNED_CHAR_THRESHOLD, total


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect(path: Path) -> FileInfo:
    ext = path.suffix.lower()
    info = FileInfo(path=path, ext=ext, size=path.stat().st_size, sha256=_sha256(path))
    if ext == ".pdf":
        info.page_count, info.has_text_layer, info.text_char_count = probe_pdf(path)
        info.is_scanned = not info.has_text_layer
    return info


def scan_corpus(root: Path) -> list[FileInfo]:
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not p.name.startswith(".") and p.name not in SUPPORT_FILES
    )
    infos: list[FileInfo] = []
    for path in files:
        try:
            infos.append(_inspect(path))
        except Exception as exc:  # noqa: BLE001 - record and keep scanning
            infos.append(
                FileInfo(
                    path=path,
                    ext=path.suffix.lower(),
                    size=path.stat().st_size if path.exists() else 0,
                    sha256="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return infos


def write_inventory_jsonl(files: list[FileInfo], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for info in files:
        record = asdict(info)
        record["path"] = str(info.path)
        lines.append(json.dumps(record, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_inventory_md(files: list[FileInfo]) -> str:
    by_ext: dict[str, int] = {}
    for info in files:
        by_ext[info.ext or "(无扩展名)"] = by_ext.get(info.ext or "(无扩展名)", 0) + 1

    pdfs = [i for i in files if i.ext == ".pdf"]
    scanned = [i for i in pdfs if i.is_scanned]
    errors = [i for i in files if i.error]

    lines = [
        "# 语料盘点报告",
        "",
        f"- 文件总数：{len(files)}",
        f"- 格式分布：{'、'.join(f'{ext} {count} 个' for ext, count in sorted(by_ext.items()))}",
        f"- PDF 扫描件：{len(scanned)} / {len(pdfs)}",
        f"- 损坏或无法读取：{len(errors)}",
        "",
        "| 文件 | 格式 | 大小(KB) | 页数 | 文本层 | 扫描件 | 错误 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for info in files:
        lines.append(
            "| {name} | {ext} | {size:.1f} | {pages} | {text} | {scanned} | {error} |".format(
                name=info.path.name,
                ext=info.ext,
                size=info.size / 1024,
                pages=info.page_count if info.page_count is not None else "-",
                text="是" if info.has_text_layer else ("否" if info.has_text_layer is False else "-"),
                scanned="是" if info.is_scanned else "否",
                error=info.error or "",
            )
        )
    return "\n".join(lines) + "\n"

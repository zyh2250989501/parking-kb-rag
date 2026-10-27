import json
from dataclasses import asdict
from pathlib import Path

from parking_kb.ingest.chunk import chunk_document
from parking_kb.ingest.clean import clean_document
from parking_kb.ingest.metadata import extract_metadata, load_overrides
from parking_kb.ingest.parse.ocr import OcrEngine
from parking_kb.ingest.parse.registry import parse_document
from parking_kb.ingest.report import (
    IngestFailure,
    IngestReport,
    IngestWarning,
    write_report,
)
from parking_kb.ingest.scan import scan_corpus
from parking_kb.types import Chunk, DocumentMeta

MISSING_FIELDS = ("doc_no", "effective_date", "issuer", "version", "source_url")


def _write_jsonl(items: list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def run_ingest(
    corpus_dir: Path,
    data_dir: Path,
    overrides_path: Path | None = None,
    ocr_engine: OcrEngine | None = None,
) -> IngestReport:
    data_dir.mkdir(parents=True, exist_ok=True)
    if overrides_path is None:
        candidate = corpus_dir / "metadata_overrides.yaml"
        overrides_path = candidate if candidate.exists() else None
    overrides = load_overrides(overrides_path) if overrides_path else {}

    infos = scan_corpus(corpus_dir)
    documents: dict[str, DocumentMeta] = {}
    chunks: list[Chunk] = []
    failures: list[IngestFailure] = []
    warnings: list[IngestWarning] = []
    succeeded = 0

    for info in infos:
        if info.error:
            failures.append(IngestFailure(str(info.path), info.error))
            continue
        doc_id = info.sha256[:12] or info.path.stem
        try:
            doc = clean_document(parse_document(info.path, doc_id, ocr_engine))
            warnings.extend(
                IngestWarning(str(info.path), message) for message in doc.warnings
            )
            meta = extract_metadata(doc, info, overrides)
            doc_chunks = chunk_document(doc, meta)
            if not doc_chunks:
                detail = "；".join(doc.warnings) or "文档为空"
                failures.append(
                    IngestFailure(
                        str(info.path), f"解析后没有可入库内容：{detail}"
                    )
                )
                continue
            documents[meta.doc_id] = meta
            chunks.extend(doc_chunks)
            succeeded += 1
        except Exception as exc:  # noqa: BLE001 - isolate per-file failures
            failures.append(
                IngestFailure(str(info.path), f"{type(exc).__name__}: {exc}")
            )

    _write_jsonl(list(documents.values()), data_dir / "documents.jsonl")
    _write_jsonl(chunks, data_dir / "chunks.jsonl")

    missing = {
        name: sum(1 for meta in documents.values() if getattr(meta, name) is None)
        for name in MISSING_FIELDS
    }
    avg_len = round(sum(len(chunk.text) for chunk in chunks) / len(chunks), 1) if chunks else 0.0
    report = IngestReport(
        total=len(infos),
        succeeded=succeeded,
        failed=len(failures),
        failures=failures,
        document_count=len(documents),
        chunk_count=len(chunks),
        scanned_docs=sum(1 for info in infos if info.is_scanned),
        avg_chunk_len=avg_len,
        missing_metadata=missing,
        warnings=warnings,
    )
    write_report(report, data_dir / "ingest_report.md")
    return report

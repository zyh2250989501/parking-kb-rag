from pathlib import Path

from parking_kb.ingest.parse.docx import parse_docx
from parking_kb.ingest.parse.ocr import OcrEngine, RapidOcrEngine, parse_scanned_pdf
from parking_kb.ingest.parse.pdf import ParseError, parse_pdf
from parking_kb.ingest.parse.xlsx import parse_xlsx
from parking_kb.ingest.scan import probe_pdf
from parking_kb.types import ParsedDoc

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xlsm"}


def parse_document(
    path: Path, doc_id: str, ocr_engine: OcrEngine | None = None
) -> ParsedDoc:
    ext = path.suffix.lower()
    if ext == ".pdf":
        _, has_text_layer, _ = probe_pdf(path)
        if has_text_layer:
            return parse_pdf(path, doc_id)
        return parse_scanned_pdf(path, doc_id, ocr_engine or RapidOcrEngine())
    if ext == ".docx":
        return parse_docx(path, doc_id)
    if ext in {".xlsx", ".xlsm"}:
        return parse_xlsx(path, doc_id)
    raise ParseError(path, f"unsupported extension: {ext or '(none)'}")

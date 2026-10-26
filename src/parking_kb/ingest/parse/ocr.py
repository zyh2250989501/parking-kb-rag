from pathlib import Path
from typing import Protocol

import pymupdf as fitz

from parking_kb.ingest.parse.pdf import ParseError
from parking_kb.types import Block, ParsedDoc


class OcrEngine(Protocol):
    def recognize(self, image_png: bytes) -> str: ...


class RapidOcrEngine:
    """Lazy wrapper so importing the project never loads the OCR models."""

    def __init__(self) -> None:
        self._engine = None

    def recognize(self, image_png: bytes) -> str:
        if self._engine is None:
            import cv2
            import numpy as np
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
            self._cv2 = cv2
            self._np = np
        image = self._cv2.imdecode(
            self._np.frombuffer(image_png, dtype=self._np.uint8),
            self._cv2.IMREAD_COLOR,
        )
        result, _ = self._engine(image)
        if not result:
            return ""
        return "\n".join(item[1] for item in result)


def parse_scanned_pdf(path: Path, doc_id: str, engine: OcrEngine) -> ParsedDoc:
    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001 - normalize library errors
        raise ParseError(path, str(exc)) from exc

    try:
        if doc.needs_pass:
            raise ParseError(path, "PDF is password protected")
        blocks: list[Block] = []
        warnings: list[str] = []
        for page_no, page in enumerate(doc, start=1):
            png = page.get_pixmap(dpi=200).tobytes("png")
            text = engine.recognize(png).strip()
            if text:
                blocks.append(Block(kind="ocr_text", text=text, page=page_no))
            else:
                warnings.append(f"第 {page_no} 页 OCR 未识别出文本")
        return ParsedDoc(
            doc_id=doc_id,
            path=path,
            title=path.stem,
            blocks=blocks,
            page_count=len(doc),
            warnings=warnings,
        )
    except ParseError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize library errors
        raise ParseError(path, str(exc)) from exc
    finally:
        doc.close()

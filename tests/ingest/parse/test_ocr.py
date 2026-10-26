from parking_kb.ingest.parse.registry import parse_document
from tests.fixtures.builders import (
    make_docx_with_headings,
    make_image_only_pdf,
    make_text_pdf,
)


class FakeOcr:
    def __init__(self, text: str):
        self.text = text

    def recognize(self, image_png: bytes) -> str:
        return self.text


def test_scanned_pdf_routes_to_ocr(tmp_path):
    p = make_image_only_pdf(tmp_path / "scan.pdf")
    doc = parse_document(p, "s1", ocr_engine=FakeOcr("停车收费标准"))
    assert doc.blocks[0].kind == "ocr_text"
    assert "停车收费标准" in doc.blocks[0].text
    assert doc.blocks[0].page == 1


def test_text_pdf_never_calls_ocr(tmp_path):
    p = make_text_pdf(tmp_path / "text.pdf")

    class Boom:
        def recognize(self, _):
            raise AssertionError("OCR should not be called")

    doc = parse_document(p, "t1", ocr_engine=Boom())
    assert doc.blocks
    assert all(b.kind != "ocr_text" for b in doc.blocks)


def test_empty_ocr_text_records_warning(tmp_path):
    p = make_image_only_pdf(tmp_path / "scan.pdf")
    doc = parse_document(p, "s2", ocr_engine=FakeOcr("   "))
    assert doc.blocks == []
    assert doc.warnings


def test_parse_document_dispatches_docx(tmp_path):
    p = make_docx_with_headings(tmp_path / "policy.docx")
    doc = parse_document(p, "w1")
    assert doc.blocks[0].kind == "heading"

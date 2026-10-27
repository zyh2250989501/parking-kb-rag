import json
import shutil

from parking_kb.ingest.pipeline import run_ingest
from parking_kb.ingest.report import write_report
from tests.fixtures.builders import (
    make_fixture_corpus,
    make_image_only_pdf,
    make_text_pdf,
)


class EmptyOcr:
    def recognize(self, image_png: bytes) -> str:
        return ""


class PageOcr:
    def __init__(self, texts: list[str]):
        self.texts = list(texts)

    def recognize(self, image_png: bytes) -> str:
        return self.texts.pop(0) if self.texts else ""


def test_ingest_continues_after_corrupt_file(tmp_path):
    corpus, data = make_fixture_corpus(tmp_path)
    report = run_ingest(corpus, data)
    assert report.succeeded == 2
    assert report.failed == 1
    assert "bad.pdf" in report.failures[0].path
    assert (data / "chunks.jsonl").exists()


def test_ingest_writes_report_with_failure_reason(tmp_path):
    corpus, data = make_fixture_corpus(tmp_path)
    report = run_ingest(corpus, data)
    write_report(report, data / "ingest_report.md")
    text = (data / "ingest_report.md").read_text(encoding="utf-8")
    assert "bad.pdf" in text and "失败原因" in text


def test_ingest_writes_documents_and_chunks_jsonl(tmp_path):
    corpus, data = make_fixture_corpus(tmp_path)
    run_ingest(corpus, data)
    documents = [
        json.loads(line)
        for line in (data / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    chunks = [
        json.loads(line)
        for line in (data / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(documents) == 2
    assert chunks
    assert {
        "chunk_id",
        "doc_id",
        "text",
        "section_path",
        "page_start",
        "page_end",
        "meta",
    } <= set(chunks[0])
    assert chunks[0]["meta"]["doc_id"] == chunks[0]["doc_id"]


def test_duplicate_files_count_as_successes_but_one_document(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    make_text_pdf(corpus / "a.pdf")
    shutil.copy(corpus / "a.pdf", corpus / "b.pdf")
    report = run_ingest(corpus, tmp_path / "data")
    assert report.succeeded == 2
    assert report.document_count == 1


def test_empty_ocr_document_is_a_failure_not_a_silent_success(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    make_image_only_pdf(corpus / "scan.pdf")
    report = run_ingest(corpus, tmp_path / "data", ocr_engine=EmptyOcr())
    assert report.succeeded == 0
    assert report.failed == 1
    assert report.chunk_count == 0
    assert "OCR" in report.failures[0].error


def test_ocr_warnings_are_surfaced_in_the_report(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    make_image_only_pdf(corpus / "scan.pdf", pages=2)
    report = run_ingest(
        corpus,
        tmp_path / "data",
        ocr_engine=PageOcr(["", "停车收费标准按小时计算。"]),
    )
    assert report.succeeded == 1
    assert any(
        "第 1 页 OCR 未识别出文本" in warning.message
        for warning in report.warnings
    )
    text = (tmp_path / "data" / "ingest_report.md").read_text(encoding="utf-8")
    assert "第 1 页 OCR 未识别出文本" in text

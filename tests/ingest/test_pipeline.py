import json
import shutil

from parking_kb.ingest.pipeline import run_ingest
from parking_kb.ingest.report import write_report
from tests.fixtures.builders import make_fixture_corpus, make_text_pdf


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

from parking_kb.ingest.scan import render_inventory_md, scan_corpus
from tests.fixtures.builders import make_image_only_pdf, make_text_pdf


def test_scan_detects_scanned_pdf(tmp_path):
    make_image_only_pdf(tmp_path / "scan.pdf")
    info = scan_corpus(tmp_path)[0]
    assert info.ext == ".pdf"
    assert info.is_scanned is True
    assert info.has_text_layer is False


def test_scan_records_corrupt_file_and_continues(tmp_path):
    make_text_pdf(tmp_path / "ok.pdf")
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    infos = scan_corpus(tmp_path)
    assert len(infos) == 2
    assert any(i.error for i in infos)
    assert any(i.error is None for i in infos)


def test_inventory_report_contains_counts(tmp_path):
    make_text_pdf(tmp_path / "ok.pdf")
    make_image_only_pdf(tmp_path / "scan.pdf")
    md = render_inventory_md(scan_corpus(tmp_path))
    assert "PDF" in md
    assert "扫描" in md
    assert "2" in md

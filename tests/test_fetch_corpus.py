from pathlib import Path

from parking_kb.ingest.fetch import (
    SourceEntry,
    download_sources,
    load_sources,
    render_sources_md,
)


def test_download_sources_writes_files(tmp_path):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return b"%PDF-1.4 test"

    sources = [SourceEntry("a.pdf", "https://example.com/a.pdf", "标准", "public")]
    paths = download_sources(sources, tmp_path, fetch=fake_fetch)
    assert paths == [tmp_path / "a.pdf"]
    assert calls == ["https://example.com/a.pdf"]


def test_download_sources_skips_existing(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"old")
    paths = download_sources(
        [SourceEntry("a.pdf", "https://x", "标准", "public")], tmp_path
    )
    assert paths[0].read_bytes() == b"old"


def test_download_sources_skips_manual_entries(tmp_path):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return b"data"

    sources = [
        SourceEntry("a.pdf", "https://x/a.pdf", "标准", "public", manual=True),
        SourceEntry("b.pdf", "https://x/b.pdf", "标准", "public"),
    ]
    paths = download_sources(sources, tmp_path, fetch=fake_fetch)
    assert [p.name for p in paths] == ["b.pdf"]
    assert calls == ["https://x/b.pdf"]


def test_load_sources_reads_yaml(tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        "sources:\n"
        "  - name: a.pdf\n"
        "    url: https://x/a.pdf\n"
        "    doc_type: 标准\n"
        "    license_note: public\n",
        encoding="utf-8",
    )
    sources = load_sources(manifest)
    assert sources[0].name == "a.pdf"
    assert sources[0].manual is False


def test_render_sources_md_lists_entries():
    md = render_sources_md(
        [SourceEntry("a.pdf", "https://x/a.pdf", "标准", "public")]
    )
    assert "a.pdf" in md and "https://x/a.pdf" in md and "标准" in md

from pathlib import Path

from parking_kb.ingest.fetch import download_sources, load_sources, render_sources_md


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "corpus" / "sources.yaml"
    sources = load_sources(manifest)
    (root / "corpus" / "sources.md").write_text(
        render_sources_md(sources), encoding="utf-8"
    )
    paths = download_sources(sources, root / "corpus")
    print(f"downloaded {len(paths)} files into {root / 'corpus'}")


if __name__ == "__main__":
    main()

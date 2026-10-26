import argparse
from pathlib import Path

from parking_kb.ingest.pipeline import run_ingest
from parking_kb.ingest.report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest corpus into chunks and metadata.")
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=Path("reports/ingest_report.md"))
    args = parser.parse_args()

    report = run_ingest(args.corpus, args.data, args.overrides)
    write_report(report, args.report)
    print(
        f"files={report.total} succeeded={report.succeeded} failed={report.failed} "
        f"documents={report.document_count} chunks={report.chunk_count}"
    )


if __name__ == "__main__":
    main()

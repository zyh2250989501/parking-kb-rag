from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IngestFailure:
    path: str
    error: str


@dataclass
class IngestWarning:
    path: str
    message: str


@dataclass
class IngestReport:
    total: int
    succeeded: int
    failed: int
    failures: list[IngestFailure] = field(default_factory=list)
    document_count: int = 0
    chunk_count: int = 0
    scanned_docs: int = 0
    avg_chunk_len: float = 0.0
    missing_metadata: dict[str, int] = field(default_factory=dict)
    warnings: list[IngestWarning] = field(default_factory=list)


def write_report(report: IngestReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 入库报告",
        "",
        "## 概览",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 文件总数 | {report.total} |",
        f"| 成功 | {report.succeeded} |",
        f"| 失败 | {report.failed} |",
        f"| 文档数 | {report.document_count} |",
        f"| Chunk 数 | {report.chunk_count} |",
        f"| 平均 Chunk 长度 | {report.avg_chunk_len} |",
        f"| 扫描件 | {report.scanned_docs} |",
        "",
        "## 元数据缺失",
        "",
        "| 字段 | 缺失文档数 |",
        "| --- | --- |",
    ]
    if report.missing_metadata:
        lines.extend(
            f"| {name} | {count} |" for name, count in sorted(report.missing_metadata.items())
        )
    else:
        lines.append("| 无 | 0 |")

    lines.extend(["", "## 失败清单", "", "| 文件 | 失败原因 |", "| --- | --- |"])
    if report.failures:
        lines.extend(f"| {item.path} | {item.error} |" for item in report.failures)
    else:
        lines.append("| 无 | - |")

    lines.extend(["", "## 警告", "", "| 文件 | 警告 |", "| --- | --- |"])
    if report.warnings:
        lines.extend(f"| {item.path} | {item.message} |" for item in report.warnings)
    else:
        lines.append("| 无 | - |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

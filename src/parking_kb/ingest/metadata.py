import re
from datetime import date
from pathlib import Path

import yaml

from parking_kb.types import DocumentMeta, DocStatus, FileInfo, ParsedDoc

_DOC_NO = re.compile(r"[一-鿿]{1,12}[〔\[]\d{4}[〕\]]\s*\d+\s*号")
_EFFECTIVE = re.compile(
    r"(?:自|于)\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*"
    r"(?:起)?\s*(?:施行|实施|执行|生效)"
)
_SELF_EXPIRED = re.compile(
    r"本(?:办法|规定|文件|标准|规范|细则)[^。\n]{0,40}"
    r"(?:废止|失效|停止执行|已被替代)"
    r"|自[^。\n]{0,30}起(?:废止|失效)"
)
_SELF_ACTIVE = re.compile(
    r"本(?:办法|规定|文件|标准|规范|细则)\s*(?:自|于)"
    r"[^。\n]{0,40}(?:施行|实施|执行|生效)"
    r"|自\s*\d{4}\s*年[^。\n]{0,20}起\s*(?:施行|实施|执行|生效)"
)
META_FIELDS = {
    "title",
    "doc_type",
    "status",
    "doc_no",
    "issuer",
    "effective_date",
    "version",
    "source_url",
    "page_count",
}


def parse_doc_no(text: str) -> str | None:
    match = _DOC_NO.search(text)
    return match.group(0) if match else None


def parse_effective_date(text: str) -> str | None:
    match = _EFFECTIVE.search(text)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def parse_status(text: str) -> DocStatus:
    if _SELF_EXPIRED.search(text):
        return "失效"
    if _SELF_ACTIVE.search(text):
        return "现行"
    return "未知"


def load_overrides(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict) and "overrides" in data:
        data = data["overrides"] or {}
    if not isinstance(data, dict):
        return {}
    return {str(key): dict(value) for key, value in data.items() if isinstance(value, dict)}


def extract_metadata(
    doc: ParsedDoc, file_info: FileInfo, overrides: dict[str, dict]
) -> DocumentMeta:
    text = "\n".join(block.text for block in doc.blocks)
    values = {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "doc_type": "未分类",
        "status": parse_status(text),
        "doc_no": parse_doc_no(text),
        "effective_date": parse_effective_date(text),
        "page_count": file_info.page_count or doc.page_count,
    }
    for key, value in overrides.get(file_info.sha256, {}).items():
        if key in META_FIELDS:
            values[key] = value
    return DocumentMeta(**values)

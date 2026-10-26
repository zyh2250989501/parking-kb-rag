from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx
import yaml


@dataclass
class SourceEntry:
    name: str
    url: str
    doc_type: str
    license_note: str
    manual: bool = False
    notes: str = ""


def _http_get(url: str) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    with httpx.Client(follow_redirects=True, timeout=30, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def load_sources(path: Path) -> list[SourceEntry]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [SourceEntry(**item) for item in data["sources"]]


def render_sources_md(sources: list[SourceEntry]) -> str:
    lines = [
        "# 语料来源清单",
        "",
        "| 文件名 | 类型 | 获取方式 | 来源 | 许可说明 | 备注 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in sources:
        mode = "手动下载" if s.manual else "脚本下载"
        lines.append(
            f"| {s.name} | {s.doc_type} | {mode} | {s.url} | {s.license_note} | {s.notes} |"
        )
    return "\n".join(lines) + "\n"


def download_sources(
    sources: list[SourceEntry],
    dest: Path,
    fetch: Callable[[str], bytes] = _http_get,
) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for source in sources:
        target = dest / source.name
        if source.manual:
            print(f"[manual] {source.name}: {source.url}")
            continue
        if target.exists():
            paths.append(target)
            continue
        try:
            target.write_bytes(fetch(source.url))
            paths.append(target)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"[failed] {source.name}: {source.url} ({exc})")
    return paths

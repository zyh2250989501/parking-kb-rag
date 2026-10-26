import re

from parking_kb.index.store import ChunkStore
from parking_kb.types import Candidate

UNKNOWN_WEIGHT = 0.95
_TEMPORAL = re.compile(r"旧|原|以前|历史|之前|废止|(?:19|20)\d{2}\s*年")


def has_temporal_cue(query: str) -> bool:
    return bool(_TEMPORAL.search(query))


def _latest_active_doc_ids(
    store: ChunkStore, doc_nos: set[str]
) -> dict[str, str]:
    latest: dict[str, str] = {}
    for doc_no in doc_nos:
        active = [doc for doc in store.find_versions(doc_no) if doc.status == "现行"]
        if active:
            newest = max(active, key=lambda doc: doc.effective_date or "")
            latest[doc_no] = newest.doc_id
    return latest


def apply_status_filter(
    candidates: list[Candidate], query: str, store: ChunkStore
) -> list[Candidate]:
    if has_temporal_cue(query):
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    kept: list[Candidate] = []
    for candidate in candidates:
        status = candidate.chunk.meta.status
        if status == "失效":
            continue
        if status == "未知":
            candidate.score *= UNKNOWN_WEIGHT
        kept.append(candidate)

    doc_nos = {c.chunk.meta.doc_no for c in kept if c.chunk.meta.doc_no}
    latest = _latest_active_doc_ids(store, doc_nos)
    kept = [
        candidate
        for candidate in kept
        if candidate.chunk.meta.doc_no not in latest
        or candidate.chunk.doc_id == latest[candidate.chunk.meta.doc_no]
    ]
    kept.sort(key=lambda candidate: candidate.score, reverse=True)
    return kept

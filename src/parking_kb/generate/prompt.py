import json

from parking_kb.types import AnswerResult, Candidate, Citation


def build_context(chunks: list[Candidate]) -> str:
    blocks = []
    for number, candidate in enumerate(chunks, start=1):
        chunk = candidate.chunk
        meta = chunk.meta
        details = []
        if meta.doc_no:
            details.append(meta.doc_no)
        if meta.effective_date:
            details.append(f"{meta.effective_date} 生效")
        details.append(meta.status)
        header = f"[{number}] 《{meta.title}》（{'，'.join(details)}）"
        page = f"p.{chunk.page_start}" if chunk.page_start else "页码未知"
        section = chunk.section_path or "无"
        blocks.append(
            f"{header}\n    章节：{section}\n    页码：{page}\n    内容：{chunk.text}"
        )
    return "\n\n".join(blocks)


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_answer_json(raw: str, chunks: list[Candidate]) -> AnswerResult:
    try:
        data = json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid answer json: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("answer json must be an object")
    answer = data.get("answer")
    if not isinstance(answer, str):
        raise ValueError("answer field must be a string")

    citations: list[Citation] = []
    for entry in data.get("citations") or []:
        number = entry.get("n") if isinstance(entry, dict) else entry
        if isinstance(number, bool) or not isinstance(number, int):
            continue
        if not 1 <= number <= len(chunks):
            continue
        chunk = chunks[number - 1].chunk
        citations.append(
            Citation(
                n=number,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.meta.title,
                effective_date=chunk.meta.effective_date,
                section_path=chunk.section_path,
                page_start=chunk.page_start,
            )
        )

    missing = data.get("missing") or []
    if not isinstance(missing, list):
        missing = [str(missing)]

    return AnswerResult(
        answer=answer,
        citations=citations,
        abstained=bool(data.get("abstained", False)),
        missing=[str(item) for item in missing],
    )

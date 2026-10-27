import json

from parking_kb.eval.schema import EvalQuestion
from parking_kb.generate.client import LLMProvider
from parking_kb.index.sparse import SparseIndex
from parking_kb.types import Chunk

_SYSTEM = """你是评估集质检员。判断给定资料能否直接回答用户问题。
只有在资料中出现明确的条款、数值或结论时才算“能回答”；仅仅主题相关不算。
只输出 JSON：{"answerable": true} 或 {"answerable": false}"""


def _parse_verdict(raw: str) -> bool:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return bool(json.loads(text).get("answerable"))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return False


def screen_unanswerable(
    questions: list[EvalQuestion],
    chunks: list[Chunk],
    llm: LLMProvider,
    top_k: int = 5,
    limit: int | None = None,
) -> tuple[list[EvalQuestion], list[EvalQuestion]]:
    """Drop unanswerable questions that the corpus can actually answer.

    Each question is checked against its top BM25 hits. An unparseable
    verdict keeps the question: screening is advisory, manual review decides.
    """
    if not questions:
        return [], []
    sparse = SparseIndex.build(chunks)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    kept: list[EvalQuestion] = []
    dropped: list[EvalQuestion] = []

    for question in questions:
        if limit is not None and len(kept) >= limit:
            break
        hits = sparse.search(question.question, top_k)
        if not hits:
            kept.append(question)
            continue
        context = "\n\n".join(
            f"[{number}] 《{by_id[chunk_id].meta.title}》\n{by_id[chunk_id].text[:600]}"
            for number, (chunk_id, _) in enumerate(hits, start=1)
        )
        raw = llm.complete(
            [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": f"## 资料\n{context}\n\n## 问题\n{question.question}",
                },
            ],
            temperature=0.0,
        )
        if _parse_verdict(raw):
            dropped.append(question)
        else:
            kept.append(question)
    return kept, dropped

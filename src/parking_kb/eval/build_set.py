import json
import logging
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from parking_kb.eval.schema import EvalQuestion, GoldSpan
from parking_kb.generate.client import LLMProvider
from parking_kb.types import Chunk

logger = logging.getLogger(__name__)

QUESTION_TYPES = ("fact", "table", "cross_doc", "version", "unanswerable")

_SYSTEM = """你是智慧停车企业知识库的评估集构建助手。根据给定资料生成中文评测问题。
只输出一个 JSON 对象，不要输出 Markdown 代码块或额外解释：
{"questions": [{"question": "...", "answer": "...", "quote": "...", "type": "...", "difficulty": "easy|medium|hard"}]}
规则：
- question 必须是资料能回答的问题（unanswerable 除外）。
- answer 是标准答案，简洁准确。
- quote 必须逐字摘自资料，用于定位答案出处。
- 不要在问题中加编号。"""

_INSTRUCTIONS = {
    "fact": "生成 {count} 个单文档事实查找题，覆盖不同条款或段落。",
    "table": "生成 {count} 个表格/价目查询题，quote 必须包含表头或对应数值行。",
    "cross_doc": "生成 {count} 个需要综合两份资料对比的问题。",
    "version": "生成 {count} 个新旧版本或效力对比题，答案需说明适用版本与生效日期。",
    "unanswerable": (
        "生成 {count} 个与停车业务相关、但所给资料无法回答的问题。"
        "优先询问资料未收录的其他城市或省份的管理规定、其他厂商的设备参数、"
        "企业内部制度这类资料完全不可能覆盖的内容；"
        "不要询问所给资料中已有条款、数值或概念的细节。"
        "answer 写“语料未提及”，quote 可省略。"
    ),
}


def _looks_like_table(text: str) -> bool:
    return "| --- |" in text or text.lstrip().startswith("|")


def _select_context(
    question_type: str, chunks: list[Chunk], count: int
) -> list[Chunk]:
    if question_type == "table":
        pool = [item for item in chunks if _looks_like_table(item.text)]
        return pool[: max(count * 3, 5)]

    if question_type == "cross_doc":
        by_doc: dict[str, Chunk] = {}
        for item in chunks:
            by_doc.setdefault(item.doc_id, item)
        docs = sorted(by_doc)[:2]
        return [by_doc[doc_id] for doc_id in docs] if len(docs) >= 2 else []

    if question_type == "version":
        # A version question needs two documents sharing a doc_no. Pairing
        # unrelated documents of the same doc_type produces fact questions
        # that look like version questions and would corrupt that bucket.
        by_doc_no: dict[str, dict[str, Chunk]] = defaultdict(dict)
        for item in chunks:
            if item.meta.doc_no:
                by_doc_no[item.meta.doc_no].setdefault(item.doc_id, item)
        for group in by_doc_no.values():
            if len(group) >= 2:
                return list(group.values())[:2]
        return []

    if question_type == "unanswerable":
        return chunks[: max(count, 5)]

    pool = [item for item in chunks if len(item.text) >= 20] or chunks
    return pool[: max(count * 3, 5)]


def _build_messages(
    question_type: str, context: list[Chunk], count: int
) -> list[dict[str, str]]:
    blocks = [
        f"[{number}] 《{item.meta.title}》\n{item.text[:800]}"
        for number, item in enumerate(context, start=1)
    ]
    instruction = _INSTRUCTIONS[question_type].format(count=count)
    return [
        {"role": "system", "content": f"{_SYSTEM}\n\n任务：{instruction}"},
        {"role": "user", "content": "## 资料\n" + "\n\n".join(blocks)},
    ]


def _load_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _parse_questions(
    raw: str, question_type: str, context: list[Chunk], limit: int
) -> list[EvalQuestion]:
    try:
        data = _load_json(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("failed to parse %s questions: %s", question_type, raw[:200])
        return []
    items = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []

    questions: list[EvalQuestion] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer")
        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(answer, str) or not answer.strip():
            continue
        difficulty = item.get("difficulty") or "medium"

        if question_type == "unanswerable":
            questions.append(
                EvalQuestion(
                    qid="",
                    question=question.strip(),
                    gold_answer=answer.strip(),
                    gold_span=GoldSpan("", 0, 0, ""),
                    question_type=question_type,
                    difficulty=difficulty,
                    source_docs=[],
                )
            )
        else:
            quote = item.get("quote")
            if not isinstance(quote, str) or not quote:
                logger.warning("missing quote for %s question", question_type)
                continue
            match: tuple[Chunk, int] | None = None
            for candidate in context:
                start = candidate.text.find(quote)
                if start >= 0:
                    match = (candidate, start)
                    break
            if match is None:
                logger.warning("quote not found in context: %s", quote[:50])
                continue
            source, start = match
            source_docs = (
                sorted({candidate.doc_id for candidate in context})
                if question_type in ("cross_doc", "version")
                else [source.doc_id]
            )
            questions.append(
                EvalQuestion(
                    qid="",
                    question=question.strip(),
                    gold_answer=answer.strip(),
                    gold_span=GoldSpan(
                        source.doc_id, start, start + len(quote), quote
                    ),
                    question_type=question_type,
                    difficulty=difficulty,
                    source_docs=source_docs,
                )
            )
        if len(questions) >= limit:
            break
    return questions


def build_eval_set(
    chunks: list[Chunk], llm: LLMProvider, per_type: dict[str, int]
) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    counters: dict[str, int] = defaultdict(int)
    for question_type in QUESTION_TYPES:
        limit = per_type.get(question_type, 0)
        if limit <= 0:
            continue
        context = _select_context(question_type, chunks, limit)
        if not context:
            logger.warning("no context available for %s questions", question_type)
            continue
        raw = llm.complete(_build_messages(question_type, context, limit))
        for question in _parse_questions(raw, question_type, context, limit):
            counters[question_type] += 1
            questions.append(
                replace(
                    question,
                    qid=f"{question_type}-{counters[question_type]:03d}",
                )
            )
    return questions


def write_eval_set(questions: list[EvalQuestion], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(asdict(question), ensure_ascii=False) for question in questions
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_eval_set(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            span = record.get("gold_span")
            questions.append(
                EvalQuestion(
                    qid=record["qid"],
                    question=record["question"],
                    gold_answer=record["gold_answer"],
                    gold_span=GoldSpan(**span) if span else None,
                    question_type=record["question_type"],
                    difficulty=record["difficulty"],
                    source_docs=record.get("source_docs", []),
                    verified=record.get("verified", False),
                )
            )
    return questions


def resolve_gold_chunks(question: EvalQuestion, chunks: list[Chunk]) -> list[str]:
    span = question.gold_span
    if span is None or not span.doc_id or not span.quote:
        return []
    return [
        item.chunk_id
        for item in chunks
        if item.doc_id == span.doc_id and span.quote in item.text
    ]

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldSpan:
    doc_id: str
    char_start: int
    char_end: int
    quote: str


@dataclass(frozen=True)
class EvalQuestion:
    qid: str
    question: str
    gold_answer: str
    gold_span: GoldSpan | None
    question_type: str
    difficulty: str
    source_docs: list[str]
    verified: bool = False

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

from parking_kb.eval.schema import EvalQuestion
from parking_kb.generate.client import LLMProvider
from parking_kb.types import AnswerResult, Chunk

Correctness = Literal["correct", "partial", "wrong"]

_SYSTEM = """你是独立、严格的答案评估员。只输出一个 JSON 对象，不要输出解释或代码块：
{"correct": "correct|partial|wrong", "faithful": true, "citation_precise": true, "reason": "..."}

评估口径：
- correct：对照“标准答案”判断待评估答案，correct=完全正确，partial=部分正确或遗漏关键限制，wrong=错误、答非所问或本应拒答却作答。
- faithful：逐句检查答案中的事实性陈述是否被“引用上下文”支持；出现上下文之外的内容即为 false。
- citation_precise：答案的引用是否精确指向支持该陈述的上下文条目。
- reason：一句话说明主要判断依据。"""


@dataclass(frozen=True)
class Verdict:
    correct: Correctness
    faithful: bool
    citation_precise: bool
    reason: str


class Judge(Protocol):
    def judge(
        self,
        question: EvalQuestion,
        answer: AnswerResult,
        contexts: list[Chunk],
    ) -> Verdict: ...


def _load_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


class LLMJudge:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def _messages(
        self,
        question: EvalQuestion,
        answer: AnswerResult,
        contexts: list[Chunk],
    ) -> list[dict[str, str]]:
        context_text = "\n\n".join(
            f"[{number}] 《{chunk.meta.title}》\n{chunk.text}"
            for number, chunk in enumerate(contexts, start=1)
        )
        citations = "\n".join(
            f"[{citation.n}] chunk_id={citation.chunk_id}" for citation in answer.citations
        )
        span_quote = question.gold_span.quote if question.gold_span else ""
        user = (
            f"## 问题\n{question.question}\n\n"
            f"## 标准答案\n{question.gold_answer}\n\n"
            f"## 标准答案出处\n{span_quote or '无'}\n\n"
            f"## 待评估答案\n{answer.answer}\n\n"
            f"## 答案引用\n{citations or '无'}\n\n"
            f"## 引用上下文\n{context_text or '无'}"
        )
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ]

    def judge(
        self,
        question: EvalQuestion,
        answer: AnswerResult,
        contexts: list[Chunk],
    ) -> Verdict:
        raw = self.provider.complete(
            self._messages(question, answer, contexts), temperature=0.0
        )
        try:
            data = _load_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            return Verdict("wrong", False, False, f"解析失败: {exc}")
        correct = data.get("correct")
        if correct not in ("correct", "partial", "wrong"):
            return Verdict(
                "wrong", False, False, f"解析失败: 非法 correct 值 {correct!r}"
            )
        return Verdict(
            correct=correct,
            faithful=bool(data.get("faithful", False)),
            citation_precise=bool(data.get("citation_precise", False)),
            reason=str(data.get("reason", "")),
        )


def save_verdicts(verdicts: dict[str, Verdict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"key": key, **asdict(verdict)}, ensure_ascii=False)
        for key, verdict in verdicts.items()
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_verdicts(path: Path) -> dict[str, Verdict]:
    verdicts: dict[str, Verdict] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.pop("key")
            verdicts[key] = Verdict(**record)
    return verdicts

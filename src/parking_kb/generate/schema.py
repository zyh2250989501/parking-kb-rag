from pathlib import Path

from parking_kb.generate.client import LLMProvider
from parking_kb.generate.prompt import build_context, parse_answer_json
from parking_kb.types import AnswerResult, Candidate

DEFAULT_PROMPT = Path("prompts/answer_v2.md")


class AnswerGenerator:
    def __init__(
        self, provider: LLMProvider, prompt_path: Path = DEFAULT_PROMPT
    ):
        self.provider = provider
        self.prompt = Path(prompt_path).read_text(encoding="utf-8")

    def generate(
        self, question: str, chunks: list[Candidate]
    ) -> AnswerResult:
        messages = [
            {"role": "system", "content": self.prompt},
            {
                "role": "user",
                "content": f"## 上下文\n{build_context(chunks)}\n\n## 问题\n{question}",
            },
        ]
        raw = self.provider.complete(messages, temperature=0.0)
        try:
            return parse_answer_json(raw, chunks)
        except ValueError:
            retry = self.provider.complete(messages, temperature=0.0)
            try:
                return parse_answer_json(retry, chunks)
            except ValueError:
                return AnswerResult(
                    answer=retry,
                    citations=[],
                    abstained=True,
                    missing=["模型输出无法解析"],
                )

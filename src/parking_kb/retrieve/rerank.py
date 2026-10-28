from typing import Protocol

from parking_kb.index.search_text import index_text
from parking_kb.types import Candidate


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class BgeReranker:
    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [[query, text] for text in texts]
        return [float(value) for value in self._load().predict(pairs)]


def rerank(
    candidates: list[Candidate],
    query: str,
    reranker: Reranker,
    top_k: int = 8,
) -> list[Candidate]:
    if not candidates:
        return []
    # 与检索用同一份文本：英文段落只有带上中文检索描述，才能和中文问句公平比较。
    scores = reranker.score(
        query, [index_text(candidate.chunk) for candidate in candidates]
    )
    for candidate, score in zip(candidates, scores):
        candidate.score = float(score)
    ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    return ranked[:top_k]

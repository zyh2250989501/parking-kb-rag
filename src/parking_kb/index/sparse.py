import pickle
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from parking_kb.index.search_text import index_text
from parking_kb.types import Chunk

_USERDICT = Path(__file__).resolve().parents[3] / "corpus" / "jieba_userdict.txt"
_TOKEN = re.compile(r"[一-鿿A-Za-z0-9]+")
_STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "及",
    "或",
    "在",
    "是",
    "为",
    "对",
    "由",
    "等",
    "其",
    "将",
    "并",
    "而",
    "也",
    "都",
    "就",
}
_loaded = False


def _ensure_userdict() -> None:
    global _loaded
    if not _loaded:
        if _USERDICT.exists():
            jieba.load_userdict(str(_USERDICT))
        _loaded = True


def tokenize_zh(text: str) -> list[str]:
    _ensure_userdict()
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in jieba.lcut_for_search(text):
        token = raw.strip().lower()
        if not token or token in _STOPWORDS or not _TOKEN.fullmatch(token):
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


class SparseIndex:
    def __init__(
        self,
        chunk_ids: list[str],
        tokenized: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.chunk_ids = chunk_ids
        self.tokenized = tokenized
        self.k1 = k1
        self.b = b
        self._bm25 = BM25Okapi(tokenized, k1=k1, b=b) if tokenized else None

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "SparseIndex":
        return cls(
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            tokenized=[tokenize_zh(index_text(chunk)) for chunk in chunks],
        )

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        tokens = tokenize_zh(query)
        if not tokens or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokens)
        query_terms = set(tokens)
        hits = [
            (chunk_id, float(score))
            for chunk_id, doc_tokens, score in zip(self.chunk_ids, self.tokenized, scores)
            if query_terms & set(doc_tokens)
        ]
        hits.sort(key=lambda pair: (-pair[1], pair[0]))
        return hits[:top_k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "chunk_ids": self.chunk_ids,
                    "tokenized": self.tokenized,
                    "k1": self.k1,
                    "b": self.b,
                },
                handle,
            )

    @classmethod
    def load(cls, path: Path) -> "SparseIndex":
        with path.open("rb") as handle:
            data = pickle.load(handle)
        return cls(
            chunk_ids=data["chunk_ids"],
            tokenized=data["tokenized"],
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
        )

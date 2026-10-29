from dataclasses import dataclass

from parking_kb.index.dense import DenseIndex, Embedder
from parking_kb.index.sparse import SparseIndex
from parking_kb.index.store import ChunkStore
from parking_kb.retrieve.filters import apply_status_filter
from parking_kb.retrieve.query import strip_document_prefix
from parking_kb.retrieve.rerank import Reranker, rerank
from parking_kb.types import Candidate


@dataclass(frozen=True)
class RetrievalConfig:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fused_top_k: int = 20
    rerank_top_k: int = 8
    rrf_k: int = 60
    use_dense: bool = True
    use_sparse: bool = True
    use_rerank: bool = True
    use_status_filter: bool = True
    use_query_rewrite: bool = True


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        dense: DenseIndex,
        sparse: SparseIndex,
        store: ChunkStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ):
        self.dense = dense
        self.sparse = sparse
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self._titles: list[str] | None = None

    def _document_titles(self) -> list[str]:
        if self._titles is None:
            self._titles = [doc.title for doc in self.store.list_documents()]
        return self._titles

    def retrieve(
        self, query: str, config: RetrievalConfig = RetrievalConfig()
    ) -> list[Candidate]:
        # 检索查询去掉开头的文档限定语；状态过滤仍看原话（标题里的年份是版本线索）。
        search_query = (
            strip_document_prefix(query, self._document_titles())
            if config.use_query_rewrite
            else query
        )
        dense_hits = (
            self.dense.search(search_query, self.embedder, config.dense_top_k)
            if config.use_dense
            else []
        )
        sparse_hits = (
            self.sparse.search(search_query, config.sparse_top_k)
            if config.use_sparse
            else []
        )
        dense_rank = {
            chunk_id: rank for rank, (chunk_id, _) in enumerate(dense_hits, start=1)
        }
        sparse_rank = {
            chunk_id: rank for rank, (chunk_id, _) in enumerate(sparse_hits, start=1)
        }

        rankings = [
            [chunk_id for chunk_id, _ in hits]
            for hits in (dense_hits, sparse_hits)
            if hits
        ]
        fused = rrf_fuse(rankings, config.rrf_k)[: config.fused_top_k]

        chunks = {
            item.chunk_id: item
            for item in self.store.get_chunks([chunk_id for chunk_id, _ in fused])
        }
        candidates = [
            Candidate(
                chunk=chunks[chunk_id],
                score=score,
                dense_rank=dense_rank.get(chunk_id),
                sparse_rank=sparse_rank.get(chunk_id),
            )
            for chunk_id, score in fused
            if chunk_id in chunks
        ]

        if config.use_status_filter:
            candidates = apply_status_filter(candidates, query, self.store)

        if config.use_rerank and self.reranker is not None:
            return rerank(
                candidates, search_query, self.reranker, config.rerank_top_k
            )
        return candidates[: config.rerank_top_k]

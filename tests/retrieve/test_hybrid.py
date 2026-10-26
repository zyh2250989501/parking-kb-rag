from pathlib import Path

from parking_kb.index.store import ChunkStore
from parking_kb.retrieve.hybrid import HybridRetriever, RetrievalConfig, rrf_fuse
from tests.factories import chunk, meta


class FakeDense:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, query, embedder, top_k=20):
        self.queries.append(query)
        return self.hits[:top_k]


class FakeSparse:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, query, top_k=20):
        self.queries.append(query)
        return self.hits[:top_k]


class FakeEmbedder:
    def encode(self, texts):
        return [[0.0, 0.0] for _ in texts]


class FakeReranker:
    def score(self, query, texts):
        return [10.0 if "收费标准" in text else 0.0 for text in texts]


def make_retriever(dense_hits, sparse_hits, chunks=None, reranker=None):
    chunks = chunks or [
        chunk("c1", "d1", "停车收费标准"),
        chunk("c2", "d2", "道闸设备维护"),
    ]
    store = ChunkStore(Path(":memory:"))
    docs = {item.meta.doc_id: item.meta for item in chunks}
    store.upsert_documents(list(docs.values()))
    store.upsert_chunks(chunks)
    return HybridRetriever(
        FakeDense(dense_hits), FakeSparse(sparse_hits), store, FakeEmbedder(), reranker
    )


def test_rrf_prefers_chunk_in_both_lists():
    scores = dict(rrf_fuse([["a", "b"], ["b", "c"]], k=60))
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]


def test_dense_only_config_skips_sparse():
    retriever = make_retriever(dense_hits=[("c1", 0.9)], sparse_hits=[("c2", 0.8)])
    got = retriever.retrieve(
        "q",
        RetrievalConfig(
            use_sparse=False, use_rerank=False, use_status_filter=False
        ),
    )
    assert got[0].chunk.chunk_id == "c1"


def test_sparse_only_config_skips_dense():
    retriever = make_retriever(dense_hits=[("c1", 0.9)], sparse_hits=[("c2", 0.8)])
    got = retriever.retrieve(
        "q",
        RetrievalConfig(
            use_dense=False, use_rerank=False, use_status_filter=False
        ),
    )
    assert [candidate.chunk.chunk_id for candidate in got] == ["c2"]


def test_candidates_record_dense_and_sparse_ranks():
    retriever = make_retriever(dense_hits=[("c1", 0.9)], sparse_hits=[("c2", 0.8)])
    got = retriever.retrieve(
        "q",
        RetrievalConfig(use_rerank=False, use_status_filter=False),
    )
    by_id = {candidate.chunk.chunk_id: candidate for candidate in got}
    assert (by_id["c1"].dense_rank, by_id["c1"].sparse_rank) == (1, None)
    assert (by_id["c2"].dense_rank, by_id["c2"].sparse_rank) == (None, 1)


def test_fused_top_k_truncates_candidates():
    chunks = [chunk("c1"), chunk("c2"), chunk("c3")]
    retriever = make_retriever(
        dense_hits=[("c1", 0.9), ("c2", 0.8), ("c3", 0.7)],
        sparse_hits=[],
        chunks=chunks,
    )
    got = retriever.retrieve(
        "q",
        RetrievalConfig(
            fused_top_k=2, use_rerank=False, use_status_filter=False
        ),
    )
    assert [candidate.chunk.chunk_id for candidate in got] == ["c1", "c2"]


def test_rerank_is_applied_when_enabled():
    chunks = [
        chunk("c1", "d1", "设备维护"),
        chunk("c2", "d2", "停车收费标准"),
    ]
    retriever = make_retriever(
        dense_hits=[("c1", 0.9), ("c2", 0.8)],
        sparse_hits=[],
        chunks=chunks,
        reranker=FakeReranker(),
    )
    got = retriever.retrieve(
        "收费", RetrievalConfig(use_status_filter=False, rerank_top_k=1)
    )
    assert [candidate.chunk.chunk_id for candidate in got] == ["c2"]


def test_status_filter_is_applied_by_default():
    chunks = [
        chunk("c1", "d1", "收费", meta=meta(doc_id="d1", status="现行")),
        chunk("c2", "d2", "收费", meta=meta(doc_id="d2", status="失效")),
    ]
    retriever = make_retriever(
        dense_hits=[("c1", 0.9), ("c2", 0.8)], sparse_hits=[], chunks=chunks
    )
    got = retriever.retrieve(
        "收费", RetrievalConfig(use_rerank=False)
    )
    assert [candidate.chunk.chunk_id for candidate in got] == ["c1"]


def test_empty_hits_return_no_candidates():
    retriever = make_retriever(dense_hits=[], sparse_hits=[])
    assert retriever.retrieve("q") == []

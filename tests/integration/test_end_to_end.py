import json
from pathlib import Path

from parking_kb.generate.schema import AnswerGenerator
from parking_kb.index.dense import DenseIndex
from parking_kb.index.sparse import SparseIndex
from parking_kb.index.store import ChunkStore, load_chunks_jsonl
from parking_kb.ingest.pipeline import run_ingest
from parking_kb.retrieve.hybrid import HybridRetriever, RetrievalConfig
from parking_kb.types import DocumentMeta
from tests.fixtures.builders import make_fixture_corpus


class KeywordEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] if "收费" in text else [0.0, 1.0] for text in texts]


class FakeLLM:
    def complete(self, messages, temperature=0.0):
        return (
            '{"answer":"按小时收费[1]","citations":[1],'
            '"abstained":false,"missing":[]}'
        )


def load_documents(path: Path) -> list[DocumentMeta]:
    with path.open(encoding="utf-8") as handle:
        return [
            DocumentMeta(**json.loads(line)) for line in handle if line.strip()
        ]


def test_ingest_retrieve_and_answer_with_fake_models(tmp_path):
    corpus, data = make_fixture_corpus(tmp_path)
    report = run_ingest(corpus, data)
    assert report.succeeded == 2
    assert report.failed == 1
    assert report.chunk_count > 0

    store = ChunkStore(data / "kb.sqlite")
    store.upsert_documents(load_documents(data / "documents.jsonl"))
    chunks = load_chunks_jsonl(data / "chunks.jsonl")
    store.upsert_chunks(chunks)

    embedder = KeywordEmbedder()
    dense = DenseIndex(tmp_path / "qdrant")
    dense.build(chunks, embedder)
    retriever = HybridRetriever(
        dense, SparseIndex.build(chunks), store, embedder, reranker=None
    )

    candidates = retriever.retrieve(
        "停车怎么收费",
        RetrievalConfig(use_rerank=False, use_status_filter=False),
    )
    assert candidates
    top = candidates[0]
    assert store.get_document(top.chunk.doc_id) is not None

    answer = AnswerGenerator(FakeLLM()).generate("停车怎么收费", candidates[:3])
    assert answer.abstained is False
    assert answer.citations[0].chunk_id == candidates[0].chunk.chunk_id
    assert "收费" in candidates[0].chunk.text or "收费" in candidates[1].chunk.text

    dense.close()
    store.close()

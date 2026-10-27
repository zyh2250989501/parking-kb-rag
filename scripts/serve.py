import argparse
import json
from pathlib import Path

import uvicorn

from parking_kb.api.main import ChatService, create_app
from parking_kb.config import load_settings
from parking_kb.generate.client import CachedProvider, OpenAICompatibleProvider
from parking_kb.generate.schema import AnswerGenerator
from parking_kb.index.dense import COLLECTION, BgeM3Embedder, DenseIndex
from parking_kb.index.sparse import SparseIndex
from parking_kb.index.store import ChunkStore, load_chunks_jsonl
from parking_kb.retrieve.hybrid import HybridRetriever
from parking_kb.retrieve.rerank import BgeReranker
from parking_kb.types import DocumentMeta


def load_documents(path: Path) -> list[DocumentMeta]:
    with path.open(encoding="utf-8") as handle:
        return [
            DocumentMeta(**json.loads(line))
            for line in handle
            if line.strip()
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the RAG chat demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument(
        "--documents", type=Path, default=Path("data/documents.jsonl")
    )
    parser.add_argument("--db", type=Path, default=Path("data/kb.sqlite"))
    parser.add_argument("--dense-dir", type=Path, default=Path("data/qdrant"))
    parser.add_argument("--bm25", type=Path, default=Path("data/bm25.pkl"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    env_file = args.env if args.env.exists() else None
    settings = load_settings(env_file)
    device = args.device or settings.device

    chunks = load_chunks_jsonl(args.chunks)
    store = ChunkStore(args.db)
    store.upsert_documents(load_documents(args.documents))
    store.upsert_chunks(chunks)

    embedder = BgeM3Embedder(settings.embedding_model, device=device)
    dense = DenseIndex(args.dense_dir)
    if args.rebuild_index or not dense.client.collection_exists(COLLECTION):
        dense.build(chunks, embedder)
    if args.rebuild_index or not args.bm25.exists():
        sparse = SparseIndex.build(chunks)
        sparse.save(args.bm25)
    else:
        sparse = SparseIndex.load(args.bm25)

    reranker = BgeReranker(settings.reranker_model, device=device)
    generator = AnswerGenerator(
        CachedProvider(
            OpenAICompatibleProvider(
                base_url=settings.deepseek_base_url,
                api_key=settings.deepseek_api_key,
                model=settings.deepseek_model,
            ),
            settings.data_dir / "llm_cache",
        )
    )
    service = ChatService(
        HybridRetriever(dense, sparse, store, embedder, reranker),
        generator,
        store,
    )
    app = create_app(service)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        dense.close()
        store.close()


if __name__ == "__main__":
    main()

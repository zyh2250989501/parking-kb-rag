import argparse
from pathlib import Path

from parking_kb.config import load_settings
from parking_kb.index.dense import COLLECTION, BgeM3Embedder, DenseIndex
from parking_kb.index.sparse import SparseIndex
from parking_kb.index.store import load_chunks_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="重建稠密与稀疏索引（入库或生成检索描述之后跑一次）。"
    )
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--dense-dir", type=Path, default=Path("data/qdrant"))
    parser.add_argument("--bm25", type=Path, default=Path("data/bm25.pkl"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    settings = load_settings(args.env if args.env.exists() else None)
    chunks = load_chunks_jsonl(args.chunks)
    described = sum(1 for chunk in chunks if chunk.search_text)
    print(f"chunks={len(chunks)} 带检索描述={described}")

    embedder = BgeM3Embedder(settings.embedding_model, device=args.device or settings.device)
    dense = DenseIndex(args.dense_dir)
    try:
        dense.build(chunks, embedder)
        print(f"稠密索引已重建：{COLLECTION}")
    finally:
        dense.close()

    sparse = SparseIndex.build(chunks)
    sparse.save(args.bm25)
    print(f"稀疏索引已重建：{args.bm25}")


if __name__ == "__main__":
    main()

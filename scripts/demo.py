"""演示验收：把 10 个真实业务问题在当前链路上跑一遍，结果落盘供人工判定。

脚本只负责跑和存。答案对不对、引用撑不撑得住，是人看的——自动评估用 judge，
演示这套题的用处恰恰是给人读一遍。
"""

import argparse
import json
from pathlib import Path

from parking_kb.api.main import ChatService
from parking_kb.config import load_settings
from parking_kb.generate.client import CachedProvider, OpenAICompatibleProvider
from parking_kb.generate.schema import AnswerGenerator
from parking_kb.index.dense import COLLECTION, BgeM3Embedder, DenseIndex
from parking_kb.index.sparse import SparseIndex
from parking_kb.index.store import ChunkStore, load_chunks_jsonl
from parking_kb.retrieve.hybrid import HybridRetriever
from parking_kb.retrieve.rerank import BgeReranker
from parking_kb.types import DocumentMeta

QUESTIONS = [
    "湖南办法规定哪些车辆免收停车服务费",
    "停车场应在何处公示收费标准",
    "湖北与湖南对免费停放时长是否一致",
    "客服中心接到投诉的处理流程",
    "大华系统如何统计某车场每日收入",
    "停车场车辆丢失，经营者是否赔偿",
    "大华系统如何添加操作员账号",
    "运营管理规范对夜间照明的要求",
    "北京市条例对路侧电子收费的规定",
    "停车场充电桩数量配比要求",
]


def load_documents(path: Path) -> list[DocumentMeta]:
    with path.open(encoding="utf-8") as handle:
        return [DocumentMeta(**json.loads(line)) for line in handle if line.strip()]


def build_service(args) -> ChatService:
    settings = load_settings(args.env if args.env.exists() else None)
    device = args.device or settings.device

    chunks = load_chunks_jsonl(args.chunks)
    store = ChunkStore(args.db)
    store.upsert_documents(load_documents(args.documents))
    store.upsert_chunks(chunks)

    embedder = BgeM3Embedder(settings.embedding_model, device=device)
    dense = DenseIndex(args.dense_dir)
    if args.rebuild_index or not dense.client.collection_exists(COLLECTION):
        dense.build(chunks, embedder)
    sparse = (
        SparseIndex.build(chunks)
        if args.rebuild_index or not args.bm25.exists()
        else SparseIndex.load(args.bm25)
    )
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
    return ChatService(
        HybridRetriever(dense, sparse, store, embedder, reranker), generator, store
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="跑一遍 10 题演示，结果落盘。")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--documents", type=Path, default=Path("data/documents.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("data/kb.sqlite"))
    parser.add_argument("--dense-dir", type=Path, default=Path("data/qdrant"))
    parser.add_argument("--bm25", type=Path, default=Path("data/bm25.pkl"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports/demo_results.jsonl"))
    args = parser.parse_args()

    service = build_service(args)
    rows = []
    for number, question in enumerate(QUESTIONS, start=1):
        response = service.chat(question)
        rows.append({"number": number, "question": question, **response.model_dump()})
        verdict = "拒答" if response.abstained else "作答"
        print(
            f"[{number}] {verdict} {response.latency_ms} ms 引用 {len(response.citations)} 条\n"
            f"    {question}\n    {response.answer[:120]}\n"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"结果已写入 {args.out}")


if __name__ == "__main__":
    main()

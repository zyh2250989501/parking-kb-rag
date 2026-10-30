"""重排与词面名次融合，能不能救回被重排压掉的强命中。

Q8「运营管理规范对夜间照明的要求」：5.1.5「车辆能够到达的区域应有照明设施」
在 BM25 里排第 9、进了融合池，重排之后掉到第 14，top-8 里一条照明条款都没有。
照度条款（广东指南 5.1.3，30 lux）同样在池内、排第 10，也没进。

候选池里明明有，是重排把它挤出去的。这里量一件事：把重排分数与融合名次
按 w 混合，40 道可答题的 Recall@5 / MRR@10 / nDCG@5 怎么动。

  score = norm(rerank) + w * norm(rrf)

w=0 就是当前线上行为。纯离线，不调用任何 LLM。
"""

import json
from pathlib import Path
from statistics import mean

from parking_kb.config import load_settings
from parking_kb.eval.build_set import load_eval_set, resolve_gold_chunks
from parking_kb.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from parking_kb.eval.schema import EvalQuestion
from parking_kb.index.dense import BgeM3Embedder, DenseIndex
from parking_kb.index.search_text import index_text
from parking_kb.index.sparse import SparseIndex
from parking_kb.index.store import ChunkStore, load_chunks_jsonl
from parking_kb.retrieve.filters import apply_status_filter
from parking_kb.retrieve.hybrid import RetrievalConfig, rrf_fuse
from parking_kb.retrieve.query import strip_document_prefix
from parking_kb.retrieve.rerank import BgeReranker
from parking_kb.types import DocumentMeta, Candidate

ROOT = Path(__file__).resolve().parents[2]
CONFIG = RetrievalConfig()
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5)
TOP_K = 8
CURVE_KS = (5, 8, 10, 12, 14, 16, 18, 20)

# Q8 的目标片段：运营管理服务规范 5.1.5
Q8_QUERY = "运营管理规范对夜间照明的要求"
Q8_TARGET = "6ece808a56dd35ec"


def _norm(values: list[float]) -> list[float]:
    top = max(values) if values else 0.0
    return [value / top for value in values] if top > 0 else values


def load_documents(path: Path) -> list[DocumentMeta]:
    with path.open(encoding="utf-8") as handle:
        return [DocumentMeta(**json.loads(line)) for line in handle if line.strip()]


def load_questions(path: Path) -> list[EvalQuestion]:
    rows = load_eval_set(path)
    return [row for row in rows if row.question_type != "unanswerable"]


def pool_for(retriever, query: str) -> tuple[list[Candidate], str]:
    """与 HybridRetriever.retrieve 同一条路径，但停在重排之前，保留融合名次。

    连查询归一化一起复现——跳过它会让表格题的召回对不上评估报告。
    """
    search_query = strip_document_prefix(query, retriever._document_titles())
    dense_hits = retriever.dense.search(search_query, retriever.embedder, CONFIG.dense_top_k)
    sparse_hits = retriever.sparse.search(search_query, CONFIG.sparse_top_k)
    fused = rrf_fuse(
        [[chunk_id for chunk_id, _ in hits] for hits in (dense_hits, sparse_hits)],
        CONFIG.rrf_k,
    )[: CONFIG.fused_top_k]
    pieces = {
        item.chunk_id: item
        for item in retriever.store.get_chunks([chunk_id for chunk_id, _ in fused])
    }
    candidates = [
        Candidate(chunk=pieces[chunk_id], score=score)
        for chunk_id, score in fused
        if chunk_id in pieces
    ]
    return apply_status_filter(candidates, query, retriever.store), search_query


def rank_with(candidates: list[Candidate], weights: tuple[float, ...], reranker,
              query: str) -> dict[float, list[str]]:
    scores = reranker.score(query, [index_text(c.chunk) for c in candidates])
    rrf = _norm([c.score for c in candidates])
    rerank = _norm(scores)
    out = {}
    for w in weights:
        blended = [r + w * b for r, b in zip(rerank, rrf)]
        order = sorted(range(len(candidates)), key=lambda i: blended[i], reverse=True)
        out[w] = [candidates[i].chunk.chunk_id for i in order]
    return out


def main() -> None:
    settings = load_settings(ROOT / ".env")
    chunks = load_chunks_jsonl(ROOT / "data/chunks.jsonl")
    store = ChunkStore(ROOT / "data/kb.sqlite")
    store.upsert_documents(load_documents(ROOT / "data/documents.jsonl"))
    store.upsert_chunks(chunks)

    from parking_kb.retrieve.hybrid import HybridRetriever

    retriever = HybridRetriever(
        DenseIndex(ROOT / "data/qdrant"),
        SparseIndex.load(ROOT / "data/bm25.pkl"),
        store,
        BgeM3Embedder(settings.embedding_model, device="cpu"),
        None,
    )
    reranker = BgeReranker(settings.reranker_model, device="cpu")

    questions = load_questions(ROOT / "eval/questions.jsonl")
    per_weight = {w: {"recall@5": [], "mrr@10": [], "ndcg@5": []} for w in WEIGHTS}
    curve = {k: [] for k in CURVE_KS}

    for number, question in enumerate(questions, start=1):
        gold = set(resolve_gold_chunks(question, chunks))
        if not gold:
            continue
        pool, search_query = pool_for(retriever, question.question)
        ranked = rank_with(pool, WEIGHTS, reranker, search_query)
        for w, ids in ranked.items():
            ids = ids[:TOP_K]  # 生产只把 top-8 交给生成
            per_weight[w]["recall@5"].append(recall_at_k(ids, gold, 5))
            per_weight[w]["mrr@10"].append(mrr_at_k(ids, gold, 10))
            per_weight[w]["ndcg@5"].append(ndcg_at_k(ids, gold, 5))
        # 上下文窗口该开多大：不截断，看第几名才把 gold 全收进来
        full_order = ranked[0.0]
        for k in CURVE_KS:
            curve[k].append(recall_at_k(full_order, gold, k))
        if number % 10 == 0:
            print(f"  ...{number}/{len(questions)}", flush=True)

    print(f"\n{'w':>5} {'Recall@5':>9} {'MRR@10':>8} {'nDCG@5':>8} {'top-8 含 5.1.5':>15}")
    for w in WEIGHTS:
        row = per_weight[w]
        print(
            f"{w:>5.1f} {mean(row['recall@5']):>9.3f} {mean(row['mrr@10']):>8.3f} "
            f"{mean(row['ndcg@5']):>8.3f}"
        )

    print("\nRecall@k 曲线（40 道可答题，重排后不截断）：")
    for k in CURVE_KS:
        print(f"  k={k:>2}  {mean(curve[k]):.3f}")

    # Q8 单题：5.1.5 在各 w 下的名次
    q8_pool, q8_query = pool_for(retriever, Q8_QUERY)
    q8 = rank_with(q8_pool, WEIGHTS, reranker, q8_query)
    print("\nQ8 5.1.5 名次：")
    for w, ids in q8.items():
        rank = ids.index(Q8_TARGET) + 1 if Q8_TARGET in ids else None
        print(f"  w={w:.1f}  rank={rank}  top8={'进' if rank and rank <= TOP_K else '没进'}")


if __name__ == "__main__":
    main()

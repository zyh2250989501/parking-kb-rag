import argparse
import json
from pathlib import Path

from parking_kb.config import load_settings
from parking_kb.eval.build_set import load_eval_set
from parking_kb.eval.judge import LLMJudge
from parking_kb.eval.report import (
    load_judge_verdicts,
    render_eval_report,
    write_judge_verdicts,
)
from parking_kb.eval.run import (
    CONFIGS,
    aggregate_by_config,
    rejudge_records,
    run_eval,
)
from parking_kb.generate.client import (
    CachedProvider,
    OpenAICompatibleProvider,
    RetryingProvider,
)
from parking_kb.generate.schema import AnswerGenerator
from parking_kb.index.dense import COLLECTION, BgeM3Embedder, DenseIndex
from parking_kb.index.sparse import SparseIndex
from parking_kb.index.store import ChunkStore, load_chunks_jsonl
from parking_kb.retrieve.hybrid import HybridRetriever
from parking_kb.retrieve.rerank import BgeReranker
from parking_kb.types import DocumentMeta


def parse_configs(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown configs: {unknown}")
    return names


def build_judge(settings) -> LLMJudge:
    return LLMJudge(
        CachedProvider(
            RetryingProvider(
                OpenAICompatibleProvider(
                    base_url=settings.siliconflow_base_url,
                    api_key=settings.siliconflow_api_key,
                    model=settings.judge_model,
                )
            ),
            settings.data_dir / "judge_cache",
        )
    )


def load_documents(path: Path) -> list[DocumentMeta]:
    with path.open(encoding="utf-8") as handle:
        return [
            DocumentMeta(**json.loads(line))
            for line in handle
            if line.strip()
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG ablation evaluation.")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument(
        "--configs",
        type=parse_configs,
        default=list(CONFIGS),
        help="comma separated config names",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/eval_report.md"))
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument(
        "--documents", type=Path, default=Path("data/documents.jsonl")
    )
    parser.add_argument("--db", type=Path, default=Path("data/kb.sqlite"))
    parser.add_argument("--dense-dir", type=Path, default=Path("data/qdrant"))
    parser.add_argument("--bm25", type=Path, default=Path("data/bm25.pkl"))
    parser.add_argument(
        "--verdicts", type=Path, default=Path("eval/judge_verdicts.jsonl")
    )
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild the report from saved verdicts, offline and without API keys",
    )
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="re-run only the judge on saved verdicts (no retrieval, no generation)",
    )
    args = parser.parse_args()

    if args.report_only:
        records = load_judge_verdicts(args.verdicts)
        if not records:
            raise SystemExit(f"no saved records at {args.verdicts}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            render_eval_report(aggregate_by_config(records), records),
            encoding="utf-8",
        )
        print(f"wrote {args.out} from {len(records)} saved records")
        return

    env_file = args.env if args.env.exists() else None
    settings = load_settings(env_file)
    if not settings.siliconflow_api_key:
        raise SystemExit("SILICONFLOW_API_KEY is required for the LLM judge")
    device = args.device or settings.device

    if args.rejudge:
        records = load_judge_verdicts(args.verdicts)
        if not records:
            raise SystemExit(f"no saved records at {args.verdicts}")
        chunks = load_chunks_jsonl(args.chunks)
        questions = {q.qid: q for q in load_eval_set(args.questions)}
        updated = rejudge_records(
            records,
            questions,
            {chunk.chunk_id: chunk for chunk in chunks},
            build_judge(settings),
        )
        write_judge_verdicts(updated, args.verdicts)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            render_eval_report(aggregate_by_config(updated), updated),
            encoding="utf-8",
        )
        print(f"rejudged {len(updated)} records; wrote {args.verdicts} and {args.out}")
        return

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
            RetryingProvider(
                OpenAICompatibleProvider(
                    base_url=settings.deepseek_base_url,
                    api_key=settings.deepseek_api_key,
                    model=settings.deepseek_model,
                )
            ),
            settings.data_dir / "llm_cache",
        )
    )
    judge = build_judge(settings)
    retriever = HybridRetriever(dense, sparse, store, embedder, reranker)
    questions = load_eval_set(args.questions)

    aggregated = {}
    verdicts: list[dict] = []
    for config_name in args.configs:
        result = run_eval(
            questions, config_name, retriever, generator, judge, chunks
        )
        aggregated[config_name] = result
        verdicts.extend(result["verdicts"])
        print(
            f"{config_name}: recall@5={result['recall@5']:.3f} "
            f"correct={result['answer_correct_rate']:.3f} "
            f"abstention={result['abstention_accuracy']:.3f}"
        )

    write_judge_verdicts(verdicts, args.verdicts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        render_eval_report(aggregated, verdicts), encoding="utf-8"
    )
    print(f"wrote {args.out}")
    dense.close()
    store.close()


if __name__ == "__main__":
    main()

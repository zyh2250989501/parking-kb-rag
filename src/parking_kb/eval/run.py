from collections import Counter
from dataclasses import asdict
from statistics import mean
from time import perf_counter

from parking_kb.eval.build_set import resolve_gold_chunks
from parking_kb.eval.judge import Judge, Verdict
from parking_kb.eval.metrics import (
    abstention_accuracy,
    bootstrap_ci,
    citation_precision,
    citation_recall,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from parking_kb.eval.schema import EvalQuestion
from parking_kb.generate.schema import AnswerGenerator
from parking_kb.retrieve.hybrid import RetrievalConfig
from parking_kb.types import AnswerResult, Chunk

CONFIGS: dict[str, RetrievalConfig] = {
    "dense_only": RetrievalConfig(
        use_sparse=False, use_rerank=False, use_status_filter=False
    ),
    "dense_bm25_rrf": RetrievalConfig(
        use_rerank=False, use_status_filter=False
    ),
    "dense_bm25_rrf_rerank": RetrievalConfig(use_status_filter=False),
    "full": RetrievalConfig(),
}

RETRIEVAL_METRICS = (
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr@10",
    "ndcg@5",
)
CITATION_METRICS = ("citation_precision", "citation_recall")
METRIC_KEYS = RETRIEVAL_METRICS + CITATION_METRICS + (
    "answer_correct_rate",
    "answer_partial_rate",
    "faithful_rate",
    "citation_precise_rate",
    "abstention_accuracy",
)


def _classify_error(
    question: EvalQuestion,
    gold_ids: list[str],
    ranked_ids: list[str],
    answer: AnswerResult,
    verdict: Verdict,
) -> str | None:
    if question.question_type == "unanswerable":
        return None if answer.abstained else "拒答错误"
    if not gold_ids:
        return "评估集问题"
    if not set(ranked_ids) & set(gold_ids):
        return "召回缺失"
    if answer.abstained:
        return "拒答错误"
    if verdict.correct == "wrong":
        return "版本混淆" if question.question_type == "version" else "生成错误"
    if not verdict.faithful:
        return "生成幻觉"
    return None


def _values(results: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for result in results:
        if key in RETRIEVAL_METRICS or key in CITATION_METRICS:
            if not result.get("gold_ids"):
                continue
        value = result.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _flat_metrics(results: list[dict]) -> dict:
    flat: dict = {"count": len(results)}
    for key in METRIC_KEYS:
        values = _values(results, key)
        flat[key] = mean(values) if values else 0.0
    flat["error_types"] = dict(
        Counter(
            result["error_type"]
            for result in results
            if result.get("error_type")
        )
    )
    flat["verdict_counts"] = dict(
        Counter(
            result["verdict"]["correct"]
            for result in results
            if result.get("verdict")
        )
    )
    return flat


def aggregate(results: list[dict]) -> dict:
    flat = _flat_metrics(results)
    by_type = {
        question_type: _flat_metrics(
            [
                result
                for result in results
                if result.get("question_type") == question_type
            ]
        )
        for question_type in sorted(
            {result.get("question_type") for result in results if result.get("question_type")}
        )
    }
    groups = {
        "verified": _flat_metrics(
            [result for result in results if result.get("verified")]
        ),
        "unverified": _flat_metrics(
            [result for result in results if not result.get("verified")]
        ),
        "all": flat,
    }
    return {
        **flat,
        "cis": {
            key: bootstrap_ci(_values(results, key)) for key in METRIC_KEYS
        },
        "by_type": by_type,
        "groups": groups,
    }


def aggregate_by_config(records: list[dict]) -> dict[str, dict]:
    """Rebuild the per-config aggregates from saved run records."""
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["config"], []).append(record)
    aggregated: dict[str, dict] = {}
    for config_name, config_records in grouped.items():
        metrics = aggregate(config_records)
        metrics["config"] = config_name
        aggregated[config_name] = metrics
    return aggregated


def run_eval(
    questions: list[EvalQuestion],
    config_name: str,
    retriever,
    generator: AnswerGenerator,
    judge: Judge,
    chunks: list[Chunk],
) -> dict:
    config = CONFIGS[config_name]
    records: list[dict] = []
    verdict_records: list[dict] = []
    started = perf_counter()

    for question in questions:
        question_started = perf_counter()
        candidates = retriever.retrieve(question.question, config)
        ranked_ids = [candidate.chunk.chunk_id for candidate in candidates]
        gold_ids = resolve_gold_chunks(question, chunks)
        answer = generator.generate(question.question, candidates)
        verdict = judge.judge(
            question, answer, [candidate.chunk for candidate in candidates]
        )
        predicted_citations = [
            citation.chunk_id for citation in answer.citations
        ]
        gold_set = set(gold_ids)
        expected_abstain = question.question_type == "unanswerable"
        error_type = _classify_error(
            question, gold_ids, ranked_ids, answer, verdict
        )
        record = {
            "qid": question.qid,
            "question": question.question,
            "question_type": question.question_type,
            "verified": question.verified,
            "gold_ids": gold_ids,
            "ranked_ids": ranked_ids,
            "answer": answer.answer,
            "abstained": answer.abstained,
            "citation_ids": predicted_citations,
            "verdict": asdict(verdict),
            "recall@1": recall_at_k(ranked_ids, gold_set, 1),
            "recall@5": recall_at_k(ranked_ids, gold_set, 5),
            "recall@10": recall_at_k(ranked_ids, gold_set, 10),
            "mrr@10": mrr_at_k(ranked_ids, gold_set, 10),
            "ndcg@5": ndcg_at_k(ranked_ids, gold_set, 5),
            "citation_precision": citation_precision(
                predicted_citations, gold_set
            ),
            "citation_recall": citation_recall(predicted_citations, gold_set),
            "answer_correct_rate": 1.0 if verdict.correct == "correct" else 0.0,
            "answer_partial_rate": 1.0 if verdict.correct == "partial" else 0.0,
            "faithful_rate": 1.0 if verdict.faithful else 0.0,
            "citation_precise_rate": 1.0 if verdict.citation_precise else 0.0,
            "abstention_accuracy": (
                1.0 if answer.abstained == expected_abstain else 0.0
            ),
            "error_type": error_type,
            "elapsed_ms": round(
                (perf_counter() - question_started) * 1000, 1
            ),
        }
        records.append(record)
        verdict_records.append({"config": config_name, **record})

    result = aggregate(records)
    result["config"] = config_name
    result["results"] = records
    result["verdicts"] = verdict_records
    result["elapsed_ms"] = round((perf_counter() - started) * 1000, 1)
    return result

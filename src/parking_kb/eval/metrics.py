import math
import random
from statistics import mean


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids or k <= 0:
        return 0.0
    hits = set(ranked_ids[:k]) & gold_ids
    return len(hits) / len(gold_ids)


def mrr_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, chunk_id in enumerate(ranked_ids[:k])
        if chunk_id in gold_ids
    )
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg


def citation_precision(predicted: list[str], gold: set[str]) -> float:
    predicted_set = set(predicted)
    if not predicted_set:
        return 0.0
    return len(predicted_set & gold) / len(predicted_set)


def citation_recall(predicted: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(set(predicted) & gold) / len(gold)


def claim_precision(precise_flags: list[bool]) -> float | None:
    """逐条引用的精确率；没有判定结果为 None，避免把「没测」记成 0。"""
    if not precise_flags:
        return None
    return sum(1 for flag in precise_flags if flag) / len(precise_flags)


def abstention_accuracy(predicted: list[bool], gold: list[bool]) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for pred, expected in zip(predicted, gold) if pred == expected)
    return correct / len(gold)


def bootstrap_ci(
    values: list[float], n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = [
        mean(rng.choices(values, k=len(values)))
        for _ in range(n)
    ]
    means.sort()
    low = means[int(0.025 * n)]
    high = means[max(int(0.975 * n) - 1, 0)]
    return (low, high)

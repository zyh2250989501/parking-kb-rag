import math

from parking_kb.eval.metrics import (
    abstention_accuracy,
    bootstrap_ci,
    citation_precision,
    citation_recall,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from parking_kb.eval.schema import EvalQuestion, GoldSpan


def test_recall_and_mrr_hand_computed():
    assert recall_at_k(["a", "b", "c"], {"b"}, 2) == 1.0
    assert recall_at_k(["a", "b", "c"], {"c"}, 2) == 0.0
    assert mrr_at_k(["a", "b"], {"b"}, 10) == 0.5


def test_ndcg_hand_computed():
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0
    assert round(ndcg_at_k(["a", "b"], {"b"}, 2), 4) == round(1 / math.log2(3), 4)


def test_ndcg_discounts_multiple_relevant_items():
    expected = (1 + 1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == expected
    assert round(ndcg_at_k(["a", "b"], {"a", "b"}, 1), 4) == 1.0


def test_mrr_returns_zero_when_no_relevant_hit():
    assert mrr_at_k(["a", "b"], {"c"}, 2) == 0.0


def test_citation_precision_and_recall():
    assert citation_precision(["c1", "c2"], {"c1"}) == 0.5
    assert citation_recall(["c1", "c2"], {"c1", "c3"}) == 0.5
    assert citation_precision([], {"c1"}) == 0.0
    assert citation_recall(["c1"], set()) == 0.0


def test_abstention_accuracy():
    assert abstention_accuracy([True, False, True], [True, False, False]) == 2 / 3
    assert abstention_accuracy([], []) == 0.0


def test_bootstrap_ci_is_deterministic():
    assert bootstrap_ci([1.0, 0.0, 1.0], seed=0) == bootstrap_ci(
        [1.0, 0.0, 1.0], seed=0
    )


def test_bootstrap_ci_brackets_mean():
    values = [1.0, 0.0, 1.0, 0.5, 0.0]
    low, high = bootstrap_ci(values, n=200, seed=1)
    assert low <= sum(values) / len(values) <= high
    assert bootstrap_ci([], n=200) == (0.0, 0.0)
    assert bootstrap_ci([0.7]) == (0.7, 0.7)


def test_eval_question_schema_roundtrip():
    span = GoldSpan("d1", 10, 20, "按小时收费")
    question = EvalQuestion(
        qid="q1",
        question="怎么收费",
        gold_answer="按小时收费",
        gold_span=span,
        question_type="fact",
        difficulty="easy",
        source_docs=["d1"],
    )
    assert question.verified is False
    assert question.gold_span.quote == "按小时收费"

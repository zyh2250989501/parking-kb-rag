from parking_kb.eval.judge import CitationCheck, Verdict
from parking_kb.eval.report import (
    load_judge_verdicts,
    render_eval_report,
    write_judge_verdicts,
)
from parking_kb.eval.run import (
    CONFIGS,
    aggregate,
    aggregate_by_config,
    rejudge_records,
    run_eval,
)
from parking_kb.eval.schema import EvalQuestion, GoldSpan
from parking_kb.types import AnswerResult, Candidate, Citation
from tests.factories import chunk


class FakeRetriever:
    def __init__(self, candidates):
        self.candidates = candidates
        self.configs = []

    def retrieve(self, query, config=None):
        self.configs.append(config)
        return self.candidates


class FakeGenerator:
    def __init__(self, result):
        self.result = result

    def generate(self, question, candidates):
        return self.result


class FakeJudge:
    def __init__(self, verdict):
        self.verdict = verdict
        self.payloads = []

    def judge(self, question, answer, contexts):
        self.payloads.append({"question": question, "answer": answer,
                              "contexts": contexts})
        return self.verdict


def question(
    qid="q1",
    question_type="fact",
    gold_span=None,
    source_docs=None,
):
    span = gold_span if gold_span is not None else GoldSpan("d1", 0, 5, "按小时收费")
    return EvalQuestion(
        qid=qid,
        question="怎么收费",
        gold_answer="按小时收费",
        gold_span=span,
        question_type=question_type,
        difficulty="easy",
        source_docs=source_docs if source_docs is not None else ["d1"],
    )


def candidate(chunk_id="c1"):
    return Candidate(chunk=chunk(chunk_id, "d1", "按小时收费。"), score=1.0)


def answer(abstained=False):
    return AnswerResult(
        answer="按小时收费[1]",
        citations=[
            Citation(1, "c1", "d1", "测试文档", None, "第一章 总则", 1)
        ],
        abstained=abstained,
    )


def fake_retriever():
    return FakeRetriever([candidate("c1")])


def fake_generator():
    return FakeGenerator(answer())


def fake_judge():
    return FakeJudge(Verdict("correct", True, True, "ok"))


def gold_chunk():
    return chunk("c1", "d1", "按小时收费。")


def test_run_eval_records_claim_precision_from_checks():
    verdict = Verdict(
        "correct", True, False, "ok",
        citation_checks=(CitationCheck(1, True, "直接支持"),),
    )
    result = run_eval(
        [question()],
        "dense_only",
        fake_retriever(),
        fake_generator(),
        FakeJudge(verdict),
        [gold_chunk()],
    )
    assert result["results"][0]["claim_precision"] == 1.0
    assert result["claim_precision"] == 1.0


def test_rejudge_rebuilds_answer_numbering_from_ranked_ids():
    record = {
        "qid": "q1",
        "question": "怎么收费",
        "question_type": "fact",
        "verified": True,
        "gold_ids": ["c1"],
        "ranked_ids": ["c0", "c1"],
        "answer": "按小时收费[2]",
        "abstained": False,
        "citation_ids": ["c1"],
        "verdict": {"correct": "correct", "faithful": True,
                    "citation_precise": True, "reason": "ok"},
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "claim_precision": None,
        "answer_correct_rate": 1.0,
        "answer_partial_rate": 0.0,
        "faithful_rate": 1.0,
        "citation_precise_rate": 1.0,
        "error_type": None,
    }
    judge = FakeJudge(
        Verdict("correct", True, True, "ok",
                citation_checks=(CitationCheck(2, False, "只是相邻条款"),))
    )
    out = rejudge_records([record], {"q1": question()},
                          {"c0": chunk("c0"), "c1": gold_chunk()}, judge)
    rebuilt = judge.payloads[0]
    assert [c.n for c in rebuilt["answer"].citations] == [2]
    assert [c.chunk_id for c in rebuilt["contexts"]] == ["c0", "c1"]
    assert out[0]["claim_precision"] == 0.0
    assert out[0]["citation_precision"] == 1.0  # 原有 strict 指标保留
    assert out[0]["verdict"]["citation_checks"][0]["precise"] is False


def test_run_eval_produces_metrics_and_failures():
    result = run_eval(
        [question()],
        "dense_only",
        fake_retriever(),
        fake_generator(),
        fake_judge(),
        [gold_chunk()],
    )
    assert 0.0 <= result["recall@5"] <= 1.0
    assert "error_types" in result


def test_run_eval_uses_named_config():
    retriever = fake_retriever()
    run_eval(
        [question()], "dense_only", retriever, fake_generator(), fake_judge(),
        [gold_chunk()],
    )
    assert retriever.configs[0] == CONFIGS["dense_only"]


def test_run_eval_classifies_retrieval_miss():
    result = run_eval(
        [question()],
        "dense_only",
        FakeRetriever([]),
        fake_generator(),
        fake_judge(),
        [gold_chunk()],
    )
    assert result["recall@5"] == 0.0
    assert result["error_types"] == {"召回缺失": 1}


def test_unanswerable_question_contributes_to_abstention_only():
    result = run_eval(
        [question(question_type="unanswerable", gold_span=GoldSpan("", 0, 0, ""), source_docs=[])],
        "dense_only",
        fake_retriever(),
        FakeGenerator(answer(abstained=True)),
        FakeJudge(Verdict("correct", True, True, "ok")),
        [gold_chunk()],
    )
    assert result["abstention_accuracy"] == 1.0
    # 没有 gold 的题型不进检索均值；没测出来记 None，不记 0
    assert result["recall@5"] is None
    assert result["error_types"] == {}


def test_aggregate_groups_verified_and_unverified():
    results = [
        {
            "qid": "q1",
            "question_type": "fact",
            "verified": True,
            "gold_ids": ["c1"],
            "ranked_ids": ["c1"],
            "recall@1": 1.0,
            "recall@5": 1.0,
            "recall@10": 1.0,
            "mrr@10": 1.0,
            "ndcg@5": 1.0,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "answer_correct_rate": 1.0,
            "answer_partial_rate": 0.0,
            "faithful_rate": 1.0,
            "citation_precise_rate": 1.0,
            "abstention_accuracy": 1.0,
            "error_type": None,
        },
        {
            "qid": "q2",
            "question_type": "version",
            "verified": False,
            "gold_ids": ["c2"],
            "ranked_ids": ["c3"],
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mrr@10": 0.0,
            "ndcg@5": 0.0,
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "answer_correct_rate": 0.0,
            "answer_partial_rate": 1.0,
            "faithful_rate": 0.0,
            "citation_precise_rate": 0.0,
            "abstention_accuracy": 1.0,
            "error_type": "召回缺失",
        },
    ]
    agg = aggregate(results)
    assert agg["count"] == 2
    assert agg["recall@5"] == 0.5
    assert agg["groups"]["verified"]["count"] == 1
    assert agg["groups"]["unverified"]["count"] == 1
    assert agg["by_type"]["version"]["recall@5"] == 0.0
    low, high = agg["cis"]["recall@5"]
    assert low <= 0.5 <= high


def test_report_contains_ablation_table():
    text = render_eval_report(
        {"dense_only": {"recall@5": 0.5}, "full": {"recall@5": 0.9}}, []
    )
    assert "dense_only" in text and "full" in text and "Recall@5" in text


def test_report_shows_strict_and_claim_precision_side_by_side():
    text = render_eval_report(
        {"full": {"citation_precision": 0.521, "claim_precision": 0.9}}, []
    )
    assert "引用精确率(strict)" in text and "引用精确率(claim)" in text
    assert "0.521" in text and "0.900" in text


def test_report_renders_ci_and_failures():
    aggregated = {
        "full": {
            "recall@5": 0.9,
            "cis": {"recall@5": (0.8, 1.0)},
            "by_type": {"fact": {"count": 2, "recall@5": 1.0, "answer_correct_rate": 1.0}},
            "error_types": {"召回缺失": 1},
        }
    }
    verdicts = [
        {
            "config": "full",
            "qid": "q1",
            "error_type": "召回缺失",
            "question": "怎么收费",
            "answer": "资料中未提及",
        }
    ]
    text = render_eval_report(aggregated, verdicts)
    assert "0.800" in text and "1.000" in text
    assert "召回缺失" in text and "怎么收费" in text


def test_judge_verdicts_roundtrip(tmp_path):
    verdicts = [{"qid": "q1", "config": "full", "correct": "correct"}]
    path = tmp_path / "verdicts.jsonl"
    write_judge_verdicts(verdicts, path)
    assert load_judge_verdicts(path) == verdicts


def test_verdict_records_rebuild_the_same_aggregate():
    result = run_eval(
        [question()],
        "dense_only",
        fake_retriever(),
        fake_generator(),
        fake_judge(),
        [gold_chunk()],
    )
    rebuilt = aggregate_by_config(result["verdicts"])
    assert rebuilt["dense_only"]["recall@5"] == result["recall@5"]
    assert rebuilt["dense_only"]["by_type"]["fact"]["count"] == 1
    assert rebuilt["dense_only"]["groups"]["all"]["count"] == 1


def test_offline_report_from_written_verdicts(tmp_path):
    result = run_eval(
        [question()],
        "full",
        fake_retriever(),
        fake_generator(),
        fake_judge(),
        [gold_chunk()],
    )
    path = tmp_path / "verdicts.jsonl"
    write_judge_verdicts(result["verdicts"], path)
    loaded = load_judge_verdicts(path)
    text = render_eval_report(aggregate_by_config(loaded), loaded)
    assert "Recall@5" in text and "full" in text


def test_failure_section_reads_nested_verdict():
    verdicts = [
        {
            "config": "full",
            "qid": "q1",
            "question": "怎么收费",
            "answer": "资料中未提及",
            "error_type": "召回缺失",
            "verdict": {"correct": "wrong", "reason": "上下文未覆盖"},
        }
    ]
    text = render_eval_report({"full": {}}, verdicts)
    assert "wrong" in text and "上下文未覆盖" in text

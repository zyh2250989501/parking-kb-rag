import json

from parking_kb.eval.build_set import (
    build_eval_set,
    load_eval_set,
    resolve_gold_chunks,
    write_eval_set,
)
from parking_kb.eval.schema import EvalQuestion, GoldSpan
from tests.factories import chunk, meta


class FakeJsonLLM:
    def __init__(self, payload):
        if isinstance(payload, str):
            self.payload = payload
        else:
            self.payload = json.dumps(payload, ensure_ascii=False)
        self.calls = []

    def complete(self, messages, temperature=0.0):
        self.calls.append(messages)
        return self.payload


def question_with_span(
    doc_id, quote="按小时收费", char_start=0, char_end=5
):
    return EvalQuestion(
        qid="q1",
        question="q",
        gold_answer="a",
        gold_span=GoldSpan(doc_id, char_start, char_end, quote),
        question_type="fact",
        difficulty="easy",
        source_docs=[doc_id],
    )


def test_build_eval_set_returns_valid_questions():
    llm = FakeJsonLLM(
        {
            "questions": [
                {
                    "question": "收费标准是什么",
                    "answer": "按小时",
                    "quote": "按小时收费",
                    "type": "fact",
                    "difficulty": "easy",
                }
            ]
        }
    )
    qs = build_eval_set([chunk("c1", "d1", "按小时收费。")], llm, {"fact": 1})
    assert len(qs) == 1
    assert qs[0].question == "收费标准是什么"
    assert qs[0].gold_span.quote == "按小时收费"
    assert qs[0].verified is False


def test_resolve_gold_chunks_maps_span():
    q = question_with_span("d1", quote="按小时收费", char_start=0, char_end=5)
    assert resolve_gold_chunks(q, [chunk("c1", "d1", "按小时收费。")]) == ["c1"]


def test_build_eval_set_skips_unresolvable_gold():
    llm = FakeJsonLLM(
        {
            "questions": [
                {
                    "question": "x",
                    "answer": "y",
                    "quote": "不存在的句子",
                    "type": "fact",
                    "difficulty": "easy",
                }
            ]
        }
    )
    assert build_eval_set([chunk("c1", "d1", "按小时收费。")], llm, {"fact": 1}) == []


def test_unanswerable_question_has_empty_span():
    llm = FakeJsonLLM(
        {
            "questions": [
                {
                    "question": "明年北京牌照价格是多少",
                    "answer": "语料未提及",
                    "type": "unanswerable",
                    "difficulty": "medium",
                }
            ]
        }
    )
    qs = build_eval_set([chunk("c1", "d1", "按小时收费。")], llm, {"unanswerable": 1})
    assert len(qs) == 1
    assert qs[0].gold_span == GoldSpan("", 0, 0, "")
    assert resolve_gold_chunks(qs[0], [chunk("c1", "d1", "按小时收费。")]) == []


def test_cross_doc_questions_record_both_source_docs():
    llm = FakeJsonLLM(
        {
            "questions": [
                {
                    "question": "两份文件对收费的规定有何不同",
                    "answer": "a",
                    "quote": "按小时收费",
                    "type": "cross_doc",
                    "difficulty": "hard",
                }
            ]
        }
    )
    chunks = [
        chunk("c1", "d1", "按小时收费。"),
        chunk("c2", "d2", "道闸设备维护。"),
    ]
    qs = build_eval_set(chunks, llm, {"cross_doc": 1})
    assert len(qs) == 1
    assert set(qs[0].source_docs) == {"d1", "d2"}


def test_version_questions_require_a_same_doc_no_pair():
    llm = FakeJsonLLM({"questions": []})
    chunks = [
        chunk("c1", "d1", "按小时收费。"),
        chunk("c2", "d2", "道闸设备维护。"),
    ]
    assert build_eval_set(chunks, llm, {"version": 1}) == []
    assert llm.calls == []


def test_version_questions_use_same_doc_no_pair():
    old = chunk(
        "c1",
        "d1",
        "旧版收费为每小时两元。",
        meta=meta(doc_id="d1", doc_no="DB1", status="失效"),
    )
    new = chunk(
        "c2",
        "d2",
        "新版收费为每小时三元。",
        meta=meta(doc_id="d2", doc_no="DB1", status="现行"),
    )
    llm = FakeJsonLLM(
        {
            "questions": [
                {
                    "question": "现行标准每小时收多少",
                    "answer": "三元",
                    "quote": "新版收费为每小时三元",
                    "type": "version",
                    "difficulty": "hard",
                }
            ]
        }
    )
    questions = build_eval_set([old, new], llm, {"version": 1})
    assert len(questions) == 1
    assert questions[0].question_type == "version"
    assert questions[0].gold_span.doc_id == "d2"


def test_malformed_llm_output_returns_no_questions():
    llm = FakeJsonLLM("not json")
    assert build_eval_set([chunk("c1", "d1", "按小时收费。")], llm, {"fact": 1}) == []


def test_eval_set_roundtrip(tmp_path):
    questions = [question_with_span("d1")]
    path = tmp_path / "nested" / "questions.jsonl"
    write_eval_set(questions, path)
    assert load_eval_set(path) == questions

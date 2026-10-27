from parking_kb.eval.schema import EvalQuestion, GoldSpan
from parking_kb.eval.screen import screen_unanswerable
from tests.factories import chunk


class FakeVerifierLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, temperature=0.0):
        self.calls.append(messages)
        return self.replies.pop(0)


def unanswerable(qid, question):
    return EvalQuestion(
        qid=qid,
        question=question,
        gold_answer="语料未提及",
        gold_span=GoldSpan("", 0, 0, ""),
        question_type="unanswerable",
        difficulty="medium",
        source_docs=[],
    )


def test_drops_question_the_retrieved_context_can_answer():
    chunks = [chunk("c1", "d1", "停车收费每小时两元。")]
    kept, dropped = screen_unanswerable(
        [unanswerable("u1", "停车怎么收费？")],
        chunks,
        FakeVerifierLLM(['{"answerable": true}']),
    )
    assert kept == []
    assert [question.qid for question in dropped] == ["u1"]


def test_keeps_question_without_supporting_context():
    chunks = [chunk("c1", "d1", "停车收费每小时两元。")]
    kept, dropped = screen_unanswerable(
        [unanswerable("u1", "上海停车场条例第三十七条怎么规定？")],
        chunks,
        FakeVerifierLLM(['{"answerable": false}']),
    )
    assert [question.qid for question in kept] == ["u1"]
    assert dropped == []


def test_unparseable_verdict_keeps_question():
    chunks = [chunk("c1", "d1", "停车收费每小时两元。")]
    llm = FakeVerifierLLM(["不知道"])
    kept, dropped = screen_unanswerable(
        [unanswerable("u1", "上海停车场条例第三十七条怎么规定？")], chunks, llm
    )
    assert [question.qid for question in kept] == ["u1"]
    assert dropped == []


def test_limit_truncates_surviving_questions():
    chunks = [chunk("c1", "d1", "停车收费每小时两元。")]
    llm = FakeVerifierLLM(['{"answerable": false}'] * 3)
    kept, _ = screen_unanswerable(
        [
            unanswerable("u1", "上海的规定是什么？"),
            unanswerable("u2", "北京的规定是什么？"),
            unanswerable("u3", "深圳的规定是什么？"),
        ],
        chunks,
        llm,
        limit=2,
    )
    assert [question.qid for question in kept] == ["u1", "u2"]

from parking_kb.eval.judge import LLMJudge, Verdict, load_verdicts, save_verdicts
from parking_kb.eval.schema import EvalQuestion, GoldSpan
from parking_kb.types import AnswerResult
from tests.factories import chunk


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, messages, temperature=0.0):
        self.calls.append(messages)
        return self.reply


def question():
    return EvalQuestion(
        qid="q1",
        question="怎么收费",
        gold_answer="按小时收费",
        gold_span=GoldSpan("d1", 0, 5, "按小时收费"),
        question_type="fact",
        difficulty="easy",
        source_docs=["d1"],
    )


def answer(text):
    return AnswerResult(answer=text, citations=[], abstained=False)


def test_judge_parses_verdict_json():
    llm = FakeLLM(
        '{"correct":"correct","faithful":true,"citation_precise":true,'
        '"reason":"答案与出处一致"}'
    )
    v = LLMJudge(llm).judge(question(), answer("按小时收费[1]"), [chunk("c1")])
    assert v.correct == "correct" and v.faithful is True


def test_judge_marks_invalid_json_as_wrong():
    v = LLMJudge(FakeLLM("not json")).judge(question(), answer("x"), [])
    assert v.correct == "wrong"
    assert "解析失败" in v.reason


def test_judge_rejects_unknown_correct_value():
    llm = FakeLLM(
        '{"correct":"maybe","faithful":true,"citation_precise":true,"reason":"x"}'
    )
    v = LLMJudge(llm).judge(question(), answer("x"), [])
    assert v.correct == "wrong"


def test_judge_prompt_contains_gold_answer_and_context():
    llm = FakeLLM(
        '{"correct":"partial","faithful":false,"citation_precise":false,"reason":"x"}'
    )
    LLMJudge(llm).judge(question(), answer("x"), [chunk("c1", text="按小时收费")])
    prompt = "\n".join(message["content"] for message in llm.calls[0])
    assert "按小时收费" in prompt
    assert "怎么收费" in prompt


def test_verdicts_roundtrip(tmp_path):
    verdicts = {
        "q1|full": Verdict("correct", True, False, "ok"),
        "q2|dense_only": Verdict("wrong", False, False, "无关"),
    }
    path = tmp_path / "verdicts.jsonl"
    save_verdicts(verdicts, path)
    assert load_verdicts(path) == verdicts

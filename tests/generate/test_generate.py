from parking_kb.generate.client import CachedProvider
from parking_kb.generate.prompt import build_context, parse_answer_json
from parking_kb.generate.schema import AnswerGenerator
from tests.factories import cand


class FakeLLM:
    def __init__(self, replies):
        self.replies = [replies] if isinstance(replies, str) else list(replies)
        self.calls = []

    def complete(self, messages, temperature=0.0):
        self.calls.append((messages, temperature))
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0]


def test_generate_maps_numbered_citations_to_chunk_ids():
    llm = FakeLLM(
        '{"answer":"按小时收费[1]","citations":[1],"abstained":false,"missing":[]}'
    )
    got = AnswerGenerator(llm).generate("怎么收费", [cand("c1", text="按小时收费")])
    assert got.answer == "按小时收费[1]"
    assert got.citations[0].chunk_id == "c1"
    assert got.abstained is False


def test_generate_abstains_when_model_says_so():
    llm = FakeLLM(
        '{"answer":"资料中未提及","citations":[],"abstained":true,'
        '"missing":["具体费率"]}'
    )
    got = AnswerGenerator(llm).generate("明年费率", [cand("c1")])
    assert got.abstained is True
    assert got.missing == ["具体费率"]


def test_context_includes_title_effective_date_and_page():
    ctx = build_context(
        [cand("c1", title="收费办法", effective_date="2026-01-01", page=3)]
    )
    assert "收费办法" in ctx and "2026-01-01" in ctx and "p.3" in ctx


def test_parse_answer_json_accepts_fenced_output():
    raw = '```json\n{"answer":"x[1]","citations":[1],"abstained":false}\n```'
    got = parse_answer_json(raw, [cand("c1")])
    assert got.citations[0].chunk_id == "c1"


def test_parse_answer_json_ignores_out_of_range_citation():
    raw = '{"answer":"x","citations":[9],"abstained":false,"missing":[]}'
    got = parse_answer_json(raw, [cand("c1")])
    assert got.citations == []


def test_malformed_output_retries_once_then_abstains():
    llm = FakeLLM(["not json", "still not json"])
    got = AnswerGenerator(llm).generate("q", [cand("c1")])
    assert got.abstained is True
    assert len(llm.calls) == 2
    assert "still not json" in got.answer


def test_cached_provider_reuses_response(tmp_path):
    inner = FakeLLM("same")
    cached = CachedProvider(inner, tmp_path / "cache")
    first = cached.complete([{"role": "user", "content": "q"}])
    second = cached.complete([{"role": "user", "content": "q"}])
    assert first == second == "same"
    assert len(inner.calls) == 1
    assert len(list((tmp_path / "cache").glob("*"))) == 1


def test_default_prompt_is_v2_with_partial_evidence_rule():
    """v2 把「有没有可引条款」和「答案完不完整」分开。"""
    from parking_kb.generate.schema import DEFAULT_PROMPT

    assert DEFAULT_PROMPT.name == "answer_v2.md"
    text = DEFAULT_PROMPT.read_text(encoding="utf-8")
    assert "没覆盖问题追加的限定条件" in text
    # 提示词里不许出现演示题的字样——那是拿单题调参
    assert "夜间照明" not in text

from parking_kb.retrieve.rerank import rerank
from tests.factories import cand


class FakeReranker:
    def score(self, query, texts):
        return [10.0 if "收费标准" in text else 0.0 for text in texts]


def test_rerank_reorders_and_truncates():
    cands = [cand("c1", text="设备维护"), cand("c2", text="停车收费标准")]
    got = rerank(cands, "收费", FakeReranker(), top_k=1)
    assert [c.chunk.chunk_id for c in got] == ["c2"]


def test_rerank_empty_input():
    assert rerank([], "q", FakeReranker(), top_k=8) == []


def test_rerank_writes_model_scores_onto_candidates():
    got = rerank(
        [cand("c1", text="设备维护", score=99.0), cand("c2", text="停车收费标准")],
        "收费",
        FakeReranker(),
        top_k=2,
    )
    assert [c.score for c in got] == [10.0, 0.0]

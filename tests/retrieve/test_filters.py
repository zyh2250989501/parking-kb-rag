import pytest

from parking_kb.index.store import ChunkStore
from parking_kb.retrieve.filters import apply_status_filter, has_temporal_cue
from tests.factories import cand, meta


@pytest.fixture
def store(tmp_path):
    chunk_store = ChunkStore(tmp_path / "kb.sqlite")
    yield chunk_store
    chunk_store.close()


def add_doc(store, doc_id, doc_no, effective_date, status="现行"):
    store.upsert_documents(
        [
            meta(
                doc_id=doc_id,
                doc_no=doc_no,
                effective_date=effective_date,
                status=status,
            )
        ]
    )


def test_active_and_unknown_kept_inactive_excluded(store):
    cands = [cand("c1", status="现行"), cand("c2", status="失效"), cand("c3", status="未知")]
    got = apply_status_filter(cands, "停车怎么收费", store)
    assert [c.chunk.chunk_id for c in got] == ["c1", "c3"]


def test_unknown_is_deprioritized(store):
    got = apply_status_filter(
        [cand("c1", status="未知"), cand("c2", status="现行")], "收费", store
    )
    assert got[0].chunk.chunk_id == "c2"


def test_unknown_score_is_downweighted(store):
    got = apply_status_filter([cand("c1", status="未知", score=1.0)], "收费", store)
    assert got[0].score == pytest.approx(0.95)


def test_temporal_cue_allows_old_version(store):
    got = apply_status_filter(
        [cand("old", status="失效"), cand("new", status="现行")],
        "2023年的旧标准怎么收费",
        store,
    )
    assert {c.chunk.chunk_id for c in got} == {"old", "new"}


def test_has_temporal_cue_matches_years_and_history_words():
    assert has_temporal_cue("2023年的旧标准怎么收费")
    assert has_temporal_cue("以前的停车规定")
    assert not has_temporal_cue("停车怎么收费")


def test_same_doc_no_keeps_latest_active_version(store):
    add_doc(store, "old", "X〔2020〕1号", "2020-01-01")
    add_doc(store, "new", "X〔2020〕1号", "2025-01-01")
    cands = [
        cand("cold", status="现行", doc_id="old", doc_no="X〔2020〕1号", effective_date="2020-01-01"),
        cand("cnew", status="现行", doc_id="new", doc_no="X〔2020〕1号", effective_date="2025-01-01"),
    ]
    got = apply_status_filter(cands, "停车收费", store)
    assert [c.chunk.doc_id for c in got] == ["new"]


def test_docs_without_doc_no_are_not_grouped(store):
    got = apply_status_filter(
        [cand("c1", doc_id="d1"), cand("c2", doc_id="d2")], "收费", store
    )
    assert {c.chunk.chunk_id for c in got} == {"c1", "c2"}

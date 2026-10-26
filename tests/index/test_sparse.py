from parking_kb.index.sparse import SparseIndex, tokenize_zh
from tests.factories import chunk


def test_bm25_ranks_relevant_chunk_first():
    idx = SparseIndex.build(
        [
            chunk("c1", "d1", "停车收费按车型和停放时长计算"),
            chunk("c2", "d2", "道闸设备每月检查一次"),
        ]
    )
    hits = idx.search("停车收费标准", top_k=2)
    assert hits[0][0] == "c1"


def test_sparse_index_persistence(tmp_path):
    idx = SparseIndex.build([chunk("c1", "d1", "月租车办理流程")])
    idx.save(tmp_path / "bm25.pkl")
    assert SparseIndex.load(tmp_path / "bm25.pkl").search("月租", top_k=1)[0][0] == "c1"


def test_tokenize_keeps_parking_terms_and_drops_punctuation():
    tokens = tokenize_zh("月租车办理，道闸设备检查。")
    assert "月租车" in tokens
    assert "道闸" in tokens
    assert "，" not in tokens
    assert "。" not in tokens
    assert all(token.strip() for token in tokens)


def test_blank_query_returns_no_hits():
    idx = SparseIndex.build([chunk("c1", "d1", "停车收费")])
    assert idx.search("，。 ", top_k=5) == []

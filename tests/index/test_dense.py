import pytest

from parking_kb.index.dense import DenseIndex
from tests.factories import chunk


class KeywordEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            if "月租" in text:
                vectors.append([1.0, 0.5])
            elif "收费" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def test_dense_index_roundtrip(tmp_path):
    idx = DenseIndex(tmp_path / "q")
    idx.build(
        [chunk("c1", "d1", "收费按小时计"), chunk("c2", "d2", "道闸维护")],
        KeywordEmbedder(),
    )
    idx.close()
    idx2 = DenseIndex(tmp_path / "q")
    hits = idx2.search("收费", KeywordEmbedder(), top_k=1)
    assert hits[0][0] == "c1"
    idx2.close()


def test_dense_build_embeds_search_text_when_present(tmp_path):
    """英文原文配了中文检索描述时，向量应按中文描述建，否则中文查询打不中。"""
    idx = DenseIndex(tmp_path / "q")
    idx.build(
        [chunk("c1", "d1", "You can add vehicle owner.", search_text="月租")],
        KeywordEmbedder(),
    )
    hits = idx.search("月租", KeywordEmbedder(), top_k=1)
    assert hits[0][0] == "c1"
    idx.close()


def test_dense_search_is_ranked_by_similarity(tmp_path):
    idx = DenseIndex(tmp_path / "q")
    idx.build(
        [
            chunk("c1", "d1", "收费按小时计"),
            chunk("c2", "d2", "道闸维护"),
            chunk("c3", "d3", "收费与月租"),
        ],
        KeywordEmbedder(),
    )
    hits = idx.search("收费", KeywordEmbedder(), top_k=2)
    assert [chunk_id for chunk_id, _ in hits] == ["c1", "c3"]
    assert all(0.0 <= score <= 1.0 for _, score in hits)
    idx.close()


def test_build_replaces_previous_collection(tmp_path):
    idx = DenseIndex(tmp_path / "q")
    idx.build([chunk("old", "d1", "收费")], KeywordEmbedder())
    idx.build([chunk("new", "d2", "道闸")], KeywordEmbedder())
    hits = idx.search("道闸", KeywordEmbedder(), top_k=10)
    assert [chunk_id for chunk_id, _ in hits] == ["new"]
    idx.close()


@pytest.mark.slow
def test_bge_m3_real_model_smoke(tmp_path):
    pytest.importorskip("sentence_transformers")
    from parking_kb.index.dense import BgeM3Embedder

    try:
        embedder = BgeM3Embedder(device="cpu")
        vectors = embedder.encode(["停车收费标准", "道闸设备维护"])
    except Exception as exc:  # model unavailable offline
        pytest.skip(f"bge-m3 unavailable: {exc}")
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])

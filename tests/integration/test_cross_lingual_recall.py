"""跨语言召回回归：中文问题要能在单通道里命中英文手册的具体操作段落。

演示验收里 Q5（每日收入统计）与 Q7（添加操作员账号）两题误拒。
原因是英文手册正文与中文查询没有词面交集：修复前 Q7 的目标 chunk
在 dense 排 36、sparse 完全不命中。给英文 chunk 生成中文检索描述
（`Chunk.search_text`）之后，sparse 能把它排到 14 名、dense 排到 31 名。

**这不等于端到端可答。** 默认融合池 20 条按 RRF 取分，只被单通道命中的
chunk 拿不到两项分数，仍在池外；实测把漏斗放宽到 50/50/40 可以让它进入
最终上下文（两题都进 top-8），但 Recall@5 仍是 0.812、MRR/nDCG 反而略降、
检索耗时 11.4s → 26.8s，不划算，因此没有采纳。本用例只守住
「描述让英文段落变得可检索」这一条，最终上下文的取舍留在报告里。

用例跑真实索引与真实模型，需要 `data/` 已入库、生成描述并重建索引，默认跳过：
`pytest -m slow tests/integration/test_cross_lingual_recall.py`
"""

from pathlib import Path

import pytest

from parking_kb.index.store import load_chunks_jsonl

pytestmark = pytest.mark.slow

DATA = Path("data")

# (问题, 正确段落应包含的英文标题)
CASES = [
    ("大华停车场管理系统如何统计某个车场的每日收入？", "Report Statistics"),
    ("大华系统里怎么给一个车场添加新的操作员账号？", "User Management"),
]


def channel_ranks(question: str, section: str) -> dict[str, int | None]:
    """目标段落分别在 dense / sparse 前 50 里的名次。"""
    pytest.importorskip("sentence_transformers")
    from parking_kb.config import load_settings
    from parking_kb.index.dense import BgeM3Embedder, DenseIndex
    from parking_kb.index.sparse import SparseIndex

    if not (DATA / "qdrant").exists() or not (DATA / "bm25.pkl").exists():
        pytest.skip("索引不存在，先入库并重建索引")

    settings = load_settings(Path(".env") if Path(".env").exists() else None)
    try:
        embedder = BgeM3Embedder(settings.embedding_model, device="cpu")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"模型不可用：{exc}")

    chunks = load_chunks_jsonl(DATA / "chunks.jsonl")
    targets = {
        chunk.chunk_id
        for chunk in chunks
        if chunk.meta.title.startswith("大华")
        and section.lower() in chunk.section_path.lower()
    }
    dense = DenseIndex(DATA / "qdrant")
    sparse = SparseIndex.load(DATA / "bm25.pkl")
    try:
        dense_ids = [cid for cid, _ in dense.search(question, embedder, 50)]
        sparse_ids = [cid for cid, _ in sparse.search(question, 50)]
    finally:
        dense.close()

    def best(ids: list[str]) -> int | None:
        hits = [ids.index(cid) + 1 for cid in targets if cid in ids]
        return min(hits) if hits else None

    return {"dense": best(dense_ids), "sparse": best(sparse_ids)}


@pytest.mark.parametrize("question,section", CASES)
def test_english_section_is_reachable_by_one_channel(question, section):
    ranks = channel_ranks(question, section)
    assert any(rank is not None and rank <= 20 for rank in ranks.values()), (
        f"{question!r} 在 dense/sparse 前 20 都找不到含 {section!r} 的段落，实际排名 {ranks}"
    )


def test_every_english_chunk_has_chinese_description():
    """索引侧的前提：英文 chunk 必须带中文描述，否则检索描述形同虚设。"""
    chunks = [
        c for c in load_chunks_jsonl(DATA / "chunks.jsonl") if c.meta.title.startswith("大华")
    ]
    missing = [chunk.chunk_id for chunk in chunks if not chunk.search_text]
    assert not missing, f"大华手册 {len(missing)}/{len(chunks)} 个 chunk 缺中文描述"

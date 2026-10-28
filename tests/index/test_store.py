import json
import threading
from dataclasses import asdict

from parking_kb.index.store import ChunkStore, load_chunks_jsonl
from tests.factories import chunk, meta


def test_store_roundtrip_and_version_lookup(tmp_path):
    store = ChunkStore(tmp_path / "kb.sqlite")
    store.upsert_documents(
        [
            meta(
                doc_id="d1",
                doc_no="X〔2025〕1号",
                effective_date="2026-01-01",
                status="现行",
            )
        ]
    )
    store.upsert_chunks([chunk("c1", "d1", "收费按小时计")])
    assert store.get_document("d1").doc_no == "X〔2025〕1号"
    assert store.get_chunks(["c1"])[0].text == "收费按小时计"
    assert len(store.find_versions("X〔2025〕1号")) == 1
    store.close()


def test_list_documents_filters_by_status(tmp_path):
    store = ChunkStore(tmp_path / "kb.sqlite")
    store.upsert_documents([meta(doc_id="d1", status="现行"), meta(doc_id="d2", status="失效")])
    assert [d.doc_id for d in store.list_documents(status="现行")] == ["d1"]
    assert len(store.list_documents()) == 2
    store.close()


def test_get_chunks_preserves_requested_order(tmp_path):
    store = ChunkStore(tmp_path / "kb.sqlite")
    store.upsert_documents([meta(doc_id="d1")])
    store.upsert_chunks([chunk("c1", "d1", "一"), chunk("c2", "d1", "二")])
    assert [c.chunk_id for c in store.get_chunks(["c2", "c1"])] == ["c2", "c1"]
    store.close()


def test_store_is_usable_from_another_thread(tmp_path):
    """FastAPI 在线程池里执行同步端点，连接必须能跨线程使用。"""
    store = ChunkStore(tmp_path / "kb.sqlite")
    store.upsert_documents([meta(doc_id="d1")])
    store.upsert_chunks([chunk("c1", "d1", "收费按小时计")])
    result = {}

    def work():
        try:
            result["rows"] = store.get_chunks(["c1"])
            result["docs"] = store.list_documents()
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert "error" not in result, result.get("error")
    assert result["rows"][0].text == "收费按小时计"
    store.close()


def test_load_chunks_jsonl_roundtrip(tmp_path):
    record = asdict(chunk("c1", "d1", "收费按小时计"))
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    loaded = load_chunks_jsonl(path)
    assert loaded[0].chunk_id == "c1"
    assert loaded[0].meta.doc_id == "d1"
    assert loaded[0].meta.status == "现行"

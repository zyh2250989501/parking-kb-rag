from fastapi.testclient import TestClient

from parking_kb.api.main import ChatService, create_app
from parking_kb.types import AnswerResult, Candidate, Citation
from tests.factories import chunk, meta


class FakeRetriever:
    def __init__(self, candidates):
        self.candidates = candidates

    def retrieve(self, query, config=None):
        return self.candidates


class FakeGenerator:
    def __init__(self, result):
        self.result = result

    def generate(self, question, candidates):
        return self.result


class FakeStore:
    def __init__(self, documents=None):
        self.documents = documents or []

    def list_documents(self, status=None):
        return self.documents


def answer(abstained=False):
    return AnswerResult(
        answer="按小时收费[1]" if not abstained else "资料中未提及",
        citations=[] if abstained else [
            Citation(1, "c1", "d1", "收费办法", "2026-01-01", "第二章 收费", 3)
        ],
        abstained=abstained,
    )


def fake_service(abstained=False):
    return ChatService(
        FakeRetriever([Candidate(chunk("c1", "d1", "按小时收费。"), 1.0)]),
        FakeGenerator(answer(abstained)),
        FakeStore([meta(doc_id="d1", title="收费办法")]),
    )


def test_health():
    client = TestClient(create_app(fake_service()))
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_returns_answer_and_citations():
    resp = TestClient(create_app(fake_service())).post(
        "/chat", json={"question": "怎么收费"}
    )
    body = resp.json()
    assert body["answer"]
    assert body["citations"][0]["chunk_id"] == "c1"


def test_chat_citation_includes_source_text():
    body = TestClient(create_app(fake_service())).post(
        "/chat", json={"question": "怎么收费"}
    ).json()
    assert body["citations"][0]["text"] == "按小时收费。"
    assert body["latency_ms"] >= 0


def test_empty_question_rejected():
    resp = TestClient(create_app(fake_service())).post(
        "/chat", json={"question": "  "}
    )
    assert resp.status_code == 422


def test_documents_route_lists_documents():
    body = TestClient(create_app(fake_service())).get("/documents").json()
    assert body[0]["doc_id"] == "d1"


def test_abstained_answer_is_preserved():
    body = TestClient(create_app(fake_service(abstained=True))).post(
        "/chat", json={"question": "明年费率"}
    ).json()
    assert body["abstained"] is True
    assert body["citations"] == []


def test_index_page_is_served():
    resp = TestClient(create_app(fake_service())).get("/")
    assert resp.status_code == 200
    assert "智慧停车" in resp.text

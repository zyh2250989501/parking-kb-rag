from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from parking_kb.generate.schema import AnswerGenerator
from parking_kb.index.store import ChunkStore
from parking_kb.retrieve.hybrid import HybridRetriever

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    abstained: bool
    latency_ms: int


class ChatService:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        store: ChunkStore,
    ):
        self.retriever = retriever
        self.generator = generator
        self.store = store

    def chat(self, question: str) -> ChatResponse:
        started = perf_counter()
        candidates = self.retriever.retrieve(question)
        result = self.generator.generate(question, candidates)
        by_chunk = {
            candidate.chunk.chunk_id: candidate.chunk for candidate in candidates
        }
        citations = []
        for citation in result.citations:
            data = asdict(citation)
            chunk = by_chunk.get(citation.chunk_id)
            data["text"] = chunk.text if chunk else ""
            citations.append(data)
        return ChatResponse(
            answer=result.answer,
            citations=citations,
            abstained=result.abstained,
            latency_ms=int((perf_counter() - started) * 1000),
        )


def create_app(service: ChatService) -> FastAPI:
    app = FastAPI(title="智慧停车知识库问答")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/documents")
    def documents() -> list[dict]:
        return [asdict(meta) for meta in service.store.list_documents()]

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        return service.chat(request.question)

    return app

from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from parking_kb.index.search_text import index_text
from parking_kb.types import Chunk

COLLECTION = "chunks"


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class BgeM3Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(
            texts, batch_size=32, normalize_embeddings=True
        )
        return [vector.tolist() for vector in vectors]


class DenseIndex:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))

    def build(self, chunks: list[Chunk], embedder: Embedder) -> None:
        vectors = embedder.encode([index_text(chunk) for chunk in chunks])
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=index,
                vector=vector,
                payload={"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id},
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=COLLECTION, points=points)

    def search(
        self, query: str, embedder: Embedder, top_k: int = 20
    ) -> list[tuple[str, float]]:
        if not self.client.collection_exists(COLLECTION):
            return []
        vector = embedder.encode([query])[0]
        response = self.client.query_points(
            collection_name=COLLECTION, query=vector, limit=top_k, with_payload=True
        )
        return [
            (point.payload["chunk_id"], float(point.score))
            for point in response.points
        ]

    def close(self) -> None:
        self.client.close()

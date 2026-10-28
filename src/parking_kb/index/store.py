import json
import sqlite3
import threading
from pathlib import Path

from parking_kb.types import Chunk, DocumentMeta

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    status TEXT NOT NULL,
    doc_no TEXT,
    issuer TEXT,
    effective_date TEXT,
    version TEXT,
    source_url TEXT,
    page_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    section_path TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS versions (
    doc_no TEXT NOT NULL,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    effective_date TEXT,
    status TEXT NOT NULL,
    PRIMARY KEY (doc_no, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_versions_no ON versions(doc_no);
"""

_META_COLUMNS = (
    "doc_id",
    "title",
    "doc_type",
    "status",
    "doc_no",
    "issuer",
    "effective_date",
    "version",
    "source_url",
    "page_count",
)


def _row_to_meta(row: sqlite3.Row) -> DocumentMeta:
    return DocumentMeta(**{name: row[name] for name in _META_COLUMNS})


class ChunkStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # 服务端在线程池里执行同步端点，单连接跨线程使用必须关掉同线程检查，
        # 并由 _lock 串行化访问。
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)

    def upsert_documents(self, docs: list[DocumentMeta]) -> None:
        with self._lock, self.conn:
            for meta in docs:
                self.conn.execute(
                    """
                    INSERT INTO documents (
                        doc_id, title, doc_type, status, doc_no, issuer,
                        effective_date, version, source_url, page_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        title = excluded.title,
                        doc_type = excluded.doc_type,
                        status = excluded.status,
                        doc_no = excluded.doc_no,
                        issuer = excluded.issuer,
                        effective_date = excluded.effective_date,
                        version = excluded.version,
                        source_url = excluded.source_url,
                        page_count = excluded.page_count
                    """,
                    (
                        meta.doc_id,
                        meta.title,
                        meta.doc_type,
                        meta.status,
                        meta.doc_no,
                        meta.issuer,
                        meta.effective_date,
                        meta.version,
                        meta.source_url,
                        meta.page_count,
                    ),
                )
                self.conn.execute("DELETE FROM versions WHERE doc_id = ?", (meta.doc_id,))
                if meta.doc_no:
                    self.conn.execute(
                        """
                        INSERT INTO versions (doc_no, doc_id, effective_date, status)
                        VALUES (?, ?, ?, ?)
                        """,
                        (meta.doc_no, meta.doc_id, meta.effective_date, meta.status),
                    )

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        with self._lock, self.conn:
            for chunk in chunks:
                self.conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, text, section_path, page_start, page_end
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        doc_id = excluded.doc_id,
                        text = excluded.text,
                        section_path = excluded.section_path,
                        page_start = excluded.page_start,
                        page_end = excluded.page_end
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.text,
                        chunk.section_path,
                        chunk.page_start,
                        chunk.page_end,
                    ),
                )

    def get_document(self, doc_id: str) -> DocumentMeta | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return _row_to_meta(row) if row else None

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT c.chunk_id, c.doc_id, c.text, c.section_path,
                       c.page_start, c.page_end,
                       d.title, d.doc_type, d.status, d.doc_no, d.issuer,
                       d.effective_date, d.version, d.source_url, d.page_count
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.chunk_id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
        by_id = {
            row["chunk_id"]: Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                text=row["text"],
                section_path=row["section_path"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                meta=_row_to_meta(row),
            )
            for row in rows
        }
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def get_chunks_by_doc(self, doc_id: str) -> list[Chunk]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id = ? ORDER BY rowid",
                (doc_id,),
            ).fetchall()
        return self.get_chunks([row["chunk_id"] for row in rows])

    def list_documents(self, status: str | None = None) -> list[DocumentMeta]:
        with self._lock:
            if status is None:
                rows = self.conn.execute(
                    "SELECT * FROM documents ORDER BY doc_id"
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM documents WHERE status = ? ORDER BY doc_id",
                    (status,),
                ).fetchall()
        return [_row_to_meta(row) for row in rows]

    def find_versions(self, doc_no: str) -> list[DocumentMeta]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT doc_id FROM versions
                WHERE doc_no = ?
                ORDER BY effective_date IS NULL, effective_date
                """,
                (doc_no,),
            ).fetchall()
        docs = [self.get_document(row["doc_id"]) for row in rows]
        return [doc for doc in docs if doc is not None]

    def close(self) -> None:
        with self._lock:
            self.conn.close()


def load_chunks_jsonl(path: Path) -> list[Chunk]:
    chunks = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=record["chunk_id"],
                    doc_id=record["doc_id"],
                    text=record["text"],
                    section_path=record["section_path"],
                    page_start=record["page_start"],
                    page_end=record["page_end"],
                    meta=DocumentMeta(**record["meta"]),
                )
            )
    return chunks

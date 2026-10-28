import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from parking_kb.config import load_settings
from parking_kb.generate.client import CachedProvider, OpenAICompatibleProvider
from parking_kb.index.describe import (
    build_messages,
    chunks_to_describe,
    clean_description,
    description_is_current,
    text_fingerprint,
)
from parking_kb.index.store import load_chunks_jsonl


def load_descriptions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {
            record["chunk_id"]: record
            for record in (json.loads(line) for line in handle if line.strip())
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="为英文 chunk 生成中文检索描述（doc2query）。"
    )
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument(
        "--descriptions", type=Path, default=Path("data/chunk_descriptions.jsonl")
    )
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条，用于试跑")
    parser.add_argument(
        "--dry-run", action="store_true", help="只列出需要生成的 chunk，不调用模型"
    )
    args = parser.parse_args()

    env_file = args.env if args.env.exists() else None
    settings = load_settings(env_file)
    chunks = load_chunks_jsonl(args.chunks)
    descriptions = load_descriptions(args.descriptions)
    for chunk in chunks:
        record = descriptions.get(chunk.chunk_id, {})
        if description_is_current(chunk, record):
            chunk.search_text = record["search_text"]

    pending = chunks_to_describe(chunks)
    if args.limit:
        pending = pending[: args.limit]
    print(f"总 chunk {len(chunks)}，需要生成描述 {len(pending)}", flush=True)
    if args.dry_run:
        for chunk in pending[:20]:
            print(f"  {chunk.chunk_id} p.{chunk.page_start} {chunk.section_path[:50]}")
        return

    llm = CachedProvider(
        OpenAICompatibleProvider(
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
        ),
        settings.data_dir / "llm_cache",
    )

    args.descriptions.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.descriptions.open("a", encoding="utf-8") as handle:
        for index, chunk in enumerate(pending, start=1):
            description = ""
            for attempt in range(3):
                try:
                    description = clean_description(llm.complete(build_messages(chunk)))
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        print(f"  {chunk.chunk_id} 失败：{exc}", flush=True)
                    else:
                        time.sleep(2 * (attempt + 1))
            if not description:
                continue
            record = {
                "chunk_id": chunk.chunk_id,
                "search_text": description,
                "text_sha": text_fingerprint(chunk.text),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            chunk.search_text = description
            descriptions[chunk.chunk_id] = record
            written += 1
            if index % 20 == 0 or index == len(pending):
                print(f"  [{index}/{len(pending)}] 已写 {written}", flush=True)

    with args.chunks.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    # 每轮只追加，重跑会留下过期记录；最后按当前 chunk 重写一遍，保持一条一份。
    with args.descriptions.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            if not chunk.search_text:
                continue
            handle.write(
                json.dumps(
                    {
                        "chunk_id": chunk.chunk_id,
                        "search_text": chunk.search_text,
                        "text_sha": text_fingerprint(chunk.text),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"完成：{written} 条新描述，已写回 {args.chunks}", flush=True)
    print("下一步：重建索引（serve.py --rebuild-index 或 scripts/ingest 后重建）", flush=True)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

from parking_kb.config import load_settings
from parking_kb.eval.build_set import QUESTION_TYPES, build_eval_set, write_eval_set
from parking_kb.generate.client import OpenAICompatibleProvider
from parking_kb.index.store import load_chunks_jsonl


def parse_per_type(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for part in value.split(","):
        key, _, count = part.partition("=")
        key = key.strip()
        if key not in QUESTION_TYPES:
            raise argparse.ArgumentTypeError(f"unknown question type: {key}")
        result[key] = int(count)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stratified evaluation set.")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument(
        "--per-type",
        type=parse_per_type,
        default=parse_per_type(
            "fact=20,table=12,cross_doc=10,version=10,unanswerable=10"
        ),
    )
    parser.add_argument("--env", type=Path, default=Path(".env"))
    args = parser.parse_args()

    env_file = args.env if args.env.exists() else None
    settings = load_settings(env_file)
    llm = OpenAICompatibleProvider(
        base_url=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
    )
    chunks = load_chunks_jsonl(args.chunks)
    questions = build_eval_set(chunks, llm, args.per_type)
    write_eval_set(questions, args.out)
    counts: dict[str, int] = {name: 0 for name in QUESTION_TYPES}
    for question in questions:
        counts[question.question_type] += 1
    print(f"wrote {len(questions)} questions to {args.out}: {counts}")


if __name__ == "__main__":
    main()

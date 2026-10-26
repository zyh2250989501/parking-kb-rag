.PHONY: test ingest eval-report dev

test:
	python -m pytest -q

ingest:
	python scripts/ingest.py --corpus corpus --data data

eval-report:
	python scripts/eval.py --report-only --questions eval/questions.jsonl --out reports/eval_report.md

dev:
	uvicorn parking_kb.api.main:app --reload

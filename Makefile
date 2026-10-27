PYTHON ?= .venv/Scripts/python.exe

.PHONY: test ingest eval-report dev

test:
	$(PYTHON) -m pytest -q

ingest:
	$(PYTHON) scripts/ingest.py --corpus corpus --data data

eval-report:
	$(PYTHON) scripts/eval.py --report-only \
		--verdicts eval/judge_verdicts.jsonl \
		--out reports/eval_report.md

dev:
	$(PYTHON) scripts/serve.py --device cpu

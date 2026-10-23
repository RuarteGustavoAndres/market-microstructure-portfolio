# Deterministic demo + tests (stdlib-only demo; tests need pytest)
.PHONY: demo test
demo:
	python3 demo/audit_demo.py --write
test:
	python3 -m pytest tests/ -q

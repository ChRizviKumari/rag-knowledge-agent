export PYTHONPATH := src

.PHONY: test demo eval serve
test:
	python -m pytest -q
demo:
	python scripts/demo.py
eval:
	python scripts/run_eval.py
serve:
	python -m rag_agent serve --port 5000

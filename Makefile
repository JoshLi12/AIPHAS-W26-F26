.PHONY: setup run-smoke

# Create venv and install Winter26 (editable) with dependencies.
setup:
	cd Winter26 && (test -d .venv || python3 -m venv .venv) && . .venv/bin/activate && pip install -U pip && pip install -e .

# Quick pipeline run (no Tier 2 / SBERT); needs sample_data CSV present.
run-smoke:
	cd Winter26 && . .venv/bin/activate && python -m winter26 --max-rows 50 --skip-tier2

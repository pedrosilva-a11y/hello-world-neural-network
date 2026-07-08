.PHONY: lint type-check test coverage check format

ARGS ?=
INDEX ?= 200
PYTHONPATH := src


lint:
	PYTHONPATH=$(PYTHONPATH) uv run ruff check .

format:
	PYTHONPATH=$(PYTHONPATH) uv run ruff format .

type-check:
	PYTHONPATH=$(PYTHONPATH) uv run mypy src scripts tests

test:
	PYTHONPATH=$(PYTHONPATH) uv run python -m unittest discover -s tests -t . -p "test_*.py"

coverage:
	PYTHONPATH=$(PYTHONPATH) uv run coverage run -m unittest discover -s tests -t . -p "test_*.py"
	PYTHONPATH=$(PYTHONPATH) uv run coverage report

check: lint type-check coverage


visualize:
	PYTHONPATH=$(PYTHONPATH) uv run python scripts/visualization/visualize_input.py --index $(INDEX)

distribution:
	PYTHONPATH=$(PYTHONPATH) uv run python scripts/visualization/visualize_pixel_value_distribution.py --index $(INDEX) $(ARGS)

orchestrate:
	PYTHONPATH=$(PYTHONPATH) uv run python -m orchestration.orchestrator
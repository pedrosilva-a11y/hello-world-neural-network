.PHONY: lint type-check test coverage check format visualize distribution orchestrate plot-experiment plot-label-distribution

ARGS ?=
INDEX ?= 200
EXPERIMENT_NAME ?= experiment_1
NUM_ITERATIONS ?= 5
PYTHONPATH := src

NORMALIZE ?= no
NORMALIZE_FLAG :=
ifeq ($(NORMALIZE),yes)
NORMALIZE_FLAG := --normalize-pixels
endif

LEARNING_RATE ?= 1e-5

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

orchestrate:
	PYTHONPATH=$(PYTHONPATH) uv run python -m orchestration.orchestrator --experiment-name $(EXPERIMENT_NAME) --num-iterations $(NUM_ITERATIONS) --learning-rate $(LEARNING_RATE) $(NORMALIZE_FLAG)

plot-experiment:
ifeq ($(SPLIT),yes)
	PYTHONPATH=$(PYTHONPATH) uv run python scripts/visualization/visualize_experiment_metrics.py --experiment-name $(EXPERIMENT_NAME) --split-metrics
else
	PYTHONPATH=$(PYTHONPATH) uv run python scripts/visualization/visualize_experiment_metrics.py --experiment-name $(EXPERIMENT_NAME)
endif

plot-label-distribution:
	PYTHONPATH=$(PYTHONPATH) uv run python scripts/visualization/visualize_label_distribution.py --experiment-name $(EXPERIMENT_NAME)
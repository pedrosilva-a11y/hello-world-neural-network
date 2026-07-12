.PHONY: lint type-check test coverage check format visualize orchestrate inference dashboard-install dashboard dashboard-build dashboard-check dashboard-dev plot-experiment plot-label-distribution

ARGS ?=
CONFIG ?= conf/experiments/softmax_baseline.yaml
EXPERIMENT_NAME ?= softmax_baseline
INDEX ?= 200
PYTHONPATH := src
SUBMISSION_FILE_NAME ?= submission.csv

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
	PYTHONPATH=$(PYTHONPATH) uv run python -m orchestration.orchestrator --config $(CONFIG)

plot-experiment:
ifeq ($(SPLIT),yes)
	PYTHONPATH=$(PYTHONPATH) uv run python scripts/visualization/visualize_experiment_metrics.py --experiment-name $(EXPERIMENT_NAME) --split-metrics
else
	PYTHONPATH=$(PYTHONPATH) uv run python scripts/visualization/visualize_experiment_metrics.py --experiment-name $(EXPERIMENT_NAME)
endif

plot-label-distribution:
	PYTHONPATH=$(PYTHONPATH) uv run python scripts/visualization/visualize_label_distribution.py --experiment-name $(EXPERIMENT_NAME)

inference:
	PYTHONPATH=$(PYTHONPATH) uv run python -m orchestration.inference_orchestrator --experiment-name $(EXPERIMENT_NAME) --submission-file-name $(SUBMISSION_FILE_NAME)

dashboard-install:
	cd dashboard && npm install

dashboard:
	cd dashboard && npm run dev

dashboard-build:
	cd dashboard && npm run build

dashboard-check:
	cd dashboard && npm run check

dashboard-dev: check dashboard-check dashboard
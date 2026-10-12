.PHONY: help setup data lint baseline models hybrids report all

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## create the environment
	uv sync

data:  ## download, prepare and summarize the dataset
	uv run vgrec download
	uv run vgrec prepare
	uv run vgrec stats

lint:  ## ruff
	uv run ruff check src

baseline:  ## popularity baseline
	uv run vgrec run --model popularity

models:  ## ALS, content, NCF
	uv run vgrec run --model als
	uv run vgrec run --model content
	uv run vgrec run --model ncf

hybrids:  ## CF + content hybrids (needs baseline + models first)
	uv run vgrec run --model hybrid
	uv run vgrec run --model hybrid_pop

report:  ## aggregate results/
	uv run vgrec report

all: data baseline models hybrids report  ## everything, in order

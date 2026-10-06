PY ?= python3
CONFIGS := $(sort $(wildcard configs/experiments/0*.yaml))

.PHONY: help install lint fmt test smoke sim report-sim gb10-setup gb10 gb10-quick report-gb10 nsys kv-math

help:  ## list targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:  ## install harness + dev deps
	$(PY) -m pip install -e ".[dev]"

lint:  ## ruff lint + format check
	ruff check servebench tests && ruff format --check servebench tests

fmt:
	ruff format servebench tests && ruff check --fix servebench tests

test:  ## unit + end-to-end tests against the simulator
	$(PY) -m pytest -q

smoke:  ## fast end-to-end run of all experiments on the simulator (~2 min)
	SERVEBENCH_MOCK_TIME_SCALE=0.05 $(PY) -m servebench run $(CONFIGS) --launcher mock --quick --out /tmp/servebench-smoke

sim:  ## full simulated suite -> results/simulated (real-time simulator, ~25 min, parallel)
	rm -rf results/simulated/0* results/simulated/plots
	@port=8100; for c in $(CONFIGS); do port=$$((port+1)); \
	  $(PY) -m servebench run $$c --launcher mock --no-report --out results/simulated --port $$port & done; wait
	$(MAKE) report-sim

report-sim:
	$(PY) -m servebench report results/simulated --baseline 00_baseline

gb10-setup:  ## (on DGX Spark) pull image, download models, sanity checks
	scripts/gb10_setup.sh

gb10:  ## (on DGX Spark) full suite -> results/gb10 (~2-3 h)
	scripts/run_gb10_suite.sh results/gb10

gb10-quick:  ## (on DGX Spark) 1/8-size suite (~20-30 min)
	scripts/run_gb10_suite.sh results/gb10-quick --quick

report-gb10:
	$(PY) -m servebench report results/gb10 --baseline 00_baseline

nsys:  ## (on DGX Spark) Nsight Systems capture of baseline and combined configs
	scripts/nsys_profile.sh configs/experiments/00_baseline.yaml
	scripts/nsys_profile.sh configs/experiments/05_combined.yaml

kv-math:  ## print KV-cache capacity arithmetic for GB10
	$(PY) -m servebench kv-math --model Qwen/Qwen3-8B
	$(PY) -m servebench kv-math --model Qwen/Qwen3-8B-FP8 --kv-cache-dtype fp8

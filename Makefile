# LLM Forensics: `make help` lists everything.
#   make setup check                      local environment (CPU is fine for gpt2 + exercises)
#   make all CONFIG=configs/edits/eiffel_rome.yaml
#   make salad-run CONFIGS="configs/edits/eiffel_rome.yaml"     GPU run on SaladCloud
#   make site serve                       results site at http://localhost:8000

CONFIG  ?= configs/edits/eiffel_rome.yaml
CONFIGS ?= $(CONFIG)
UV      ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
PY      ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
LLMF     = $(PY) -m llm_forensics.cli
SALAD    = python3 scripts/salad/salad.py
PORT    ?= 8000

.DEFAULT_GOAL := help
.PHONY: help setup check test exercises edit diff blind literature investigate evaluate all all-configs \
        site serve example clean salad-check salad-gpus salad-run salad-status salad-logs salad-ssh \
        salad-ls salad-pull salad-s4 salad-stop salad-delete salad-reap gcp-create gcp-run gcp-pull gcp-stop gcp-delete

help:  ## show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- local
setup:  ## create .venv and install the package (+ anthropic, pytest) with uv
	@test -x "$(UV)" || { echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	$(UV) venv --allow-existing .venv
	$(UV) pip install -p .venv -e ".[llm,dev]"
	@test -f .env || cp .env.example .env
	@echo "next: make check"

check:  ## environment checklist (GPU, HF token, Anthropic key, disk)
	$(LLMF) check

test:  ## run the unit tests
	$(PY) -m pytest -q tests

exercises:  ## run every exercise solution (they should all pass)
	@for f in exercises/solutions/0*.py; do echo "== $$f"; $(PY) $$f || exit 1; done

# ---------------------------------------------------------------- pipeline (one config)
edit:  ## plant the edit -> runs/<name>/
	$(LLMF) edit $(CONFIG)
diff:  ## weight diff vs original -> results/<name>/diff.json
	$(LLMF) diff $(CONFIG)
blind:  ## original-free statistics -> results/<name>/blind.json
	$(LLMF) blind $(CONFIG)
literature:  ## published detectors -> results/<name>/literature.json
	$(LLMF) literature $(CONFIG)
investigate:  ## Claude investigator (needs ANTHROPIC_API_KEY) -> results/<name>/investigator_*.json
	$(LLMF) investigate $(CONFIG)
evaluate:  ## score every report vs ground truth -> results/<name>/summary.json
	$(LLMF) evaluate $(CONFIG)
all:  ## everything above for CONFIG
	$(LLMF) all $(CONFIG) --keep-going
all-configs:  ## `make all` for every configs/edits/*.yaml
	@for c in configs/edits/*.yaml; do $(LLMF) all $$c --keep-going || true; done

# ---------------------------------------------------------------- site
site:  ## build the static site into site/_build
	$(PY) site/build.py
example:  ## regenerate the synthetic results/_example fixture
	$(PY) site/make_example.py
serve: site  ## build and serve the site locally
	$(PY) -m http.server -d site/_build $(PORT)

clean:  ## remove build output and caches (keeps runs/ and results/)
	rm -rf site/_build .pytest_cache **/__pycache__

# ---------------------------------------------------------------- SaladCloud (primary GPU path)
salad-check:  ## verify the Salad API key / org / project (read-only)
	$(SALAD) check
salad-gpus:  ## list 24 GB GPU classes and prices at PRIORITY (default low)
	$(SALAD) gpus "24 GB"
salad-run:  ## create + start a group running `llmf all` on CONFIGS (billable; self-stops)
	LLMF_CONFIGS="$(CONFIGS)" $(SALAD) create --start
salad-status:  ## state of NAME (default llmf-run) and its SSH endpoint
	$(SALAD) status
salad-ls:  ## every group in the project + estimated cost so far
	$(SALAD) ls
salad-logs:  ## container stdout/stderr, last hour
	$(SALAD) logs
salad-ssh:  ## shell on the running instance
	$(SALAD) ssh
salad-pull:  ## wait for the run to finish, then scp results/ back
	$(SALAD) pull --wait
salad-s4:  ## fetch the latest results backup from Salad S4 (after the group stopped)
	$(SALAD) s4 --fetch
salad-stop:  ## stop NAME (no charges while stopped)
	$(SALAD) stop
salad-delete:  ## delete NAME
	$(SALAD) delete
salad-reap:  ## stop EVERY running group in the project
	$(SALAD) reap

# ---------------------------------------------------------------- GCP (optional fallback)
gcp-create:  ## create a GPU VM (L4 by default; SPOT=1, GPU_TYPE=...)
	scripts/gcp/create_vm.sh
gcp-run:  ## sync repo to the VM and run `make all CONFIG=...` in tmux
	scripts/gcp/run_remote.sh $(CONFIG)
gcp-pull:  ## copy results/ back from the VM
	scripts/gcp/run_remote.sh --pull
gcp-stop:  ## stop the VM
	scripts/gcp/stop_vm.sh
gcp-delete:  ## delete the VM and its disk
	scripts/gcp/delete_vm.sh

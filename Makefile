# Fail-fast shell for all recipes
SHELL := bash
.SHELLFLAGS := -e -o pipefail -c

UV_BIN ?= $(HOME)/.local/bin/uv
ZEN_UV_VERSION ?= 0.7.5
TWINE ?= $(HOME)/.local/bin/twine
V ?= 0

ifeq ($(V),1)
  $(info Running in VERBOSE mode)
  UV := MAX_JOBS=8 UV_LOG=debug $(UV_BIN)
  TWINE_VERBOSE := --verbose
else
  UV := MAX_JOBS=8 $(UV_BIN)
endif

.DEFAULT_GOAL := dep
.PHONY: dep init check lock build upload ensure_env

$(UV_BIN):
	curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/$(ZEN_UV_VERSION)/install.sh | sh

init: $(UV_BIN)
	$(UV) self update $(ZEN_UV_VERSION)

ensure_env:
	@echo "Checking environment variables..."
	@missing=0; \
	[ -z "$$CUDA_PATH" ] && { echo "-- NOT FOUND! Try: export CUDA_PATH=/usr/local/cuda"; missing=1; } || echo "-- CUDA_PATH=$$CUDA_PATH"; \
	[ -z "$$LD_LIBRARY_PATH" ] && { echo "-- NOT FOUND! Try: export LD_LIBRARY_PATH=\$$CUDA_PATH/lib64"; missing=1; } || echo "-- LD_LIBRARY_PATH=$$LD_LIBRARY_PATH"; \
	[ "$$missing" -eq 0 ] || { echo "Aborting due to missing env"; exit 1; }

.venv: uv.lock $(UV_BIN) ensure_env
	$(UV) sync
	touch .venv

dep: init .venv

check:
	$(UV) lock --check

lock:
	$(UV) lock

$(TWINE):
	$(UV) tool install twine

build: dep
	$(UV) build

upload: $(TWINE)
	@command -v gcloud >/dev/null || { echo "gcloud not found"; exit 1; }
	@test -n "$$(ls -1 dist 2>/dev/null)" || { echo "No files in dist/"; exit 1; }
	TWINE_USERNAME=oauth2accesstoken \
	TWINE_PASSWORD=$$(gcloud auth print-access-token) \
	$(TWINE) upload $(TWINE_VERBOSE) --non-interactive \
	  --repository-url https://us-central1-python.pkg.dev/artifacts-443721/python-packages/ \
	  dist/*

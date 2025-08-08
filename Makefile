PYTHON ?= python3.11
UV_BIN ?= $(HOME)/.local/bin/uv
ZEN_UV_VERSION ?= 0.7.5
TWINE ?= .venv/bin/twine
V ?= 0

ifeq ($(V),1)
  $(info Running in VERBOSE mode)
  UV := MAX_JOBS=8 UV_LOG=debug $(UV_BIN)
  TWINE_VERBOSE := --verbose
else
  UV := MAX_JOBS=8 $(UV_BIN)
endif


###########################################
# Make commands:
###########################################

$(UV_BIN):
	curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/$(ZEN_UV_VERSION)/install.sh | sh

.venv: uv.lock $(UV_BIN) ensure_env
	@set -e; \
		$(UV) sync
		touch .venv # touch to update the timestamp

.PHONY: init
init: $(UV_BIN)
	$(UV) self update $(ZEN_UV_VERSION)

DEFAULT_CUDA_PATH ?= /usr/local/cuda
LD_LIBRARY_PATH ?= $(DEFAULT_CUDA_PATH)/lib64
.PHONY: ensure_env
ensure_env:
	@echo "Checking environment variables..."
	@missing=0; \
	if [ -z "$$CUDA_PATH" ]; then \
		echo "-- CUDA_PATH not set. Try this:"; \
                echo "export CUDA_PATH=$(DEFAULT_CUDA_PATH)"; \
		missing=1; \
	else \
		echo "-- CUDA_PATH is set to $$CUDA_PATH"; \
	fi; \
	if [ -z "$$LD_LIBRARY_PATH" ]; then \
		echo "-- LD_LIBRARY_PATH not set. Try this:"; \
                echo "export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)"; \
		missing=1; \
	else \
		echo "-- LD_LIBRARY_PATH is set to $$LD_LIBRARY_PATH"; \
	fi; \
	if [ "$$missing" -eq 1 ]; then \
		echo "Aborting build due to missing environment variables."; \
		exit 1; \
	fi

.PHONY: dep
.DEFAULT_GOAL := dep
dep: init .venv

.PHONY: check
check:
	$(UV) lock --check

.PHONY: lock
lock:
	$(UV) lock

$(TWINE):
	$(UV) pip install .[dev]

.PHONY: upload
upload: $(TWINE)
	@TWINE_USERNAME=oauth2accesstoken \
	TWINE_PASSWORD=$$(gcloud auth print-access-token) \
	$(TWINE) upload $(TWINE_VERBOSE) \
	  --non-interactive \
	  --repository-url https://us-central1-python.pkg.dev/artifacts-443721/python-packages/ \
	  dist/*

.PHONY: build
build: dep
	$(UV) build

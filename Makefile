PYTHON ?= python3.11
UV ?= $(HOME)/.local/bin/uv
ZEN_UV_VERSION ?= 0.7.5

###########################################
# Make commands:
###########################################

$(UV):
	curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/$(ZEN_UV_VERSION)/install.sh | sh

.venv: uv.lock $(UV)
	@set -e; \
		MAX_JOBS=10 $(UV) sync
		touch .venv # touch to update the timestamp

.PHONY: init
init: $(UV)
	$(UV) self update $(ZEN_UV_VERSION)

.PHONY: dep
.DEFAULT_GOAL := dep
dep: init .venv

.PHONY: check
check:
	$(UV) lock --check

.PHONY: lock
lock:
	$(UV) lock

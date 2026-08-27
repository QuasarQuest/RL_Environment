# Canonical build / dev commands for RL_Environment.
# IDE run configs (.idea/runConfigurations) call these targets instead of
# duplicating the underlying commands — this file is the single source of
# truth for how to build, check, and run things. Run everything from the
# repo root.

# Non-login shells (IDE run configs, non-interactive `make` invocations)
# never source ~/.bashrc or ~/.profile, so cargo/rustc from rustup, and uv /
# python3.12 from uv, would be missing from PATH without this.
export PATH := $(HOME)/.cargo/bin:$(HOME)/.local/bin:$(PATH)

VENV        := $(CURDIR)/rl/.venv

# Pin the target venv explicitly: maturin auto-detects a venv from cwd (it
# prefers a stray ./.venv over the one its own binary lives in), so without
# this it can silently install into the wrong environment.
export VIRTUAL_ENV := $(VENV)
PYTHON      := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
MATURIN     := $(VENV)/bin/maturin
ATB_TRAIN   := $(VENV)/bin/atb-train
ATB_MONITOR := $(VENV)/bin/atb-monitor

.PHONY: help venv build build-debug check check-python check-rust \
        train train-sequential resume-latest monitor profile \
        viewer viewer-fast clean

help:
	@echo "venv           create rl/.venv and install rl[dev] + maturin"
	@echo "build          build the Rust extension (release) into rl/.venv"
	@echo "build-debug    same, debug profile"
	@echo "check          full lint + test suite (ruff, pyright, clippy, cargo test)"
	@echo "check-python   ruff + pyright only"
	@echo "check-rust     clippy + cargo test only"
	@echo "train          atb-train (ARGS=\"...\" to pass overrides)"
	@echo "train-sequential  run rl/scripts/train_sequential.py"
	@echo "resume-latest  resume the most recent run (+1M steps)"
	@echo "monitor        atb-monitor (ARGS=\"...\" for the stats file)"
	@echo "profile        run rl/scripts/profile_training.py"
	@echo "viewer         cargo run -p viewer"
	@echo "viewer-fast    cargo run --profile viewer-fast -p viewer --bin atb-view"
	@echo "clean          remove target/ and rl/.venv"

# Pinned to 3.12 (matches CI's python:3.12 image) via uv, not the system
# python3 — hydra-core 1.3.5 crashes on Python 3.14's stricter argparse.
venv:
	uv python install 3.12
	python3.12 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e "./rl[dev]" maturin

build:
	$(MATURIN) develop --manifest-path sim/Cargo.toml --features python --release

build-debug:
	$(MATURIN) develop --manifest-path sim/Cargo.toml --features python

check:
	./checker.bash

check-python:
	./checker.bash python

check-rust:
	./checker.bash rust

train:
	cd rl && PYTHONUNBUFFERED=1 $(ATB_TRAIN) $(ARGS)

train-sequential:
	cd rl && PYTHONUNBUFFERED=1 $(PYTHON) scripts/train_sequential.py $(ARGS)

resume-latest:
	cd rl && PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE $(PYTHON) scripts/resume_latest.py --add 1000000

monitor:
	cd rl && $(ATB_MONITOR) $(ARGS)

profile:
	cd rl && $(PYTHON) scripts/profile_training.py

viewer:
	cargo run -p viewer

viewer-fast:
	cargo run --profile viewer-fast --package viewer --bin atb-view

clean:
	rm -rf target rl/.venv

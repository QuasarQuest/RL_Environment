#!/usr/bin/env bash
# Full check suite for the repo — run manually before pushing, or let the
# pre-commit hook run the same tools on staged files (.pre-commit-config.yaml;
# install once with: rl/.venv/bin/pre-commit install).
#
#   ./checker.bash          # everything (Python lint + types, Rust lint + tests)
#   ./checker.bash python   # only ruff + pyright
#   ./checker.bash rust     # only clippy + cargo test
set -euo pipefail
cd "$(dirname "$0")"

what="${1:-all}"

if [[ "$what" == "all" || "$what" == "python" ]]; then
    echo "── ruff ─────────────────────────────────────────────"
    rl/.venv/bin/ruff check rl/src rl/scripts

    echo "── pyright ──────────────────────────────────────────"
    rl/.venv/bin/pyright
fi

if [[ "$what" == "all" || "$what" == "rust" ]]; then
    echo "── clippy (deny warnings) ───────────────────────────"
    cargo clippy -p algorithm_test_bed --all-features -- -D warnings
    cargo clippy -p viewer -- -D warnings

    echo "── cargo test ───────────────────────────────────────"
    cargo test -p algorithm_test_bed --all-features --quiet
fi

echo "All checks passed."

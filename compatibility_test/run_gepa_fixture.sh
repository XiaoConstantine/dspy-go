#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE_DIR="$ROOT_DIR/compatibility_test"
PYTHON_RESULTS="$FIXTURE_DIR/dspy_gepa_fixture_results.json"
GO_RESULTS="$FIXTURE_DIR/go_gepa_fixture_results.json"
REPORT_RESULTS="$FIXTURE_DIR/gepa_fixture_report.json"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/dspy-go-gepa-compat-uv-cache}"
export GOCACHE="${GOCACHE:-${TMPDIR:-/tmp}/dspy-go-gepa-compat-gocache}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to run the deterministic DSPy fixture"
    exit 1
fi

echo "Running deterministic DSPy GEPA fixture..."
(
    cd "$FIXTURE_DIR"
    uv run --python 3.10 dspy_gepa_fixture.py --output "$PYTHON_RESULTS"
)

echo "Running deterministic dspy-go GEPA fixture..."
(
    cd "$ROOT_DIR"
    go run ./compatibility_test/go_gepa_fixture.go --output "$GO_RESULTS"
)

echo "Comparing GEPA fixture outputs..."
(
    cd "$FIXTURE_DIR"
    uv run --python 3.10 compare_gepa_fixture.py \
        --python-results "$PYTHON_RESULTS" \
        --go-results "$GO_RESULTS" \
        --output "$REPORT_RESULTS"
)

echo "GEPA fixture report written to $REPORT_RESULTS"

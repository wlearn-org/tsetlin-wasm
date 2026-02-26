#!/bin/bash
set -euo pipefail

# Verify that the built WASM module exports all expected symbols.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WASM_FILE="${PROJECT_DIR}/wasm/tsetlin.cjs"

if [ ! -f "$WASM_FILE" ]; then
  echo "ERROR: ${WASM_FILE} not found. Run build-wasm.sh first."
  exit 1
fi

EXPECTED_EXPORTS=(
  wl_tm_get_last_error
  wl_tm_create
  wl_tm_free
  wl_tm_fit
  wl_tm_predict
  wl_tm_predict_votes
  wl_tm_save
  wl_tm_load
  wl_tm_free_buffer
  wl_tm_get_n_features
  wl_tm_get_n_classes
  wl_tm_get_n_clauses
  wl_tm_get_task
  wl_tm_get_n_binary
  wl_tm_get_threshold
  wl_tm_get_y_min
  wl_tm_get_y_max
)

missing=0
for fn in "${EXPECTED_EXPORTS[@]}"; do
  if ! grep -q "_${fn}" "$WASM_FILE"; then
    echo "MISSING: _${fn}"
    missing=$((missing + 1))
  fi
done

if [ $missing -gt 0 ]; then
  echo "ERROR: ${missing} exports missing from ${WASM_FILE}"
  exit 1
fi

echo "All ${#EXPECTED_EXPORTS[@]} exports verified."

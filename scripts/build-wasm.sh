#!/bin/bash
set -euo pipefail

# Build Tsetlin Machine WASM via Emscripten
# Prerequisites: emsdk activated (emcc in PATH)
#
# Compiles wl_tm_api.c (glue layer) + TMU C core (ClauseBank, WeightBank, Tools, PRNG)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UPSTREAM_DIR="${PROJECT_DIR}/upstream/tmu/tmu/lib"
OUTPUT_DIR="${PROJECT_DIR}/wasm"

# Verify prerequisites
if ! command -v emcc &> /dev/null; then
  echo "ERROR: emcc not found. Activate emsdk first:"
  echo "  source /path/to/emsdk/emsdk_env.sh"
  exit 1
fi

if [ ! -f "$UPSTREAM_DIR/src/ClauseBank.c" ]; then
  echo "ERROR: TMU upstream not found at ${UPSTREAM_DIR}"
  echo "  git submodule update --init"
  exit 1
fi

echo "=== Compiling WASM ==="
mkdir -p "$OUTPUT_DIR"

EXPORTED_FUNCTIONS='[
  "_wl_tm_get_last_error",
  "_wl_tm_create",
  "_wl_tm_free",
  "_wl_tm_fit",
  "_wl_tm_predict",
  "_wl_tm_predict_votes",
  "_wl_tm_save",
  "_wl_tm_load",
  "_wl_tm_free_buffer",
  "_wl_tm_get_n_features",
  "_wl_tm_get_n_classes",
  "_wl_tm_get_n_clauses",
  "_wl_tm_get_task",
  "_wl_tm_get_n_binary",
  "_wl_tm_get_threshold",
  "_wl_tm_get_y_min",
  "_wl_tm_get_y_max",
  "_malloc",
  "_free"
]'

EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF64","HEAPU8","HEAP32"]'

emcc \
  "${PROJECT_DIR}/csrc/wl_tm_api.c" \
  "${UPSTREAM_DIR}/src/ClauseBank.c" \
  "${UPSTREAM_DIR}/src/WeightBank.c" \
  "${UPSTREAM_DIR}/src/Tools.c" \
  "${UPSTREAM_DIR}/src/random/pcg32_fast.c" \
  -I "${UPSTREAM_DIR}/include" \
  -o "${OUTPUT_DIR}/tsetlin.cjs" \
  -std=c11 \
  -s MODULARIZE=1 \
  -s SINGLE_FILE=1 \
  -s EXPORT_NAME=createTsetlin \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCTIONS}" \
  -s EXPORTED_RUNTIME_METHODS="${EXPORTED_RUNTIME_METHODS}" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=16777216 \
  -s ENVIRONMENT='web,node' \
  -O2

echo "=== Verifying exports ==="
bash "${SCRIPT_DIR}/verify-exports.sh"

echo "=== Writing BUILD_INFO ==="
cat > "${OUTPUT_DIR}/BUILD_INFO" <<EOF
upstream: TMU v0.8.3 (Tsetlin Machine Unified)
upstream_commit: $(cd "${PROJECT_DIR}/upstream/tmu" && git rev-parse HEAD 2>/dev/null || echo "unknown")
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
emscripten: $(emcc --version | head -1)
build_flags: -O2 -std=c11 SINGLE_FILE=1
wasm_embedded: true
EOF

echo "=== Build complete ==="
ls -lh "${OUTPUT_DIR}/tsetlin.cjs"
cat "${OUTPUT_DIR}/BUILD_INFO"

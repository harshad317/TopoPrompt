#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CACHE_PATH="$ROOT_DIR/cache/refined_base_v1.npz"
STRESS_CACHE_PATH="$ROOT_DIR/cache/stress_signals_crosses_v1.npz"

cd "$ROOT_DIR"

while [[ ! -f "$CACHE_PATH" ]]; do
  sleep 60
done

run_experiment() {
  local log_path="$1"
  shift

  if ! PYTHONUNBUFFERED=1 uv run scripts/solution.py "$@" > "$log_path" 2>&1; then
    echo "Experiment failed: $*" >&2
  fi
}

run_experiment logreg_stack_raw.log \
  --run-name logreg_stack_raw_v1 \
  --decision-policy logreg_stack \
  --meta-raw-features \
  --prediction-cache "$CACHE_PATH" \
  --skip-predictions

run_experiment ann_stack_raw.log \
  --run-name ann_stack_raw_v1 \
  --decision-policy mlp_stack \
  --meta-raw-features \
  --prediction-cache "$CACHE_PATH" \
  --skip-predictions

run_experiment stress_signals_crosses.log \
  --run-name stress_signals_crosses_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --prediction-cache "$STRESS_CACHE_PATH" \
  --skip-predictions

run_experiment stress_signals_class_scale.log \
  --run-name stress_signals_class_scale_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --decision-policy class_scale_search \
  --prediction-cache "$STRESS_CACHE_PATH" \
  --skip-predictions

run_experiment stress_signals_ann_stack.log \
  --run-name stress_signals_ann_stack_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --decision-policy mlp_stack \
  --meta-raw-features \
  --prediction-cache "$STRESS_CACHE_PATH" \
  --skip-predictions

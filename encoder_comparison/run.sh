#!/usr/bin/env bash
# Usage: bash run.sh labels|features|probes|all|test [options]
# GPU sets the card for features, default 3. Other users' jobs are never touched.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
PYTHON="${CSFM_PYTHON:-$(command -v python3)}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${GPU:-3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$HERE:$HERE/probe_labels:$REPO/finetune/csfm_mae${PYTHONPATH:+:$PYTHONPATH}"
stage="${1:-}"
[[ $# -gt 0 ]] && shift
cd "$HERE"
case "$stage" in
    labels) exec "$PYTHON" probe_labels/build.py "$@" ;;
    features) exec "$PYTHON" features/extract.py "$@" ;;
    probes) exec "$PYTHON" probes/evaluate.py "$@" ;;
    all)
        "$PYTHON" probe_labels/build.py
        "$PYTHON" features/extract.py
        exec "$PYTHON" probes/evaluate.py
        ;;
    test) exec "$PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -t "$HERE" -v ;;
    *) printf 'Usage: bash run.sh labels|features|probes|all|test [options]\n' >&2; exit 2 ;;
esac

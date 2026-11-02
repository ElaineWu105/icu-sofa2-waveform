#!/usr/bin/env bash
# Usage: bash run.sh plan | cut [--workers N ...] | report | test
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SIGNAL_PYTHON:-$(command -v python3)}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
stage="${1:-}"
[[ $# -gt 0 ]] && shift
cd "$HERE"
case "$stage" in
    plan) exec "$PYTHON" plan.py "$@" ;;
    cut) exec "$PYTHON" cut.py "$@" ;;
    report) exec "$PYTHON" cut.py --report-only "$@" ;;
    test) exec "$PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -v ;;
    *) printf 'Usage: bash run.sh plan|cut|report|test [options]\n' >&2; exit 2 ;;
esac

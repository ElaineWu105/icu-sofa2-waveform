#!/usr/bin/env bash
# Usage: bash run.sh extract | score | finalize
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${LABEL_PYTHON:-$(command -v python3)}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$HERE"
case "${1:-}" in
    extract)
        mkdir -m 700 -p private/raw
        for task in chart lab emar small; do
            "$PYTHON" extract.py "$task" private/raw > "private/raw/$task.log" &
        done
        wait
        ;;
    score) "$PYTHON" score.py | tee private/score.log ;;
    finalize) "$PYTHON" finalize.py | tee private/finalize.log ;;
    *) printf 'Usage: bash run.sh extract|score|finalize\n' >&2; exit 2 ;;
esac

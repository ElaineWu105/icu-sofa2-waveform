#!/usr/bin/env bash
# Usage: bash run.sh extract | score | finalize | vitals | e2 | test
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${LABEL_PYTHON:-$(command -v python3)}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$HERE"
case "${1:-}" in
    extract)
        mkdir -m 700 -p private/raw
        for task in chart lab emar small; do
            "$PYTHON" extract/extract.py "$task" private/raw > "private/raw/$task.log" &
        done
        wait
        ;;
    score) "$PYTHON" score/score.py | tee private/score.log ;;
    finalize) "$PYTHON" score/finalize.py | tee private/finalize.log ;;
    vitals) "$PYTHON" extract/extract.py vitals private > private/vitals.log ;;
    e2) "$PYTHON" e2/e2.py | tee private/e2.log ;;
    test) "$PYTHON" -m unittest discover -s "$HERE/score" -p 'test_*.py' ;;
    *) printf 'Usage: bash run.sh extract|score|finalize|vitals|e2|test\n' >&2; exit 2 ;;
esac

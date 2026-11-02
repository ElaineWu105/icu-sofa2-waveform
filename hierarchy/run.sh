#!/usr/bin/env bash
# Usage: bash run.sh index | three_min | features | check | one_hour | day | test [options]
# GPUS picks the cards, for example GPUS=3,7. Other users' jobs are never touched.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${HIERARCHY_PYTHON:-$(command -v python3)}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPUS:-3}"
count="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
parallel() {
    if [[ "$count" -gt 1 ]]; then
        exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$count" "$@"
    fi
    exec "$PYTHON" "$@"
}
stage="${1:-}"
[[ $# -gt 0 ]] && shift
cd "$HERE"
case "$stage" in
    index) exec "$PYTHON" build_index.py "$@" ;;
    three_min) parallel train_three_min.py "$@" ;;
    features) parallel extract_three_min.py "$@" ;;
    check) exec "$PYTHON" staged_check.py "$@" ;;
    one_hour) exec "$PYTHON" train_one_hour.py "$@" ;;
    day) exec "$PYTHON" train_day.py "$@" ;;
    test) exec "$PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' ;;
    *) printf 'Usage: bash run.sh index|three_min|features|check|one_hour|day|test [options]\n' >&2; exit 2 ;;
esac

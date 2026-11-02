#!/usr/bin/env bash
# Usage: bash run.sh text | check | test | train A|B [options]
# Default GPUs: arm A 0,1,2,3 and arm B 4,5,6,7. Override with GPUS=...
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${CSFM_PYTHON:-$(command -v python3)}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
# Default NCCL peer-to-peer hangs on this host; shared memory transport works.
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
stage="${1:-}"
[[ $# -gt 0 ]] && shift
cd "$HERE"
case "$stage" in
    text) exec "$PYTHON" prepare_text.py "$@" ;;
    check) exec "$PYTHON" check_mae.py "$@" ;;
    test) exec "$PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -v ;;
    train)
        arm="$1"; shift
        case "$arm" in
            A) default=0,1,2,3; port=29541 ;;
            B) default=4,5,6,7; port=29542 ;;
            *) printf 'Arm must be A or B\n' >&2; exit 2 ;;
        esac
        export CUDA_VISIBLE_DEVICES="${GPUS:-$default}"
        count="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
        exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$count" --master_port="$port" \
            finetune.py --arm "$arm" --output "${OUTPUT:-$HERE/runs/arm_$arm}" "$@"
        ;;
    *) printf 'Usage: bash run.sh text|check|test|train A|B [options]\n' >&2; exit 2 ;;
esac

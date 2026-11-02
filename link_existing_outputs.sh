#!/usr/bin/env bash
# Point this repository at outputs that already exist, so finished stages need not rerun.
# Usage: bash link_existing_outputs.sh [PROJECT_ROOT]
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "${1:-$REPO/../..}" && pwd)"
link() {
    [[ -e "$PROJECT/$2" ]] || { printf 'Missing: %s\n' "$PROJECT/$2" >&2; exit 1; }
    [[ -e "$REPO/$1" || -L "$REPO/$1" ]] || ln -s "$PROJECT/$2" "$REPO/$1"
}
link pretrained_model_for_share  pretrained_model_for_share
link waveform/windows_10s/output 10.14/signal_preprocess_10s_joint_csfm/output
link waveform/icu_cut/inputs     10.16/csfm_ehr_text_coverage/output
link waveform/icu_cut/output     10.30/sofa2_icu_pipeline/waveform
link finetune/arms/text          10.30/sofa2_icu_pipeline/csfm/text
link finetune/arms/runs          10.30/sofa2_icu_pipeline/csfm/runs
link encoder_comparison/private                 11.1/csfm_downstream/private
link encoder_comparison/label_audit/private     11.1/csfm_downstream/label_audit/private
link encoder_comparison/epoch17_best_encoder.pt 10.17/csfm_finetune/runs/csfm_base_10s_100ep/best_encoder.pt

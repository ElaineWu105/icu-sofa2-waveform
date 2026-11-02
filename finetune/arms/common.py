"""Shared paths and reused 10.17 MAE code for the ICU fine-tune."""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FINETUNE_DIR = ROOT / "finetune/csfm_mae"
CUT_CODE_DIR = ROOT / "waveform/icu_cut"
for _path in (CUT_CODE_DIR, FINETUNE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

CUT_DIR = CUT_CODE_DIR / "output/cut"
PLAN_PATH = CUT_CODE_DIR / "output/plan/plan.sqlite"
TEXT_DIR = HERE / "text"
CHECKPOINT = ROOT / "pretrained_model_for_share/CSFM_base.pth"

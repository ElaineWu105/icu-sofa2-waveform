# ICU waveforms to SOFA-2 change

Predict the change in six SOFA-2 organ scores and the total over the next 24 hours, from ICU ECG, PPG and Resp waveforms plus baseline SOFA-2. Data: MIMIC-IV.

| Stage | Status |
|---|---|
| SOFA-2 labels and cohort | Done |
| 10-second waveform cut | Done |
| CSFM fine-tuning, arms A and B | Done |
| Encoder comparison | Done, arm B chosen |
| 3-minute SSL | Running |
| 1-hour SSL, 24-hour SOFA-2 training | Code ready |

## Setup

Needs Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Edit the two paths in `env.sh`, then load them. Do this in every new terminal, along with `source .venv/bin/activate`.

```bash
source env.sh
```

Check the install:

```bash
bash waveform/icu_cut/run.sh test
bash finetune/arms/run.sh test
bash encoder_comparison/run.sh test
```

## Data

Credentialed PhysioNet access is required for both MIMIC datasets.

| Item | Put it here |
|---|---|
| [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/) | `MIMIC_IV_DIR` in `env.sh` |
| MIMIC-IV Waveform Database v0.2 prerelease, with `v0.2_tmp_starttime_index.csv` | `MIMIC_WAVEFORM_DIR` in `env.sh` |
| CSFM-Base weights, from the [CSFM authors](https://github.com/guxiao0822/Cardiac-Sensing-FM) | `pretrained_model_for_share/CSFM_base.pth` |
| Patient split, columns `subject_id,split` | `waveform/icu_cut/inputs/pairs.csv` |

The split file holds patient IDs, so it is not shared. Without it, create a header-only file and every patient is assigned 70/15/15 by a hash of their ID:

```bash
mkdir -p waveform/icu_cut/inputs
printf 'subject_id,split\n' > waveform/icu_cut/inputs/pairs.csv
```

The 10.17 epoch 17 encoder is only a comparison arm. Its weights are not reproduced here, so without `encoder_comparison/epoch17_best_encoder.pt` remove `epoch17` from `ENCODERS` in `encoder_comparison/common.py`.

## Run

Run in this order, from the repository root. Times are from our run. The cut needs about 5.1 TB of disk.

**1. Cut the waveforms**

```bash
bash waveform/icu_cut/run.sh plan --workers 16      # 3 minutes
bash waveform/icu_cut/run.sh cut --workers 64       # 11.5 hours
```

A stopped cut resumes when you rerun the same command.

The waveform index ends some records early. These two commands add the 1,005 ICU days that this drops, without changing the days above. They take 25 minutes.

```bash
T="$PWD/waveform/icu_cut/output_tail"
bash waveform/icu_cut/run.sh plan --output "$T/plan" --workers 16 --ignore-index-end \
    --skip-plan "$PWD/waveform/icu_cut/output/plan/plan.sqlite"
bash waveform/icu_cut/run.sh cut --plan "$T/plan/plan.sqlite" --output "$T/cut" --workers 48
```

**2. Build SOFA-2 labels and the cohort**

```bash
bash labels/run.sh extract      # 11 minutes
bash labels/run.sh score        # 2 minutes
bash labels/run.sh finalize     # 10 seconds
```

Patient tables go to `labels/private/`. Cohort counts go to `labels/cohort_report.json`.

**3. Fine-tune CSFM**

```bash
bash finetune/arms/run.sh text      # 10 minutes, charted text for arm B
bash finetune/arms/run.sh check     # 1 minute, confirms the training target
GPUS=0,1,2,3 bash finetune/arms/run.sh train A
GPUS=4,5,6,7 bash finetune/arms/run.sh train B
```

Each arm took 1 to 1.5 days on one shared GPU. To use one GPU, keep GPUs × batch × accumulation at 256:

```bash
GPUS=0 bash finetune/arms/run.sh train A --batch-size 32 --accumulation 8 --workers 12
```

Weights go to `finetune/arms/runs/arm_A/` and `arm_B/`. `best_overall.pt` is the selected encoder. A stopped run resumes from `last.pt`.

**4. Compare encoders**

```bash
python encoder_comparison/label_audit/extract_chart.py      # 25 minutes
GPU=0 bash encoder_comparison/run.sh all                    # 2.5 hours
```

Read `encoder_comparison/results/report.md`. Results for our run are already there. Our comparison ran before the added days.

**5. Train the three levels**

Only level 3 has a SOFA-2 loss. `GPUS` picks the cards. Times are for two GPUs.

```bash
bash hierarchy/run.sh index                    # 1 minute
GPUS=3,7 bash hierarchy/run.sh three_min       # Level 1, up to 2.5 days
GPUS=3,7 bash hierarchy/run.sh features        # 2.5 days, resumes by ICU day
bash hierarchy/run.sh check                    # Is there signal? Read this before going on
GPUS=3 bash hierarchy/run.sh one_hour          # Level 2
GPUS=3 bash hierarchy/run.sh day               # Level 3 and its controls
```

Results are in `hierarchy/private/runs/`: `staged_check.json` for the check and `day/validation_metrics.json` for level 3. Compare `hierarchy` with `baseline` and persistence. The test split is not read.

## Layout

```
labels/                SOFA-2 scoring and cohort
waveform/
  native_index/        WFDB headers and clock checks
  windows_10s/         Read and clean one 10-second window
  icu_cut/             ICU time grid, planning and cutting
  coverage_audit/      Where ICU days are lost
finetune/
  csfm_mae/            CSFM model and tokenizer
  arms/                Text, target check and training
encoder_comparison/    Frozen encoders on downstream tasks
hierarchy/             Three training levels and the SOFA-2 head
docs/                  Method documents
```

## Documents

| File | Content |
|---|---|
| `docs/COHORT.md` | Cohort definition, sample sizes, splits, missingness, how SOFA-2 is scored |
| `docs/SETTING.md` | Full method, including the planned training levels |
| `docs/WAVEFORM_AND_FINETUNE.md` | How the cut and fine-tuning were done, and their results |
| `encoder_comparison/README.md` | Comparison method and limits |
| `SOURCES.tsv` | Where each file came from, and what changed |

## Patient data

Do not commit or share `labels/private/`, `encoder_comparison/private/`, `hierarchy/private/`, waveform outputs or the split file. `.gitignore` excludes them. MIMIC data is covered by the PhysioNet data use agreement.

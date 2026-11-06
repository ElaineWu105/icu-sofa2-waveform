# ICU waveforms to SOFA-2 change

Predict how each of the six SOFA-2 organ scores changes over the next 24 hours of an ICU stay, from bedside ECG, PPG and Resp waveforms plus charted EHR. Data: MIMIC-IV 3.1 and the MIMIC-IV Waveform Database v0.2 prerelease.

The pipeline adds a Resp channel to the CSFM foundation model, builds a 24-hour waveform representation on top of it in three levels, and trains a SOFA-2 head on the last level only.

## Pipeline

| Step | Folder | What it does | State on 2026-11-06 |
|---|---|---|---|
| 1. Cohort and labels | `labels/` | SOFA-2 per ICU day, anchors, death and discharge handling, EHR inputs of the head | Done. 22,341 anchors, 6,416 stays, 5,406 patients |
| 2. Waveform cut | `waveform/` | ECG, PPG and Resp cut into 10-second windows on the ICU time grid | Done. 23,131 planned days, about 195M windows |
| 3. Fine-tuning | `finetune/` | CSFM with a new Resp channel, masked training. Arms A (no text), B, B2 and B3 (with text) | Done. 60,000 updates each |
| 4. Encoder choice | `encoder_comparison/` | Frozen encoders with XGBoost probes on validation patients | Done. **Arm A chosen** |
| 5. Levels 1 and 2 | `hierarchy/` | 3-minute and 1-hour self-supervised levels on the frozen encoder | Code written and tested; not yet trained on the new split |
| 6. Level 3 and SOFA-2 | `hierarchy/` | 24-hour level with the SOFA-2 head, controls and endpoints | Code written and tested; not yet trained |

### 1. Cohort and labels

An anchor is a prediction time `t = intime + 24k` hours, `k ≥ 1`, in an adult ICU stay of a patient with a waveform record. The ICU day before `t` must have at least 384 of 480 valid three-minute slots.

| | Interval |
|---|---|
| Waveform input | `[t − 24h, t)` |
| Baseline | Worst SOFA-2 per organ in `[t − 24h, t)` |
| Target | Worst SOFA-2 per organ in `[t, t + 24h)` |
| Death and discharge in `[t, t + 24h)` | Main analysis: death scores 24, discharge 0, after de Grooth 2017. Other handlings are sensitivity analyses |

Patients are split 70/15/15 by `SHA256(subject_id)`. The test split is not read until the setting is locked. Attrition, splits and scoring: `docs/COHORT.md`. Endpoints: `labels/README.md`, section 7.

### 2. Waveform cut

Slot `k` is the 10 seconds from `intime + 10k` s, so 18 slots make 3 minutes and 8,640 make a day, aligned to `t`. ECG (one lead per segment), PPG and Resp are cleaned with NeuroKit2, resampled to 250 Hz and z-scored, as in the 10.14 preprocessing. Details: `docs/SETTING.md`, section 4.

### 3. Fine-tuning CSFM

CSFM-Base continues masked training with a new Resp encoder embedding, decoder embedding and reconstruction head: 75% of signal tokens masked, MSE on per-patch normalized targets, 60,000 updates, effective batch 256. Each arm's checkpoint is chosen by probes on train patients held out from fine-tuning.

| Arm | Text |
|---|---|
| A | None |
| B | Charted vitals, nursing assessments, chest imaging reports, ECG machine reports; whole word masking at 50% |
| B2 | As B without chest reports, respiratory nursing items only |
| B3 | As B2, masking values only |

Design and sources: `finetune/README.md`.

### 4. Encoder choice

| | Original CSFM | A | B | B2 | B3 |
|---|---:|---:|---:|---:|---:|
| Monitor RR MAE, three channels | 4.21* | **3.06** | 3.81 | 3.81 | 4.00 |
| Invasive support AUROC, three channels | 0.738* | **0.810** | 0.779 | 0.789 | 0.778 |
| Heart rate MAE | 3.98 | **3.46** | 4.33 | 4.47 | 4.32 |
| Retention checks lost, of 31 | Reference | 1 | 0 | 0 | 0 |

\* No Resp channel; ECG + PPG.

Arm A is best on every Resp task and on heart rate. The text arms gain only on findings that ECG machine text describes and tie with A elsewhere. The pre-specified rule would pick B, because A lost one check (sex from ECG alone, −0.032), so choosing A is reported as a decision made after the results. Reasons: `docs/SETTING.md`, "Encoder choice on the new split". Full tables: `encoder_comparison/results/report.md`.

### 5 and 6. Three levels and the SOFA-2 head

One model, the long-horizon CSFM extension of Tang et al. 2026, applied three times. Each level is trained on the frozen level below.

| Level | Tokens | Loss |
|---|---|---|
| 1, 3 minutes | 18 ten-second windows of arm A, with prompts and global positions | Masked window prediction + L_local + L_dyn |
| 2, 1 hour | 20 clips × 18 window features | Masked clip prediction + L_local + L_dyn |
| 3, 24 hours | 24 hours × 20 clip features | L_SOFA + 0.1 L_local + 0.1 L_dyn |

Masked prediction hides 75% of windows or clips. A light decoder predicts the frozen features of the level below, centered within their clip or hour, with MSE on the hidden ones only.

The level 3 head has a baseline path on the E2 inputs: baseline and previous-day organ scores, ICU day, and hourly heart rate, respiratory rate, SpO2 and MAP. A waveform correction from the day feature is added to the same logits. It outputs four ordinal CORN logits per organ and one composite logit. `L_SOFA` is the organ mean of the CORN losses plus the composite cross entropy.

**Endpoints**, on validation patients with 1,000 paired patient draws:

- Primary 1: composite AUROC. The event is total change ≥ 2 or death.
- Primary 2: mean AUROC over organs of organ worsening.
- Also reported: calibration, RPS and MAE.

**Controls:**

- `e0`: baseline scores only.
- `e2`: E2 only.
- `m`: E2 plus mean 10-second CLS.
- `wave_only`: level 3 without EHR.
- Persistence.

Each model is trained with three seeds. Design: `hierarchy/README.md`.

## Setup

Needs Python 3.11, [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU. `requirements.txt` pins every package, with PyTorch for CUDA 12.6.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Edit the paths in `env.sh`. In every new terminal, run:

```bash
source .venv/bin/activate
source env.sh
```

Check the install:

```bash
bash labels/run.sh test
bash waveform/icu_cut/run.sh test
bash finetune/arms/run.sh test
bash encoder_comparison/run.sh test
bash hierarchy/run.sh test
```

## Data

Credentialed PhysioNet access is required for all four MIMIC datasets.

| Item | Put it here |
|---|---|
| [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/) | `MIMIC_IV_DIR` in `env.sh` |
| MIMIC-IV Waveform Database v0.2 prerelease, with `v0.2_tmp_starttime_index.csv` | `MIMIC_WAVEFORM_DIR` in `env.sh` |
| [MIMIC-IV-Note 2.2](https://physionet.org/content/mimic-iv-note/2.2/), text arms only | `MIMIC_IV_NOTE_DIR` in `env.sh` |
| [MIMIC-IV-ECG 1.0](https://physionet.org/content/mimic-iv-ecg/1.0/), `machine_measurements.csv`, text arms only | `MIMIC_IV_ECG_DIR` in `env.sh` |
| CSFM-Base weights, from the [CSFM authors](https://github.com/guxiao0822/Cardiac-Sensing-FM) | `pretrained_model_for_share/CSFM_base.pth`. Link an existing copy with `ln -s /path/to/pretrained_model_for_share pretrained_model_for_share` |

No split file is needed. `plan` assigns every patient from `SHA256(subject_id)`, so a patient gets the same split on any machine.

## Run

Run from the repository root, in this order. Times are from our runs. The cut needs about 5.1 TB of disk. Use `tmux`, `nohup` or `setsid` for long steps, so that a closed terminal does not stop them.

**1. Cut the waveforms**

```bash
bash waveform/icu_cut/run.sh plan --workers 16      # 5 minutes
bash waveform/icu_cut/run.sh cut --workers 64       # 12 hours; rerun the same command to resume
```

**2. Labels, cohort and head inputs**

```bash
bash labels/run.sh extract      # 11 minutes
bash labels/run.sh score        # 2 minutes
bash labels/run.sh finalize     # 10 seconds; cohort counts in labels/cohort_report.json
bash labels/run.sh vitals       # hourly vital signs for E2
bash labels/run.sh e2           # E2 inputs, labels/private/e2.pkl
```

**3. Fine-tune CSFM**

```bash
bash finetune/arms/run.sh text                    # text records for arm B
bash finetune/arms/run.sh text --set signal       # text records for arms B2 and B3
bash finetune/arms/run.sh check                   # confirms the reconstruction target
GPUS=0,1,2,3 bash finetune/arms/run.sh train A
GPUS=4,5,6,7 bash finetune/arms/run.sh train B
OUTPUT=$PWD/finetune/arms/runs/arm_B2 bash finetune/arms/run.sh train B --text-dir $PWD/finetune/arms/text_signal
OUTPUT=$PWD/finetune/arms/runs/arm_B3 bash finetune/arms/run.sh train B --text-dir $PWD/finetune/arms/text_signal --text-mask values
```

With fewer GPUs, keep GPUs × batch × accumulation at 256, for example `GPUS=0 ... train A --batch-size 32 --accumulation 8`. A stopped run resumes from `last.pt`.

Choose each arm's checkpoint. This needs the comparison's windows and labels:

```bash
python encoder_comparison/label_audit/extract_chart.py      # 25 minutes
bash encoder_comparison/run.sh labels                       # 6 minutes
GPUS=0 bash finetune/arms/run.sh select A                   # writes runs/arm_A/selected.pt
GPUS=0 OUTPUT=$PWD/finetune/arms/runs/arm_B2 bash finetune/arms/run.sh select B
```

**4. Compare encoders**

```bash
GPU=0 bash encoder_comparison/run.sh features
bash encoder_comparison/run.sh probes           # results in encoder_comparison/results/
```

The chosen encoder is `ENCODER` in `hierarchy/common.py`, now `finetune/arms/runs/arm_A/selected.pt`.

**5. Levels 1 and 2**

```bash
bash hierarchy/run.sh index                     # valid clips on the new split; rerun after labels change
GPUS=3,7 bash hierarchy/run.sh three_min        # level 1
GPUS=3,7 bash hierarchy/run.sh features         # level 1 features, about 150 GPU hours; resumes by ICU day
bash hierarchy/run.sh check                     # signal check; read it before going on
GPUS=3 bash hierarchy/run.sh one_hour           # level 2 and its features
```

**6. Level 3, controls and endpoints**

```bash
GPUS=3 bash hierarchy/run.sh day                # e0, e2, m, hierarchy, wave_only; seeds 42, 43, 44
bash hierarchy/run.sh endpoints                 # hierarchy/private/runs/day/main/endpoints.json
```

For a sensitivity analysis, add `--handling` with `locf`, `partial_discharge`, `no_death` or `stayed_only` to both commands.

## Layout

```
labels/                    1. SOFA-2 labels and cohort
  extract/                   MIMIC-IV items
  score/                     day scores, vasopressor runs, anchors
  e2/                        EHR inputs of the level 3 head
waveform/                  2. Waveform cut
  native_index/              WFDB headers and clock checks
  windows_10s/               read and clean one 10-second window
  icu_cut/                   ICU time grid, plan and cut
  coverage_audit/            where ICU days are lost
finetune/                  3. CSFM fine-tuning
  csfm_mae/                  model, tokenizer and its assets
  arms/
    build_text/              text records of the text arms
    check/                   reconstruction-target check
    train/                   windows, masking and training
    select/                  checkpoint choice by probes
    tests/
encoder_comparison/        4. Encoder choice
  probe_labels/              windows and probe labels
  label_audit/               label source checks
  features/                  frozen CLS features
  probes/                    XGBoost, bootstrap and decision
  results/                   aggregate report
hierarchy/                 5 and 6. Three levels and the SOFA-2 head
  data/                      clip index, day arrays, targets, head inputs
  models/                    level models, losses and the SOFA-2 head
  level1_three_min/          level 1 training, features, signal check
  level2_one_hour/           level 2 training and features
  level3_day/                level 3, controls and endpoints
  tests/
docs/                      cohort summary and full setting
```

Each step's `run.sh` sets its Python paths, so run steps through it. Patient data, weights and outputs stay in each step's `private/`, `runs/`, `output/` or `text*/`. Git ignores them.

## Documents

| File | Content |
|---|---|
| `docs/COHORT.md` | Cohort definition, attrition, splits, missingness, SOFA-2 scoring |
| `docs/SETTING.md` | Full method with tags `[source]`, `[engineering]`, `[extension]`, `[open]`, and every dated change |
| `labels/README.md` | Labels, death and discharge handling, endpoints, label statistics |
| `finetune/README.md` | Fine-tuning: the CSFM paper's method, ours, data, differences, arms B2 and B3, results |
| `encoder_comparison/README.md` | Comparison method, decision and limits |
| `hierarchy/README.md` | The three levels and the SOFA-2 head: paper, ours, differences, evaluation |
| `SOURCES.tsv` | Where each file came from, and what changed |

## Patient data

Never commit or share `labels/private/`, `encoder_comparison/private/`, `encoder_comparison/label_audit/private/`, `hierarchy/private/`, `finetune/arms/text*/`, `finetune/arms/runs/` or waveform outputs. `.gitignore` excludes them. MIMIC data is covered by the PhysioNet data use agreement.

Some validation and test patients of the current split were validation patients of an earlier split, so we have seen earlier results on them. No model was trained on them, but the test split is not fully unseen by us.

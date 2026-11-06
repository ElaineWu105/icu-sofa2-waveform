# Fine-tuning CSFM to add a Resp channel

This folder adapts the Cardiac Sensing Foundation Model (CSFM, Gu et al. 2026) so that it accepts a third signal, the impedance respiration waveform (Resp), next to ECG and PPG. It keeps the masked-modelling objective of the CSFM paper.

This page covers what the paper does, what we do, which data we use, where we depart from the paper and why, and the options we considered and did not take.

**Status.** Arms A, B, B2 and B3 were trained on the new split with the design in sections 3 to 5 and compared on 2026-11-06. Arm A is the chosen encoder (end of section 3.10). Section 7 describes an earlier run on the earlier split.

Tags: `[source]` taken from a cited work; `[engineering]` our choice with no scientific source; `[extension]` our own method extension, to be stated as such in any paper; `[open]` not yet fixed.

## Contents

- [Fine-tuning CSFM to add a Resp channel](#fine-tuning-csfm-to-add-a-resp-channel)
  - [Contents](#contents)
  - [1. What the CSFM paper does](#1-what-the-csfm-paper-does)
  - [2. Data](#2-data)
  - [3. Our method](#3-our-method)
    - [3.1 Arms](#31-arms)
    - [3.2 Signal input](#32-signal-input)
    - [3.3 Signal masking and target](#33-signal-masking-and-target)
    - [3.4 Text sources](#34-text-sources)
    - [3.5 When a window has text](#35-when-a-window-has-text)
    - [3.6 Fitting text into 64 tokens](#36-fitting-text-into-64-tokens)
    - [3.7 Text masking](#37-text-masking)
    - [3.8 Loss and optimization](#38-loss-and-optimization)
    - [3.9 Checkpoint selection](#39-checkpoint-selection)
    - [3.10 Arms B2 and B3](#310-arms-b2-and-b3)
  - [4. Differences from the paper](#4-differences-from-the-paper)
  - [5. Options considered and not used](#5-options-considered-and-not-used)
  - [6. Evaluation](#6-evaluation)
  - [7. The earlier run and what it showed](#7-the-earlier-run-and-what-it-showed)
  - [8. Implementation status](#8-implementation-status)
  - [9. Known limits](#9-known-limits)
  - [10. Files](#10-files)
  - [11. References](#11-references)

## 1. What the CSFM paper does

All statements below come from the published Methods and figures of Gu et al., *Nature Machine Intelligence* 8, 220–233 (2026) [1]. The Supplementary Information was not checked.

| Item | Paper |
|---|---|
| Pretraining data | MIMIC-III-WDB: 270,562 segments, 7,425 individuals. MIMIC-IV-ECG: 787,677 segments, 160,821 individuals. CODE-Full: 1,558,748 segments, one per individual (Fig. 1b) |
| Signals | 10-second segments. MIMIC-III-WDB gives lead II ECG and PPG, taken every 6 hours of each record; MIMIC-IV-ECG and CODE give diagnostic 12-lead ECG. "Basic filtering and z-score normalization of each biomedical signal" |
| Text | MIMIC-III-WDB is paired with cardiologist ECG reports from the MIMIC-III notes "within 1 week of each waveform segment", which the authors call a "weak match". MIMIC-IV-ECG is paired with machine-generated reports. CODE has no text. Text exists for 224,205 MIMIC-III and 787,677 MIMIC-IV segments (Fig. 1b) |
| Signal tokens | Non-overlapping 0.1-second patches, a linear projection, plus learnable channel embeddings (12 ECG leads and PPG) and temporal embeddings |
| Text tokens | WordPiece IDs from a pretrained vocabulary, 64 tokens with padding, a learnable embedding plus a text type embedding and a 1D positional embedding |
| Input, Eq. 1 | `[CLS; signal]` when no report exists and `[CLS; signal; text]` when one does. Samples of both kinds are trained together |
| Missing channels, Fig. 6b | All available tokens are flattened, and only available modalities are reconstructed |
| Masking | "we randomly mask out 75% tokens from biosignals and 50% tokens from text", following the Masked Autoencoder [2] |
| Encoder and decoder | A ViT-style transformer encoder over the visible tokens. A light decoder (MLP and shallow transformer blocks) receives the encoded tokens plus mask tokens with channel and position embeddings |
| Loss | Eq. 2: L2 between true and reconstructed masked signal patches. Eq. 3: negative log-likelihood of the masked text tokens. The loss uses masked positions only |
| Optimizer | AdamW, weight decay 0.05, betas (0.9, 0.95). Learning rate 1e-3 and batch 256 for CSFM-Base |
| Sampling | "uniform sampling from different datasets" to form each minibatch |
| Schedule | Up to 25,000,000 steps, "early stopping based on validation performance every 2,500 steps", cosine schedule with 5% warmup |
| Validation split | 80% train and 20% validation, with no individual in both |
| Downstream readout | The CLS token after the last encoder layer |
| Downstream use | Either frozen features with logistic regression, random forest or XGBoost, or fine-tuning with a learning rate of 1e-4 and layer-wise decay 0.75. Three random seeds per result |

What the paper does not state: the reconstruction target's normalization, the weights of Eq. 2 and Eq. 3, the exact tokenizer, and the validation metric used for early stopping. The paper has no ablation that removes text.

Our own estimate, not stated in the paper: with uniform sampling over the three datasets, about 61% of training samples carry text. This is (224,205 / 270,562 + 1 + 0) / 3.

## 2. Data

| Source | Version | Used for | Notes |
|---|---|---|---|
| MIMIC-IV Waveform Database [3] | v0.2 prerelease, with `v0.2_tmp_starttime_index.csv` | ECG, PPG and Resp signals | Not public. The public release is v0.1.0 [3]. Cutting is described in `docs/SETTING.md`, section 4 |
| MIMIC-IV [4, 5] | 3.1 | ICU stays, splits, charted vital signs and nursing assessments (`icu/chartevents`) | |
| MIMIC-IV-Note [6] | 2.2 | Chest imaging reports (`note/radiology`, exam names from `radiology_detail`) | |
| MIMIC-IV-ECG [7] | 1.0 | ECG machine reports (`machine_measurements.csv`, `report_0` to `report_17`) | Diagnostic 12-lead ECGs; only the report text is used |
| CSFM-Base weights [1] | `CSFM_base.pth` from the authors' repository | Start point, encoder and decoder | |

All of these need credentialed PhysioNet access [8]. No patient-level file is kept in this repository.

**Patients.** All encoder training uses train patients only. Patients are split 70/15/15 by `SHA256(subject_id)`, see `docs/SETTING.md`, section "Split". In addition, 10% of train patients are held out from fine-tuning to select the checkpoint (section 3.9).

**ICU days.** All cut ICU days of the cohort are used. How they were cut is in `docs/SETTING.md`, section 4.

## 3. Our method

### 3.1 Arms

| Arm | Text | Windows |
|---|---|---|
| A | Never | One random valid 10-second window per ICU hour, redrawn each epoch |
| B | Text when a record exists near the window, as in Eq. 1 `[source]` | The same windows as arm A |
| B2 | As B, with the `signal` text set: no chest imaging, respiratory nursing items only (section 3.10) | The same windows as arm A |
| B3 | As B2, and charted text masks values only (section 3.10) | The same windows as arm A |

The two arms see the same windows, so they differ only in text `[engineering]`. Arm A matches the paper's signal-only samples. Arm B matches the paper's mix of samples with and without text. The mix comes from the data, not from a fixed ratio.

### 3.2 Signal input

| Item | Setting | Tag |
|---|---|---|
| Length | 10 seconds | `[source]` |
| Sampling | 250 Hz. PPG (native 125 Hz) and Resp (native 62.5 Hz) are FFT-resampled to 250 Hz | `[engineering]`; the paper's rate is not in the main text |
| Cleaning | NeuroKit2 [9]: `ecg_clean` (default method), `ppg_clean` (Elgendi et al. 2013 [10]), `rsp_clean` (Khodadad et al. 2018 [11]) | `[engineering]`; the paper says only "basic filtering" |
| Normalization | z-score per channel per window | `[source]` |
| Patches | 0.1 s, 25 samples, 100 per channel | `[source]` |
| ECG lead | One monitor lead per segment, by priority II, I, III, aVR, aVL, aVF, V1 to V6, with its own channel embedding | `[source]` for channel embeddings |
| PPG | Channel embedding 12, as in the checkpoint | `[source]` |
| Resp | New channel 13: a new encoder channel embedding, decoder channel embedding and reconstruction head | `[extension]` |
| Missing channels | Not filled; only available channels are tokenized and reconstructed | `[source]` Fig. 6b |

### 3.3 Signal masking and target

| Item | Setting | Tag |
|---|---|---|
| Masking | Random 75% of all signal tokens of a window, after flattening its channels | `[source]` |
| Target | Per-patch normalized patch, `(x − mean) / sqrt(var + 1e-6)` | Inferred from the checkpoint, see below |
| Loss | Mean squared error over masked patches of all channels, Eq. 2 | `[source]` |

The target normalization is not written in the paper. We checked it on the released checkpoint (`arms/check/check_mae.py`). On 512 ICU windows, the original CSFM's masked MSE was 0.92 against raw patches and 0.34 against per-patch normalized patches for ECG, and 1.40 against 0.18 for PPG, where predicting zero scores 1.0. The decoder therefore reconstructs normalized patches, and we train Resp with the same target. The MAE forward pass is not public, so ours is a reconstruction.

### 3.4 Text sources

| Source | Content | Tag |
|---|---|---|
| Charted vital signs (`chartevents`) | Heart rate 220045, heart rhythm 220048, ectopy type 224650, SpO2 220277, noninvasive mean BP 220181, respiratory rate 220210, breathing pattern 229322, O2 delivery device 226732, ventilator mode 223849 and 229314, FiO2 223835, PEEP 220339, tidal volume 224685 | `[extension]` |
| Charted nursing assessments (`chartevents`, text items) | Respiratory pattern 223985, respiratory effort 223990, current dyspnea assessment 229323, accessory muscle use 224723, lung sounds RUL 223986, RLL 223987, LUL 223988, LLL 223989, cough effort 223991, cough type 223992, sputum consistency 224369, colour 224370 and amount 224373, airway type 223836; heart sounds 224389; level of consciousness 226104, orientation 223898, follows commands 223910 | `[extension]` |
| Chest imaging reports (MIMIC-IV-Note) | The IMPRESSION section of reports whose exam name contains "CHEST", such as portable chest radiographs and chest CT. 96% of these reports have an IMPRESSION section | `[extension]` |
| ECG machine reports (MIMIC-IV-ECG) | `report_0` to `report_17` joined in order, such as "Demand pacing. Anteroseptal infarct - age undetermined. Abnormal ECG." | Same text type as the paper's MIMIC-IV-ECG reports `[source]` |

Excluded:

- **Discharge summaries:** they cover a whole admission and hold its outcome.
- **ICD codes:** they are probe labels in the evaluation.
- **Medication and procedure names, and GCS items:** they are inputs of SOFA-2 scoring.

**Format of charted text.** One record is every selected item charted at one time for one stay, written `<label>: <value>.` in the field order above, as in the earlier run. An item with two different values at one time is left out. Vital-sign records need heart rate, respiratory rate or SpO2. Report-style descriptions, such as "tachypnea" for respiratory rate above 20, remain an option `[open]`.

Chest reports keep the IMPRESSION section up to the next section heading, with de-identification blanks (`___`) removed. ECG reports join the non-empty report lines with full stops. Text is read for train patients only, from one day before ICU admission to one day after discharge. Code: `arms/build_text/prepare_text.py`.

### 3.5 When a window has text

Each source is matched to a window by the time between the window and the record. The allowed time follows how long the described state lasts `[engineering]`.

| Source | Allowed time | Reason |
|---|---|---|
| Vital signs and nursing assessments | ±15 minutes | Bedside measurements change within minutes |
| Chest imaging report | ±12 hours | Effusion, edema and consolidation change over hours to days |
| ECG machine report | ±24 hours | Rhythm, conduction and infarct statements persist for days. The paper pairs MIMIC-III ECG reports within one week |

Measured on 60,000 random training windows, one per ICU hour, of the earlier plan:

| Source | Windows with a match |
|---|---:|
| Vital signs, ±15 minutes | 52.4% |
| Chest imaging report, ±12 hours | 55.4% |
| ECG machine report, ±24 hours | 20.2% |
| **At least one source** | **82.4%** |

Of windows with text, 53% have one source, 39% have two and 8% have all three. Coverage of nursing assessments is not yet measured.

For comparison, other allowed times for vital signs give: ±10 minutes 35.8%, ±30 minutes 97.4%. For chest reports: ±6 hours 30%, ±24 hours 75%. For ECG reports: ±12 hours 12%.

### 3.6 Fitting text into 64 tokens

The paper uses 64 text tokens, and the checkpoint has 64 text positions, so the length cannot grow without new parameters.

Token lengths on our cohort, with the BERT uncased tokenizer, including `[CLS]` and `[SEP]`:

| Text | Median | 90th percentile | Over 64 |
|---|---:|---:|---:|
| ECG machine report | 29 | 53 | 3% |
| Chest imaging IMPRESSION | 56 | 114 | 41% |
| Charted vital signs, earlier format | about 42 | | |

**Rule** `[engineering]`: each sample carries the text of **one** source. The source is drawn uniformly from those available for that window, and the draw is redone each epoch. A text longer than 64 tokens is cut at the last whole word that fits, so whole word masking never meets half a word.

This follows the paper's pairing of one report per sample. Joining all sources would let the chest report fill the 64 tokens on its own in 41% of cases.

### 3.7 Text masking

Text masking uses BERT whole word masking [12, 13] `[source]`, at CSFM's rate. Code: `arms/train/finetune.py`, `whole_word_mask`. It was checked against `create_pretraining_data.py` in google-research/bert, commit `eedf571`.

**As in BERT:**

1. Candidates are all tokens except `[CLS]` and `[SEP]`. Punctuation counts as a word.
2. A token starting with `##` joins the word before it, so "o2 saturation pulseoxymetry" becomes "o ##2", "sat ##uration", "pulse ##ox ##yme ##try".
3. Words are shuffled.
4. The budget is `max(1, round(n × rate))`, where `n` counts `[CLS]` and `[SEP]` and `round` is Python's round half to even.
5. Words are taken in shuffled order. A word that would pass the budget is skipped, and the search goes on.
6. Every masked piece is predicted on its own over the whole vocabulary.

**Different from BERT, following CSFM:**

| BERT | Here | Why |
|---|---|---|
| Rate 15% | 50% | CSFM's text masking rate [1] |
| At most 20 predictions per sequence | No cap | The cap belongs to BERT's 15% and 128-token setting; CSFM has none |
| Chosen tokens replaced 80% `[MASK]`, 10% unchanged, 10% random | Chosen tokens dropped from the encoder input | CSFM is a Masked Autoencoder: the encoder never sees masked tokens, and the decoder fills them with mask tokens [1, 2] |

On 2,000 earlier charted texts, about 40 content tokens each, this masks 53% of content tokens. The share is above 50% because the budget counts `[CLS]` and `[SEP]`.

### 3.8 Loss and optimization

| Item | Setting | Tag |
|---|---|---|
| Loss | Signal MSE (Eq. 2) plus text cross entropy (Eq. 3), weights 1 and 1. Arm A has no text term | `[source]` for the terms; the weights are `[engineering]`, as the paper gives none |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.05 | `[source]` |
| Learning rate | 1e-4 with layer-wise decay 0.75. New Resp parameters use the base rate | The paper's downstream fine-tuning setting `[source]`; using it for continued masked training is `[engineering]` |
| Effective batch | 256 | `[source]` |
| Length | 60,000 updates for both arms, about 40 epochs at the earlier window count | `[engineering]` |
| Schedule | Cosine with 5% warmup, decaying to zero at the last update. No early stopping | Warmup and cosine `[source]`; the fixed length `[engineering]` |
| Seeds | One first; the paper reports three for downstream results | `[engineering]` |

### 3.9 Checkpoint selection

The paper stops early "based on validation performance". In our earlier run, arm B had the lower Resp reconstruction error but the worse downstream Resp results (section 7), so reconstruction error is logged but not used for selection `[engineering]`.

| Item | Setting |
|---|---|
| Selection patients | 10% of train patients, held out from fine-tuning. Validation patients are used only for the final comparison |
| When | Every 2,500 updates `[source]` |
| Input | Signals only, no text, for both arms, so text cannot drive selection |
| Features | CLS of one window per ICU hour on the selection patients |
| Probes | Ridge and logistic regression on standardized CLS for monitor respiratory rate, invasive respiratory support, SpO2, SBP, DBP, heart rate and rhythm. Fitted on one half of the selection patients and scored on the other, then the halves swap; the two scores are averaged. Windows and labels are the encoder comparison's, one per ICU hour of each patient's first qualifying day |
| Rule, fixed before training | Keep checkpoints whose ECG and PPG tasks are not worse than original CSFM by more than the encoder comparison margins: rhythm macro-F1 0.02 (ECG), heart rate 1 bpm (ECG), SpO2 0.25 (PPG), SBP and DBP 1 mmHg (ECG + PPG). Among those, take the lowest respiratory rate error with all three channels. If none qualifies, nothing is chosen and the study team decides `[open]` |
| Code | Checkpoints: `arms/train/finetune.py`, every 2,500 updates. Choice: `arms/select/choose_checkpoint.py`, which writes `selection.json` and copies the chosen checkpoint to `selected.pt` |

### 3.10 Arms B2 and B3

**Why.** Arm B lost to arm A on respiratory rate and breathing pattern. A check on 1,200 windows of the held-out selection patients, with the final arm A and B models, showed where B's text went (2026-11-05; the script is not in the repository):

| Finding | Value |
|---|---|
| Text drawn for B's training windows | Chest imaging 34%, vital signs 32%, none 18%, ECG reports 11%, nursing 6% |
| Share of B's text loss from chest imaging words | About 87% |
| Text cross entropy, true signal vs the signal of another window | Chest imaging 1.81 vs 1.81; ECG reports 0.30 vs 0.37; vital-sign values 1.75 vs 2.36 |
| Vital-sign field names, such as "heart rate :" | 32% of masked text tokens, 99.7% predicted correctly |
| Vital-sign values | 4.5% of masked text tokens |
| Masked-signal error without text, A vs B | ECG 0.286 vs 0.314, PPG 0.137 vs 0.153, Resp 0.269 vs 0.273; giving B its text changes none of them |

So most of B's text loss is a language model of chest reports that does not use the signal, while the one part that does, vital-sign values, is small. Text helps a signal encoder when it describes that signal [1, 14, 15].

**Text set `signal`** (`prepare_text.py --set signal`, folder `arms/text_signal/`):

| Source | Kept | Reason |
|---|---|---|
| Vital signs | All 13 fields | Rate, rhythm, ectopy, saturation, pressure and respiratory support are visible in, or act on, the ECG, PPG and Resp window |
| Nursing | Respiratory pattern, respiratory effort, dyspnea, accessory muscle use, airway type | Breathing pattern and airway act on the Resp signal |
| Nursing, dropped | Lung sounds, cough, sputum, heart sounds, consciousness, orientation, commands | Not visible in the window |
| ECG machine reports | Kept | The CSFM paper's own text type [1] |
| Chest imaging | Dropped | Not visible in the window; the check above shows the signal is not used to predict it |

Allowed times, the one-source rule and the 64-token limit are unchanged. The rule for keeping a field is what the window shows, not which comparison task it matches. Several kept fields are also comparison labels (section 6), so arms B2 and B3 receive weak supervision on those tasks, as CSFM's ECG reports do for its diagnosis tasks [1].

**Value masking (arm B3)** `[engineering]`. In vital-sign and nursing text, only values are masked, each value whole, and field names stay visible. `max(1, round(values x 0.5))` values are drawn at random, so a one-token number and a multi-token rhythm name are equally likely to be hidden. ECG reports keep whole word masking. Code: `arms/train/finetune.py`, `value_mask`. Value positions come from `prepare_text.py`, stored as `value` in `records.npz`.

**Reading, fixed before the runs:**

| Comparison | Question |
|---|---|
| B2 against B | Does removing text the signal cannot explain help? |
| B3 against B2 | Does masking values only help? |
| B2 and B3 against A | Does text help at all? |

**Runs.** Both ran the full 60,000 updates with effective batch 256 and finished on 2026-11-06. Both were stopped once at update 15,000 when the launching terminal closed, and resumed from `last.pt`. B2 shared a GPU with another user's job, so at update 23,840 it was stopped with the stop file and resumed on one GPU with accumulation 8 instead of two GPUs with accumulation 4. The learning-rate schedule follows the update count and the effective batch is the same, so the schedule is unchanged. The window order after each resume differs from an uninterrupted run.

**Result and choice, 2026-11-06.** Arm A is the chosen encoder. Chosen checkpoints and their held-out masked error:

| Arm | Chosen update | ECG / PPG / Resp error | Monitor RR MAE, three channels | Invasive support AUROC, three channels |
|---|---:|---|---:|---:|
| A | 37,500 | 0.283 / 0.135 / 0.260 | 3.06 | 0.810 |
| B | 35,000 | 0.311 / 0.150 / 0.263 | 3.81 | 0.779 |
| B2 | 37,500 | 0.297 / 0.143 / 0.261 | 3.81 | 0.789 |
| B3 | 12,500 | 0.315 / 0.155 / 0.294 | 4.00 | 0.778 |

RR and support are from the encoder comparison on validation patients (`encoder_comparison/results/report.md`); original CSFM on ECG + PPG scores 4.21 and 0.738. Every text arm is worse than A on every Resp task and on heart rate. They are better than A only on findings that ECG machine text describes (right bundle branch block, first-degree block, QTc, PAC), and tie with A on text-free tasks. Giving the final text arms their text does not change their signal reconstruction. B3's Resp error rose after update 12,500. Why A is chosen although the pre-specified rule picks B is in `docs/SETTING.md`, "Encoder choice".

All arms are compared with the same encoder comparison and checkpoint rule (sections 3.9 and 6). Tasks whose labels come from the same sources as the text are reported apart from tasks that share nothing with it (ICD groups, deaths, age, sex and BMI). The first group is heart rate, saturation, rhythm, ectopy, breathing pattern, respiratory rate and support from charted items, blood pressure because non-invasive mean pressure is charted text, and QTc, first-degree block and bundle branch blocks because ECG machine text describes them. Code: `TEXT_FREE_TASKS` in `encoder_comparison/common.py`. The masked-signal error and the true-versus-other-signal text check are repeated on the final models.

## 4. Differences from the paper

| Item | Paper | Here | Reason |
|---|---|---|---|
| Training stage | Pretraining from scratch | Continued masked training from CSFM-Base | We add one channel and keep what CSFM learned `[extension]` |
| Channels | 12 ECG leads and PPG | One monitor ECG lead, PPG and **Resp** | Resp is the new channel `[extension]` |
| Signal data | MIMIC-III-WDB segments every 6 hours; diagnostic 12-lead ECGs | One window per ICU hour from MIMIC-IV-WDB v0.2 | Our cohort and later tasks are ICU-hourly `[engineering]` |
| Learning rate | 1e-3 for pretraining | 1e-4 with layer-wise decay 0.75 | Smaller steps on a pretrained model, as in the paper's own fine-tuning `[engineering]` |
| Stopping | Early stopping on validation | 60,000 updates with cosine decay to zero | In the earlier run the noisy Resp error stopped both arms with the learning rate still at 6e-5 to 8e-5 |
| Selection | "Validation performance" | Probes on held-out train patients | Reconstruction error did not track downstream ability; validation patients stay clean |
| Text | ECG reports | Charted vitals, nursing assessments, chest imaging reports and ECG machine reports | Only 4,806 ECG reports fall inside ICU waveform time; Resp needs respiratory text `[extension]` |
| Text timing | Within 1 week, or the paired ECG | ±15 minutes, ±12 hours or ±24 hours by source | Charted values change within minutes `[engineering]` |
| Which samples get text | Set by data source, about 61% with uniform sampling | Set by time match, 82.4% | Follows Eq. 1 on our data |
| Text packing | One report per sample | One source per sample, drawn at random | Same idea; several sources may be available |
| Text masking unit | Random tokens | BERT whole words | Pieces of a split word are trivial to recover from each other [12] |
| Tokenizer | WordPiece from an unnamed pretrained vocabulary | BERT uncased | The authors have not confirmed the mapping; an assumption |
| Target normalization | Not stated | Per-patch normalized | Inferred from the checkpoint (section 3.3) |
| Loss weights | Not stated | 1 and 1 | No source |
| Seeds | Three for downstream | One first | Compute; more seeds after the first result |

## 5. Options considered and not used

| Option | What it would do | Why not now |
|---|---|---|
| Whole-channel masking | Hide a whole channel, for example all of Resp, and rebuild it from the others | Not in the paper. Its "channel-wise" masking comes from datasets with different channels (Fig. 6b). Candidate ablation |
| Window z-score Resp target with 1 to 2 second block masking | Keeps Resp amplitude and stops neighbour interpolation | Not the pretrained decoder's convention. With the current target, linear interpolation of masked Resp scores 0.83 of zero prediction under random masking, against 0.24 for arm A, so the task is not trivial. Interpolation reaches 0.26 under a window z-score target, so that target would need block masking (1.24 with 1-second blocks, 1.59 with 2-second blocks). Candidate ablation |
| Joining all text sources, or fixed token shares per source | More context per sample | Truncation; chest reports alone exceed 64 tokens in 41% of cases |
| Monitor numerics as text | Respiratory rate from the same 10 seconds | Probe labels come from the same numerics, so evaluation would test the training target |
| Numbers as regression or ordered bins | 18 and 19 become close | Departs from Eq. 3. Candidate if numeric fields stay |
| Field- or organ-group masking | Remove shortcuts between related fields | Departs from BERT and CSFM. Candidate ablation |
| Selecting on validation patients | The reviewer's proposal | The same patients would choose the checkpoint and judge it |
| Patch-mean readout | Mean of patch tokens | The paper reads CLS; the three training levels move to CLS as well |
| A fixed text share, such as 61% | Match the paper's estimated share | The paper's share comes from its data sources, not from a set ratio |

## 6. Evaluation

On validation patients, with frozen CLS features and XGBoost probes as in `encoder_comparison/`, and a paired patient bootstrap of 1,000 draws.

The tasks are kept for their link to SOFA-2 organs `[engineering]`:

| Task | Inputs | Link |
|---|---|---|
| Monitor respiratory rate | All three; Resp alone | Respiratory |
| Invasive respiratory support | All three; Resp alone | Respiratory |
| ICD respiratory diagnoses | All three; Resp alone | Respiratory; not part of any training text |
| SpO2 | PPG | Respiratory, through SpO2/FiO2 |
| SBP and DBP | ECG, PPG, both | Cardiovascular, through MAP |
| Rhythm, heart rate | ECG | Cardiovascular state |
| 24-hour and hospital death | All | Deterioration |
| Resp rebuilt from ECG + PPG, with Resp fully hidden | ECG + PPG | Whether Resp is learned with the other channels; like the paper's cross-modal reconstruction. Metric details `[open]` |

The questions:

1. **Resp helps.** Each arm with all three channels against original CSFM with ECG + PPG.
2. **Text helps.** Arm B against arm A, paired.
3. **Resp is learned.** Respiratory tasks, ICD respiratory diagnoses and the rebuilt Resp.

Multiplicity correction is `[open]`; Benjamini–Hochberg over the retention checks has been proposed.

## 7. The earlier run and what it showed

These arms were trained on the earlier split, with the settings in `docs/SETTING.md`, section 5.

| | Arm A | Arm B |
|---|---|---|
| Text | None | Charted vital signs on every sample, `<label>: <value>.`, random 50% piece masking |
| Windows | One random valid window per ICU hour, 369,685 | The window holding a charted snapshot, 365,153 |
| Selected and stopped update | 22,500 and 47,500 | 40,000 and 65,000 |
| Validation Resp MSE at selection | 0.2427 | 0.2266 |
| Monitor respiratory rate MAE, three channels | 3.14 | 4.06 |
| Invasive support AUROC, three channels | 0.805 | 0.780 |

Original CSFM on ECG + PPG scores 4.29 and 0.720 on the same two tasks. Arm B was worse than arm A on respiratory rate by 0.92 breaths per minute (95% CI 0.83 to 1.01).

To see whether arm B linked text to the waveform, we masked only the value of one field on 4,000 validation windows and asked arm B to predict it with or without the waveform:

| Field | Waveform and field name only | Field name only | Constant guess |
|---|---:|---:|---:|
| Heart rate, MAE | 3.8 | 15.4 | 14.2 |
| Respiratory rate, MAE | 8.6 | 8.0 | 5.0 |
| O2 delivery device, exact | 0.50 | 0.77 | |

Arm B learned to read heart rate from ECG. Original CSFM cannot fill this template at all, so this was learned in fine-tuning. Arm B did not link respiratory rate to Resp: its guesses correlate 0.23 with the truth and average 15.2 against a true 20.6.

Charted respiratory rate also differs from the monitor value in the same 10 seconds by 2.4 breaths per minute on average (r = 0.75). A spectral rate from the 10-second Resp signal alone has an error of 4.8.

These results led to:

- the Eq. 1 mix, since arm B never trained without text but is always used without it;
- the broader respiratory text;
- whole word masking;
- the same windows for both arms;
- probe-based selection.

## 8. Implementation status

| Part | State |
|---|---|
| New split and all cut ICU days in fine-tuning and the comparison | In the code |
| Whole word masking | In the code, `arms/train/finetune.py`, with tests |
| Arm B with the Eq. 1 mix and arm A's windows | In the code, `arms/train/data.py`, with tests |
| Nursing assessments, chest imaging and ECG report text; time matching; one-source packing | In the code, `arms/build_text/prepare_text.py` and `arms/train/data.py`, with tests |
| Fixed 60,000-update schedule without early stopping | In the code, `arms/train/finetune.py` |
| Arms B2 and B3: `signal` text set and value masking | In the code, `arms/build_text/prepare_text.py`, `arms/train/data.py` and `arms/train/finetune.py`, with tests; runs and comparison done 2026-11-06 |
| Held-out selection patients and probe-based selection | In the code, `arms/train/data.py` and `arms/select/choose_checkpoint.py`, with tests |
| CLS readout in `hierarchy/` | To do, with the redesign of the three levels |
| Resp rebuilt from ECG + PPG in the evaluation | To do |

## 9. Known limits

- **Tokenizer:** BERT uncased is an assumption. The authors have not confirmed it.
- **MAE forward pass:** reconstructed from the paper and the released encoder, because it is not public.
- **Weak text match:** text is close in time to the window but does not describe that window.
- **Shared patients with pretraining:** CSFM was pretrained on MIMIC-IV-ECG, which shares patient IDs with our cohort, so the original model may have seen 12-lead ECGs of some of our validation and test patients. That exposure had no labels and holds for every split.
- **ECG mains filter:** ECG cleaning uses NeuroKit2's default 50 Hz powerline filter, kept from the 10.14 preprocessing, while US mains is 60 Hz. The effect on the 0.5–35 Hz monitor ECG is not measured.
- **ECG leads:** one monitor lead per segment, so 12-lead ability is not trained or tested.
- **SOFA-2 inputs in text:** charted FiO2 and ventilator mode, part of the vital-sign text, are also SOFA-2 respiratory inputs. Text is used only on training patients and never as model input downstream.

## 10. Files

| File | Role |
|---|---|
| `csfm_mae/model.py` | CSFM masked autoencoder with the Resp channel |
| `csfm_mae/tokenizer_utils.py` | Local BERT uncased tokenizer and its provenance |
| `arms/run.sh` | `text`, `check`, `test`, `train A`, `train B`, `select A` and `select B`; `GPUS`, `OUTPUT` and `PORT` override the defaults |
| `arms/common.py` | Paths shared by the steps below |
| `arms/build_text/prepare_text.py` | Text records: all four sources in `text/records.npz` (arm B), or the `signal` set in `text_signal/records.npz` (arms B2 and B3) |
| `arms/check/check_mae.py` | Confirms the reconstruction target on the checkpoint; result in `check_mae.json` |
| `arms/train/data.py` | Hourly windows from the cut ICU days |
| `arms/train/finetune.py` | Training and masking; saves a checkpoint every 2,500 updates |
| `arms/select/choose_checkpoint.py` | Probes on held-out train patients; writes `selection.json` and `selected.pt` |
| `arms/tests/test_finetune.py` | Tests |
| `arms/runs/` | Checkpoints of each arm; `arms/runs/earlier_split/` holds the earlier arms, not used by the rerun |

## 11. References

1. Gu X, Tang W, Han J, et al. Cardiac health assessment across scenarios and devices using a multimodal foundation model pretrained on data from 1.7 million individuals. *Nature Machine Intelligence* 8, 220–233 (2026). https://doi.org/10.1038/s42256-026-01180-5. Code and weights: https://github.com/guxiao0822/Cardiac-Sensing-FM
2. He K, Chen X, Xie S, Li Y, Dollár P, Girshick R. Masked autoencoders are scalable vision learners. *CVPR*, 16000–16009 (2022).
3. Moody B, Hao S, Gow B, Pollard T, Zong W, Mark R. MIMIC-IV Waveform Database (version 0.1.0). PhysioNet (2022). https://doi.org/10.13026/a2mw-f949. We use the v0.2 prerelease.
4. Johnson A, Bulgarelli L, Pollard T, Gow B, Moody B, Horng S, Celi LA, Mark R. MIMIC-IV (version 3.1). PhysioNet (2024). https://doi.org/10.13026/kpb9-mt58
5. Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data* 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x
6. Johnson A, Pollard T, Horng S, Celi LA, Mark R. MIMIC-IV-Note: Deidentified free-text clinical notes (version 2.2). PhysioNet (2023). https://doi.org/10.13026/1n74-ne17
7. Gow B, Pollard T, Nathanson LA, et al. MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset (version 1.0). PhysioNet (2023). https://doi.org/10.13026/4nqg-sb35
8. Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23), e215–e220 (2000).
9. Makowski D, Pham T, Lau ZJ, et al. NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods* 53(4), 1689–1696 (2021). https://doi.org/10.3758/s13428-020-01516-y
10. Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D. Systolic peak detection in acceleration photoplethysmograms measured from emergency responders in tropical conditions. *PLoS ONE* 8(10) (2013).
11. Khodadad D, Nordebo S, Müller B, et al. Optimized breath detection algorithm in electrical impedance tomography. *Physiological Measurement* 39(9), 094001 (2018).
12. Google Research. BERT, "Whole Word Masking Models", README entry of 2019-05-31, and `create_pretraining_data.py`, commit eedf571. https://github.com/google-research/bert
13. Devlin J, Chang M-W, Lee K, Toutanova K. BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*, 4171–4186 (2019).
14. Geng X, Liu H, Lee L, Schuurmans D, Levine S, Abbeel P. Multimodal masked autoencoders learn transferable representations. arXiv:2205.14204 (2022). https://arxiv.org/abs/2205.14204
15. Liu C, Wan Z, Ouyang C, Shah A, Bai W, Arcucci R. Zero-shot ECG classification with multimodal learning and test-time clinical knowledge enhancement. *ICML* (2024). arXiv:2403.06659. https://arxiv.org/abs/2403.06659

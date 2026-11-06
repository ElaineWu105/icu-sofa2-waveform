# SOFA-2 labels, prediction task and endpoints

This folder scores SOFA-2 (Ranzani et al. 2025 [1]) for every ICU day of the cohort and builds the anchors that the 24-hour model predicts. This page covers what SOFA-2 defines, how we score it from MIMIC-IV, the prediction task, how death and discharge are handled, the endpoints we evaluate, and label statistics on the training and validation patients.

**Status.** Labels are final as of 2026-11-01 and were rerun on the new patient split on 2026-11-03; day scores did not change. The endpoints in section 7 are proposed and not yet fixed `[open]`. The level 3 code does not yet compute them (section 10).

Tags: `[source]` taken from a cited work; `[engineering]` our choice with no scientific source; `[extension]` our own method extension, to be stated as such in any paper; `[open]` not yet fixed.

Related documents: `docs/COHORT.md` holds the cohort, attrition and the full scoring rule table; `docs/SETTING.md`, section 3, lists where our scoring differs from `mimic-code` and why; `hierarchy/README.md` describes the model that predicts these labels.

## Contents

- [SOFA-2 labels, prediction task and endpoints](#sofa-2-labels-prediction-task-and-endpoints)
  - [Contents](#contents)
  - [1. What SOFA-2 defines](#1-what-sofa-2-defines)
  - [2. Data](#2-data)
  - [3. Prediction task](#3-prediction-task)
  - [4. Scoring](#4-scoring)
  - [5. Death and discharge](#5-death-and-discharge)
  - [6. Missing values and label masks](#6-missing-values-and-label-masks)
  - [7. Endpoints](#7-endpoints)
    - [7.1 What the model outputs](#71-what-the-model-outputs)
    - [7.2 Primary endpoints](#72-primary-endpoints)
    - [7.3 Secondary and reported measures](#73-secondary-and-reported-measures)
    - [7.4 Comparisons](#74-comparisons)
  - [8. Statistics](#8-statistics)
    - [8.1 Sizes and fate](#81-sizes-and-fate)
    - [8.2 Baseline scores](#82-baseline-scores)
    - [8.3 Endpoint events, main handling](#83-endpoint-events-main-handling)
    - [8.4 Endpoint events, last observation carried forward](#84-endpoint-events-last-observation-carried-forward)
    - [8.5 Reference difficulty](#85-reference-difficulty)
  - [9. Options considered and not used](#9-options-considered-and-not-used)
  - [10. Implementation status](#10-implementation-status)
  - [11. Known limits](#11-known-limits)
  - [12. Files](#12-files)
  - [13. References](#13-references)

## 1. What SOFA-2 defines

From SOFA-2 Table 2 and its notes [1]:

| Item | Definition |
|---|---|
| Organs | Brain, respiratory, cardiovascular, liver, kidney, hemostasis |
| Score | 0 to 4 per organ from the worst value in the assessment period; total 0 to 24 |
| Missing values, note b | For bedside use, a value missing on the first day scores 0 and later missing values carry the last observation. Multiple imputation may be preferable for research |
| Sedation, notes c and d | Use the last GCS before sedation; if a domain cannot be assessed, use the motor score alone |
| Respiratory, note f | SpO2/FiO2 may replace PaO2/FiO2 when PaO2 is unavailable |
| Vasopressors, note j | Count only continuous infusions of at least 1 hour |

The 4-point cardiovascular row prints ">.02", also in the published version; we use 0.2 to match the 3-point row.

SOFA-2 defines a score, not a change. Our target, the change over the next day, follows the fixed-interval delta used in trials (de Grooth et al. 2017 [2]), and the threshold of 2 points follows the Sepsis-3 clinical criteria [3].

## 2. Data

| Source | Version | Used for |
|---|---|---|
| MIMIC-IV [4] | 3.1 | `icu/icustays`, `chartevents`, `inputevents`, `outputevents`, `procedureevents`; `hosp/labevents`, `emar`, `patients`, `admissions` |
| MIMIC-IV Waveform Database | v0.2 prerelease | Which ICU days enter the cohort |
| MIT-LCP `mimic-code` [5] | `main` branch | How MIMIC variables are read where Table 2 is silent |

All need credentialed PhysioNet access. No patient-level file is kept in this repository.

## 3. Prediction task

| Item | Definition | Tag |
|---|---|---|
| Anchor | Time `t = intime + 24k` hours, `k ≥ 1`, in one ICU stay | `[source]`, Helleberg et al. 2026 [6] |
| Waveform input | ECG, PPG and Resp in `[t − 24h, t)` | |
| Baseline | Worst SOFA-2 per organ in `[t − 24h, t)` | `[source]` |
| Future | Worst SOFA-2 per organ in `[t, t + 24h)` | `[source]` |
| Change | Future minus baseline; positive is worse | `[source]` [2] |
| Records used | Baseline scores given to the model use records stored by `t`; future scores use records stored by `t + 24h` | `[engineering]` |
| Inclusion | The baseline day is a complete ICU day with at least 384 of 480 valid three-minute waveform slots | `docs/COHORT.md` |

There is no gap between the input and the future window. Patients have several anchors, so splits, sampling and uncertainty are by patient.

## 4. Scoring

`labels/score/score.py` scores every ICU day; the full rule table is in `docs/COHORT.md`, section "SOFA-2 scoring".

- **Order of authority:** Table 2 decides the rule. `mimic-code` decides how a MIMIC variable is read where Table 2 is silent. Our own choices fill what remains and are tagged `[engineering]` in `docs/SETTING.md`, section 3.
- **Day:** `[intime + 24d, intime + 24(d + 1))`, counted from ICU admission.
- **Evidence:** each organ-day stores its score and whether it rests on data from that day (`real`), a pre-sedation GCS, a carried value, or nothing.
- **Validity:** on the first ICU stay of each waveform patient, day 1 SOFA-2 predicts ICU death with AUROC 0.817, within the published 0.79 to 0.83 (`docs/COHORT.md`, "Validity check").

## 5. Death and discharge

Every anchor is kept, because what happens next is unknown at `t` `[source]` [2, 7].

| Handling | Death in `[t, t + 24h)` | Discharge alive in `[t, t + 24h)` | Role |
|---|---|---|---|
| `main` | Each organ 4, total 24 | Each organ 0, total 0 | Main analysis `[source]` [2, 8] |
| `locf` | Baseline carried | Baseline carried | Sensitivity `[source]` [2, 7] |
| `partial_discharge` | 24 | Scores from the hours before discharge | Sensitivity `[extension]` |
| `no_death` | Excluded | 0 | Shows the effect of dropping deaths |
| `stayed_only` | Excluded | Excluded | Shows the effect of dropping both |

Death uses the hospital death time, so it can follow ICU discharge. The sources set trial endpoints, not prediction labels; using them as labels is our step `[extension]`. Discharge scored 0 assumes recovery, so it inflates improvement (section 8.3), and improvement endpoints use `locf` (section 7.3). Code: `hierarchy/data/day_data.py`, `targets`.

## 6. Missing values and label masks

| Label | Needs | Tag |
|---|---|---|
| Organ baseline | A real score in the baseline window | `[engineering]` |
| Organ future | A real score in the future window if the patient stayed; the handling's value after death or discharge | `[engineering]` |
| Organ change | Both of the above; otherwise the organ is masked for that anchor. No imputation | `[engineering]` |
| Total change | The official rule of note b at both ends | `[source]` [1] |

A carried future value would make an organ's change 0 by construction, so organ labels never use carried values. On the earlier 21,360 anchors, 66.5% of liver zeros and 28.1% of respiratory zeros would have been forced this way. Totals under the stricter rules, at most one carried window and all six real, are sensitivity analyses (`docs/COHORT.md`).

## 7. Endpoints

Decided on 2026-11-04 and 2026-11-05, before the level 3 code is changed. Each choice has a source or is tagged `[engineering]`.

### 7.1 What the model outputs

For each anchor the level 3 head gives two kinds of output [`hierarchy/models/sofa_head.py`]:

| Output | Form | Source |
|---|---|---|
| Organ scores | Per organ, four conditional probabilities P(future > k \| future > k − 1), k = 0 to 3, by CORN. The cumulative probabilities are their running products, P(future > k) = Π_{j ≤ k} P(future > j \| future > j − 1), so they always decrease with k | CORN [15] `[source]` |
| Composite | One probability of the composite event of section 7.2 | One binary output per endpoint, as in HiRID-ICU-Benchmark [14] `[source]` |

CORN trains output k only on anchors whose future score is above k − 1. Unlike independent sigmoids, its cumulative probabilities cannot cross, so every derived probability lies in 0 to 1 without an extra constraint. The training loss is in `hierarchy/README.md`, section 3.6.

Organ events take their probability from the four cumulative probabilities:

| Event, for an organ with baseline b | Eligible | Probability |
|---|---|---|
| Worsening by 1 or more | b ≤ 3 | P(future > b) |
| Worsening by 2 or more | b ≤ 2 | P(future > b + 1) |
| Improvement by 1 or more | b ≥ 1 | 1 − P(future > b − 1) |

Example: respiratory baseline 2 and conditional outputs 0.95, 0.84, 0.375, 0.167 give cumulative 0.95, 0.80, 0.30, 0.05, so worsening ≥ 1 is 0.30, worsening ≥ 2 is 0.05 and improvement ≥ 1 is 0.20.

An anchor whose organ cannot move in a direction is not eligible for that event, as HiRID predicts organ failure only "given the patient is not in failure now" [14]. Keeping it would credit the model for events that cannot happen.

### 7.2 Primary endpoints

| | Primary 1: composite | Primary 2: organ worsening |
|---|---|---|
| Event | Total change ≥ 2, or death in `[t, t + 24h)` | Organ future > organ baseline |
| Unit | Anchor | Anchor and organ, among eligible pairs |
| Labels | `main` handling, official total | `main` handling, organ masks of section 6 |
| Probability | The composite output | P(future > b) |
| Metric | AUROC | AUROC per organ, then the mean over six organs |
| Also reported | AUPRC, event rate, calibration curve, calibration slope and intercept, Brier score | Per-organ AUROC, AUPRC and event counts; the mean without liver |
| Source | Total change ≥ 2 is the Sepsis-3 criterion for organ dysfunction [3, 18]; adding death follows the persistent organ dysfunction plus death composite [13] | Organ event prediction per organ system, as in HiRID [14] |

Under `main`, a death scores 24, so death almost always meets total change ≥ 2; the composite then nearly equals total worsening ≥ 2 (section 8.3). Death is named in the endpoint so that it stays an event under `locf`.

Liver enters the mean of primary 2. It has 76 validation events against 204 to 416 for the other organs, so the mean without liver is also reported.

### 7.3 Secondary and reported measures

| Measure | Definition | Why not primary |
|---|---|---|
| Organ worsening ≥ 2 | Eligible b ≤ 2; AUROC and AUPRC for organs with at least 30 validation events | Few events, and under `locf` liver has none |
| Organ improvement ≥ 1, exploratory | Eligible b ≥ 1; labels and model both under `locf` [7] | Under `main`, discharge scored 0 makes improvement mostly mean discharge (section 8.3); no prediction study uses it as an endpoint |
| Ranked probability score | Per organ, mean over k of (P(future > k) − 1[future > k])², the mean Brier score of the four cumulative events | Scores the whole distribution; its advantages over simpler scores are disputed [11, 12] |
| MAE | Expected future score, the sum of P(future > k), minus baseline, against the true change, per organ and total; also on anchors whose score changed | Most changes are 0, so MAE rewards predicting no change. Reported as next-day SOFA prediction studies do [16] |

AUROC is the primary metric, not AUPRC: AUPRC is not generally better under class imbalance [9]. Calibration is always reported [10].

### 7.4 Comparisons

| Model | Input |
|---|---|
| P | Future equals baseline, the persistence baseline of next-day SOFA prediction [16] |
| E0 | The six baseline scores and their missing flags |
| E2 | E0 plus the EHR inputs below |
| M | E2 and the mean of frozen 10-second encoder features |
| H | E2 and the three-level waveform model |
| H without self-supervision | As H, with levels 1 and 2 trained by the SOFA-2 loss |

**E2 inputs**, all from records stored by `t`, as in section 3:

| Input | Values | Missing |
|---|---|---|
| Baseline scores | The six organ scores of `[t − 24h, t)` | 0 with a missing flag, as in section 6 |
| Previous day's scores | The six organ scores of `[t − 48h, t − 24h)` | 0 with a missing flag. About 22% of anchors have their baseline on the first ICU day, so the previous day lies before ICU admission and has no score |
| ICU day | `k`, with `t = intime + 24k` hours | Never missing |
| Hourly vital signs | Median charted heart rate, respiratory rate, SpO2 and MAP in each of the 24 hours before `t`, 96 values | 0 with a missing flag per value |

Missing values are given with indicator flags, a standard way to let a clinical prediction model use missingness [17]. Scores from ward data before ICU admission are not used: the ward has laboratory values but no complete vasopressor and ventilation records, so a ward score would not match ICU scores. The hourly median is `[engineering]`; it is robust to single charting errors.

**How the E2 table is built** (`e2/e2.py`, `private/e2.pkl`). Previous-day scores are the real SOFA-2 organ scores of the day before the baseline day; day scores already use records stored by the end of their day (section 3), so they are known at `t`. Vital signs are charted values with `charttime` in the hour and `storetime` by `t`. Items and plausible ranges follow `mimic-code` `vitalsign.sql` [5] `[source]`: heart rate 220045, 0 to 300; respiratory rate 220210 and 224690, 0 to 70; SpO2 220277, 0 to 100; mean arterial pressure 220052, 220181 and 225312 pooled, invasive and non-invasive together as that file does, 0 to 300. The level 3 head standardizes the continuous values with train-patient statistics (`hierarchy/data/day_data.py`, `context`).

**Why H includes E2.** H adds a waveform correction to an EHR baseline path (`hierarchy/README.md`, section 3.6), so H against its baseline measures what the waveform adds. Heart rate, respiratory rate and SpO2 can be read from the waveform but are also charted hourly. With E0 alone as the baseline, a gain could come from reading those values back out of the waveform. With E2, the waveform must add what charting does not hold. Both H against E0 and H against E2 are reported.

All intervals use 1,000 bootstrap draws of validation patients. Differences are paired: every model is scored on the same draws. The claim needs H better than E2 and M on both primary endpoints. P and E0 are lower bounds. Three seeds per model `[engineering]`. The five handlings of section 5 are repeated for H and E2.

Checkpoints are chosen by the validation SOFA-2 loss, not by these endpoints. The test split is scored once, after the setting is fixed.

## 8. Statistics

Computed from `labels/private/anchors.pkl` with the definitions above. Event counts are given for training and validation patients only; for test patients only sizes are given, so the test labels stay unread.

### 8.1 Sizes and fate

| Split | Patients | Stays | Anchors | Stayed in ICU | Left alive | Died |
|---|---:|---:|---:|---:|---:|---:|
| Train | 3,811 | 4,510 | 15,615 | 11,606 (74.3%) | 3,552 (22.7%) | 457 (2.9%) |
| Validation | 778 | 933 | 3,284 | 2,476 (75.4%) | 724 (22.0%) | 84 (2.6%) |
| Test | 817 | 973 | 3,442 | | | |

Anchors per patient, all splits: median 2 (IQR 1 to 4), maximum 74. Baseline day, train and validation: median ICU day 4 (IQR 2 to 8); 22.4% of anchors predict ICU day 2 from day 1.

### 8.2 Baseline scores

Train and validation anchors. Coverage is the share of anchors with a real baseline score; the score columns are shares of those.

| Organ | Coverage | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|---:|
| Brain | 84.4% | 39.2% | 38.5% | 12.1% | 3.1% | 7.1% |
| Respiratory | 80.8% | 39.0% | 12.9% | 29.5% | 14.4% | 4.2% |
| Cardiovascular | 99.9% | 22.5% | 39.8% | 23.9% | 7.4% | 6.4% |
| Liver | 40.0% | 59.6% | 16.6% | 7.9% | 6.9% | 9.0% |
| Kidney | 99.5% | 36.1% | 16.9% | 22.0% | 6.1% | 18.9% |
| Hemostasis | 96.9% | 59.3% | 19.6% | 6.1% | 7.3% | 7.7% |

Baseline total under the official rule: median 6 (IQR 3 to 8). Kidney has the most anchors at 4 (18.9%), which cannot worsen.

### 8.3 Endpoint events, main handling

**Primary 1, composite:**

| Split | Anchors | Events | Rate | Of which deaths |
|---|---:|---:|---:|---:|
| Train | 15,615 | 1,950 | 12.5% | 457 |
| Validation | 3,284 | 445 | 13.6% | 84 |

Total change ≥ 2 alone gives 1,943 and 443 events: all but 7 and 2 deaths already meet it.

**Primary 2 and secondary organ endpoints.** Labeled: anchors with an organ label. At 4: labeled anchors already at the top score. Worse ≥ 1 is among eligible anchors (b ≤ 3), worse ≥ 2 among b ≤ 2, improved ≥ 1 among b ≥ 1. Same: share of labeled anchors with no change.

Train:

| Organ | Labeled | At 4 | Eligible for worse ≥ 1 | Worse ≥ 1 | Worse ≥ 2 | Improved ≥ 1 | Same |
|---|---:|---:|---:|---:|---:|---:|---:|
| Brain | 12,578 | 856 | 11,722 | 1,448 (12.4%) | 425 | 3,018 (39.8%) | 64.5% |
| Respiratory | 11,504 | 510 | 10,994 | 1,735 (15.8%) | 636 | 2,813 (40.1%) | 60.5% |
| Cardiovascular | 15,584 | 1,008 | 14,576 | 1,764 (12.1%) | 464 | 5,229 (43.4%) | 55.1% |
| Liver | 4,616 | 530 | 4,086 | 407 (10.0%) | 193 | 600 (27.0%) | 78.2% |
| Kidney | 15,494 | 2,874 | 12,620 | 2,003 (15.9%) | 679 | 3,730 (37.5%) | 63.0% |
| Hemostasis | 14,754 | 1,138 | 13,616 | 1,192 (8.8%) | 361 | 2,513 (40.8%) | 74.9% |

Validation:

| Organ | Labeled | At 4 | Eligible for worse ≥ 1 | Worse ≥ 1 | Worse ≥ 2 | Improved ≥ 1 | Same |
|---|---:|---:|---:|---:|---:|---:|---:|
| Brain | 2,657 | 163 | 2,494 | 315 (12.6%) | 99 | 646 (40.2%) | 63.8% |
| Respiratory | 2,445 | 87 | 2,358 | 365 (15.5%) | 142 | 615 (41.1%) | 59.9% |
| Cardiovascular | 3,277 | 197 | 3,080 | 377 (12.2%) | 86 | 1,073 (41.9%) | 55.8% |
| Liver | 1,024 | 103 | 921 | 76 (8.3%) | 30 | 128 (27.7%) | 80.1% |
| Kidney | 3,253 | 670 | 2,583 | 416 (16.1%) | 156 | 783 (38.0%) | 63.1% |
| Hemostasis | 3,122 | 248 | 2,874 | 204 (7.1%) | 63 | 483 (41.2%) | 78.0% |

Validation events for primary 2: 1,753 over six organs, 1,677 without liver.

### 8.4 Endpoint events, last observation carried forward

Under `locf`, discharge and death carry the baseline, so their organ changes are 0.

| Split | Composite events | Rate | Total change ≥ 2 | Deaths |
|---|---:|---:|---:|---:|
| Train | 1,950 | 12.5% | 1,493 | 457 |
| Validation | 445 | 13.6% | 361 | 84 |

Validation organ events:

| Organ | Worse ≥ 1 | Worse ≥ 2 | Improved ≥ 1 | Same |
|---|---:|---:|---:|---:|
| Brain | 270 (10.8%) | 62 | 325 (20.2%) | 77.6% |
| Respiratory | 301 (12.8%) | 96 | 405 (27.1%) | 71.1% |
| Cardiovascular | 327 (10.6%) | 44 | 570 (22.2%) | 72.6% |
| Liver | 44 (4.8%) | 0 | 58 (12.6%) | 90.0% |
| Kidney | 370 (14.3%) | 117 | 393 (19.1%) | 76.5% |
| Hemostasis | 146 (5.1%) | 13 | 207 (17.6%) | 88.7% |

Improvement halves from `main` to `locf`, for example cardiovascular 41.9% to 22.2%: under `main`, about half of all improvement is discharge. This is why improvement uses `locf`. Worsening changes less, because it comes mostly from patients who stay.

### 8.5 Reference difficulty

E0 as transition tables fitted on train patients and scored on validation patients, `main` handling:

| Endpoint | E0 score | Validation AUROC |
|---|---|---:|
| Composite | Event rate of the training anchors with the same baseline total | 0.542 |
| Worsening ≥ 1, brain | Event rate by baseline score | 0.546 |
| Respiratory | | 0.610 |
| Cardiovascular | | 0.649 |
| Liver | | 0.604 |
| Kidney | | 0.583 |
| Hemostasis | | 0.620 |
| Mean of six organs | | 0.602 |

The current score alone ranks next-day worsening only a little better than chance, so the waveform and E2 have room to add information. The E0 model of section 7.4, fitted on the six scores together, may score higher than these tables.

## 9. Options considered and not used

| Option | Why not |
|---|---|
| Dropping anchors that end in death or discharge | Uses the future to choose anchors and biases the result [2, 7]; kept only as the `no_death` and `stayed_only` sensitivity analyses |
| MAE of the change as the primary metric | 55% to 80% of organ changes are 0 (section 8.3), so it rewards predicting no change |
| Worsening ≥ 2 as the organ primary endpoint | 30 to 156 validation events per organ, and none for liver under `locf` |
| Improvement under `main` | Mostly measures discharge (section 8.4) |
| AUPRC as the primary metric | Not generally better under imbalance [9]; reported beside AUROC |
| Including anchors already at 4 for worsening | Credits the model for events that cannot happen |
| Partial-day scores after discharge as the main handling | Suggested in review; the hours before discharge are few, and no source uses it. Kept as `partial_discharge` |
| Composite probability from the six organ distributions, combined as if independent | No source, and organs worsen together, death above all, so the upper tail would be miscalibrated. A composite output is trained instead (section 7.1) |
| Independent sigmoids on P(future > k), forced to decrease | The constraint is ours; CORN guarantees order by construction [15] |
| CORAL instead of CORN | Also rank-consistent, by shared weights; CORN does not share weights and suits the skewed score distribution [15] |
| A Huber loss on the summed organ change | No source; it trains the expected change, while the endpoint needs the probability of change ≥ 2, which the composite output gives directly |
| Previous day's score from ward data before ICU admission | Ward data lack complete vasopressor and ventilation records |
| Baseline path with E0 only in H | A gain could come from reading charted vital signs out of the waveform (section 7.4) |

## 10. Implementation status

| Part | State |
|---|---|
| Day scores, anchors, cohort report | Done: `score/score.py`, `score/finalize.py`, `cohort_report.json` |
| Death and discharge handlings | Done: `hierarchy/data/day_data.py` |
| Ordinal head and SOFA-2 loss | Done: `hierarchy/models/sofa_head.py` |
| CORN head and composite output; endpoint probabilities from the head | Done: `hierarchy/models/sofa_head.py`, with tests |
| Endpoints of section 7 with eligibility, calibration, RPS and paired bootstrap | Done: `hierarchy/level3_day/evaluate_day.py`, with tests |
| E2 inputs of section 7.4: previous day with missing flags, ICU day, hourly median vital signs stored by `t` | Done: `e2/e2.py` and `extract/extract.py vitals`; the head inputs in `hierarchy/data/day_data.py` |
| E0, E2 and M controls | Done: `hierarchy/level3_day/train_day.py` |
| No-self-supervision control | To do |
| `staged_check.py` on the same events | To do; it now reports MAE |
| Comparison of our scores with `mimic-code` `sofa.sql`, and small hand-made SOFA-2 cases | To do |

## 11. Known limits

- **Labels are not validated against a reference SOFA-2 implementation,** because none exists for MIMIC. They follow Table 2 and `mimic-code` and pass the day 1 mortality check.
- **Table 2 rules not implemented:** excluding PaO2/FiO2 changes within 1 hour (note g), ceiling-of-treatment scoring (notes h and m), and excluding nonrenal RRT (note o).
- **Norepinephrine salt versus base is unknown;** halving the dose changes 4.0% of cardiovascular day scores.
- **Liver** has a real baseline on 40% of anchors and few events.
- **Discharge scored 0** assumes recovery; a patient can leave with abnormal creatinine or platelets.
- **One day ahead only,** at anchors 24 hours apart; denser anchors and other horizons are not studied.
- **These statistics depend on the split;** they are for the 70/15/15 hash split of 2026-11-03.

## 12. Files

| File | Role |
|---|---|
| `run.sh` | `extract`, `score`, `finalize`, `vitals`, `e2`, `test` |
| `extract/extract.py` | Reads the MIMIC-IV items, into `private/raw/` |
| `score/infusion.py` | Continuous vasopressor runs (note j) |
| `score/score.py` | SOFA-2 and SOFA-1 per ICU day, `private/daily.pkl` |
| `score/finalize.py` | Joins day scores to cut waveform days; writes `private/anchors.pkl` and `cohort_report.json` |
| `e2/e2.py` | E2 inputs per anchor, `private/e2.pkl`, from `private/vitals.pkl` (`extract/extract.py vitals`) |
| `score/test_infusion.py` | Tests of the vasopressor rule |
| `cohort_report.json` | Aggregate counts, safe to share |
| `private/` | Patient-level tables; never committed or shared |

## 13. References

1. Ranzani OT, et al. Development and validation of the Sequential Organ Failure Assessment (SOFA)-2 score. *JAMA* (2025). https://jamanetwork.com/journals/jama/fullarticle/2840822
2. de Grooth H-J, Geenen IL, Girbes AR, Vincent J-L, Parienti J-J, Oudemans-van Straaten HM. SOFA and mortality endpoints in randomized controlled trials: a systematic review and meta-regression analysis. *Critical Care* 21, 38 (2017). https://doi.org/10.1186/s13054-017-1609-1
3. Seymour CW, et al. Assessment of clinical criteria for sepsis. *JAMA* 315(8), 762–774 (2016). https://jamanetwork.com/journals/jama/fullarticle/2492875
4. Johnson A, Bulgarelli L, Pollard T, Gow B, Moody B, Horng S, Celi LA, Mark R. MIMIC-IV (version 3.1). PhysioNet (2024). https://doi.org/10.13026/kpb9-mt58
5. MIT-LCP. mimic-code. https://github.com/MIT-LCP/mimic-code
6. Helleberg et al. *Critical Care* (2026). https://doi.org/10.1186/s13054-026-06093-8
7. Lambden S, Laterre PF, Levy MM, Francois B. The SOFA score: development, utility and challenges of accurate assessment in clinical trials. *Critical Care* 23, 374 (2019). https://doi.org/10.1186/s13054-019-2663-7
8. AdrenOSS-2 trial protocol, version 4.2, NCT03085758, section 10.5.3. https://cdn.clinicaltrials.gov/large-docs/58/NCT03085758/Prot_000.pdf
9. McDermott M, et al. A closer look at AUROC and AUPRC under class imbalance. arXiv:2401.06091 (2024). https://arxiv.org/abs/2401.06091
10. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Medicine* 17, 230 (2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC6912996
11. Constantinou AC, Fenton NE. Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models. *Journal of Quantitative Analysis in Sports* 8(1) (2012). https://ideas.repec.org/a/bpj/jqsprt/v8y2012i1n12.html
12. Wheatcroft E. Evaluating probabilistic forecasts of football matches: the case against the ranked probability score. *Journal of Quantitative Analysis in Sports* 17(4), 273–287 (2021). https://ideas.repec.org/a/bpj/jqsprt/v17y2021i4p273-287n1.html
13. Heyland DK, Muscedere J, Drover J, Jiang X, Day AG; Canadian Critical Care Trials Group. Persistent organ dysfunction plus death: a novel, composite outcome measure for critical care trials. *Critical Care* 15, R98 (2011). https://doi.org/10.1186/cc10110
14. Yèche H, Kuznetsova R, et al. HiRID-ICU-Benchmark: a comprehensive machine learning benchmark on high-resolution ICU data. *NeurIPS Datasets and Benchmarks* (2021). arXiv:2111.08536. https://arxiv.org/abs/2111.08536
15. Shi X, Cao W, Raschka S. Deep neural networks for rank-consistent ordinal regression based on conditional probabilities. *Pattern Analysis and Applications* 26, 941–955 (2023). https://doi.org/10.1007/s10044-023-01181-9. arXiv:2111.08851
16. Albouq R, Aslam A. Predicting future organ dysfunction in ICU patients using temporal convolutional networks on MIMIC-IV data. arXiv:2608.29301 (2026). https://arxiv.org/abs/2608.29301
17. Lipton ZC, Kale DC, Wetzel R. Directly modeling missing data in sequences with RNNs: improved classification of clinical time series. *Machine Learning for Healthcare*, PMLR 56, 253–270 (2016). https://proceedings.mlr.press/v56/Lipton16.html
18. Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA* 315(8), 801–810 (2016). https://doi.org/10.1001/jama.2016.0287

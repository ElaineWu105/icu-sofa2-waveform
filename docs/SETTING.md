# ICU waveform → SOFA-2 change: full pipeline setting

Version: draft 3, 2026-10-30, updated to 2026-11-06. Status: labels, waveform cut, fine-tuning and the encoder comparison are done; arm A is the chosen encoder (section 5); the three training levels are written and tested but not yet trained on the new split (section 7).

Location: paths in this document are the original working folders. In this repository, the same code is under `labels/`, `waveform/`, `finetune/` and `encoder_comparison/`.

Tags used below:

- `[source]` taken from a published source
- `[kept]` kept from an existing project experiment
- `[engineering]` engineering choice with no scientific source
- `[extension]` our own method extension, to be stated as such in any paper
- `[open]` open; the default here is provisional
- `[changed date]` a rule changed on that date

## 1. Task

Predict, from ECG/PPG/Resp only plus baseline SOFA-2, the change in each of the six organ scores and the total over the next 24 hours, using ICU data only.

For prediction time `t`:

| Item | Interval | Note |
|---|---|---|
| Waveform input | `[t−24h, t)` | Length is compared in section 8 `[extension]` |
| Baseline SOFA-2 | worst per organ in `[t−24h, t)` | SOFA-2 Table 2 `[source]` |
| Future SOFA-2 | worst per organ in `[t, t+24h)` | same |
| Target | future − baseline | fixed-interval delta, de Grooth 2017 `[source]` |
| Worsening | total or organ delta ≥ 2 | Sepsis-3 threshold `[source]` |

`t = intime + 24k` hours, `k ≥ 1`, so ICU days are `[intime+24d, intime+24(d+1))` as in Helleberg 2026 `[source]`. Denser anchors are a later sensitivity analysis.

There is no gap between input end and future window. Baseline scores given to the model use only records with `storetime ≤ t`. Target scores use `storetime ≤ t+24h`.

## 2. Cohort

Sources:

- EHR: `/home/public/mimiciv/3.1/` (`icu/`, `hosp/`)
- Waveform: `/home/public/MIMIC_waveform/mimic_iv_waveform_v0_2_prerelease/v0.2`

Rules, applied in this order with counts reported at each step:

1. Patient has a waveform record: 7,306 patients with ICU stays.
2. Age ≥ 18, any ICU stay: 13,604 stays. A later stay in the same admission is allowed; patient-level splitting and bootstrap handle repeated stays. Sensitivity: first ICU stay of each admission.
3. ICU day `d` is complete and `t < outtime`: 47,730 days.
4. WFDB headers can still reach ≥ 384 of 480 three-minute slots `[kept: 80% rule]`: 22,126 days, 6,527 stays, 5,497 patients. This is an upper bound until step 5.
5. After signal QC, ≥ 384 of 480 three-minute slots are valid (section 6). 80% is total coverage, not one unbroken stretch; missing slots stay in place and are masked. In 400 sampled 12h windows, total coverage kept 82.2% and one unbroken 80% stretch kept 25.5%.

Not cut by CXR. CXR is neither an anchor nor an input.

Fate of each anchor over `[t, t+24h)` (header plan, before signal QC and the added days; final counts are in `COHORT.md`):

| Fate | Anchors | Share | Use |
|---|---:|---:|---|
| Stayed in ICU | 16,337 | 73.8% | real scores |
| Left ICU alive | 5,161 | 23.3% | next window scored 0 |
| Died | 628 | 2.8% | next window scored 24; an event in the composite |

No anchor is dropped for what happens next, because that is unknown at prediction time `[changed 2026-11-02]`. Sources for scoring the window after death or ICU discharge:

| Source | Death | Discharge alive | What it says about leaving patients out |
|---|---|---|---|
| de Grooth 2017, Crit Care, systematic review of SOFA endpoints in RCTs | 24, or last observation | 0, or last observation | "Simply assigning no score to discharged patients will obviously lead to bias" |
| Lambden 2019, Crit Care, review of SOFA in trials | Last value, "a maximum sub- or total value", or an extra penalty; "no consensus" | Not covered | Dropping deaths causes "survivorship bias", as when CITRIS-ALI removed non-survivors |
| AdrenOSS-2 protocol v4.2, NCT03085758, section 10.5.3 | "assigned the maximum SOFA score (24)" | "set to 0" after transfer to a ward or home | Used as written in a registered trial |

- **Main analysis, 24 and 0 `[source]`:** the pair de Grooth names, and the rule AdrenOSS-2 applied.
- **Each organ 4 at death:** Lambden names the maximum sub-score, so organ level is `[source]` too.
- **`locf` `[source]`:** the other option in de Grooth and Lambden.
- **`no_death` and `stayed_only`:** both leave patients out, which de Grooth and Lambden warn biases the result. They are kept only to show how much the choice matters.
- **`partial_discharge` `[extension]`:** the PhD review suggested it, and no source above uses it.
- **Limit:** all three sources set trial endpoints, not prediction labels. Using their rule as a label is our step `[extension]`.
- **Discharge 0 assumes recovery,** but a patient can leave the ICU with abnormal creatinine or platelets. Improvement measures depend on it, as noted below.
- **The composite "total delta ≥ 2 or death within 24h" `[extension]`:** it joins the Sepsis-3 threshold with death, like the composite "persistent organ dysfunction or death" that AdrenOSS-2 also considers.

Sensitivity analyses, chosen with `--handling`:

| Handling | Death | Discharge |
|---|---|---|
| `main` | 24 | 0 |
| `locf` | Last observation | Last observation |
| `partial_discharge` | 24 | Scores from the part of the window before discharge |
| `no_death` | Anchor left out | 0 |
| `stayed_only` | Anchor left out | Anchor left out. This was the earlier primary analysis, following the octogenarian ICU study |

Scoring discharge as 0 makes each discharged anchor an improvement by its whole baseline score, so measures of improvement and of change size depend on this choice: on validation, total delta ≤ −2 is 34.1% under `main` and 15.0% under `locf`. Worsening does not: discharged anchors count as not worsened under both, and the composite is the same.

### Split

**Every patient takes one rule: `SHA256(subject_id)` to 70/15/15** `[engineering]` `[changed 2026-11-03]`. Code: `waveform/icu_cut/common.py`, `assign_split`, called in `plan.py`. No split file is needed, so the split is the same on any machine and no patient list has to be shared.

- **Why it replaced the earlier split:** the earlier split was two methods together (below), it needed a patient list that cannot be published, and every model is trained again anyway. Counts per split and how many patients moved are in `COHORT.md`, section "Splits".
- **What it costs:** the arms, the 10.17 epoch 17 encoder and the 3-minute model trained on the earlier split, where 796 of today's validation and test patients were train patients. None of them is evaluated on this split. The test split stays unread until the setting is locked.
- No patient crosses splits. All encoder and SSL training uses train patients only.

**Earlier split, used by all results so far.** It was the earlier assignment plus a new rule, two methods together. At every stage a patient who had an assignment kept it, and only new patients were assigned.

| Stage | Patients | Method |
|---|---:|---|
| 9.22 pilot | 380 | Seeded shuffle to 70/15/15 |
| 9.27 | 4,706 | Keep the 380; seeded shuffle of the 4,326 new patients |
| 10.16 `pairs.csv` | 4,695 | Export of the ECG, CXR and EHR pairs; assignments unchanged |
| Earlier version of this cohort | 5,406 | Keep 3,938 from `pairs.csv`; `SHA256(subject_id)` for the 1,468 new patients |

Its earlier steps ran outside this repository, so their code is copied here. Run on the stored splits with seed 42, it reproduces all 380 and all 4,326 assignments. Patient IDs are strings at these two steps, which sets the sort order.

Earlier assignment, `9.22/note_ecg_pilot/prepare.py`:

```python
def patient_splits(ids, seed):
    # Assign the entire eligible candidate population before signal/text QC.
    ids = sorted(set(ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    a, b = int(.7*n), int(.85*n)
    return {p: ('train' if i < a else 'val' if i < b else 'test') for i,p in enumerate(ids)}
```

Keeping it while adding patients, `9.27/note_ecg_all/prepare.py`:

```python
split = legacy.patient_splits(patients - previous.keys(), args.seed)
split.update({p: previous[p] for p in patients & previous.keys()})
```

In the earlier version of this cohort, `plan.py` read `pairs.csv` and gave the hash rule only to patients missing from it.

- SOFA test results are computed once, after this setting is locked.

## 3. SOFA-2 labels

Labels are final as of 2026-11-01: code in `labels/`, patient-level tables in `labels/private/`, aggregates in `labels/cohort_report.json`, summary in `COHORT.md`. Numbers from the earlier first-stay pilot further down this section are superseded by `COHORT.md`.

Thresholds and notes: SOFA-2 Table 2 (Ranzani, JAMA 2025) `[source]`. The 4-point cardiovascular row prints ">.02", also in the published version; use 0.2 to match the 3-point row.

**Order of authority.** SOFA-2 Table 2 decides the rule. `mimic-code` decides how a MIMIC variable is read where Table 2 is silent. Our own choices fill what remains and are listed under the project approximations. `mimic-code` is community code maintained by the MIMIC team. It has SOFA-1 only (`sofa.sql`), so it is not a SOFA-2 reference.

Extraction follows MIT-LCP `mimic-code` `[source]`:

| Organ | Sources |
|---|---|
| Brain | GCS 220739, 223900, 223901; motor-only if domains are unavailable; last pre-sedation GCS if sedated |
| Respiratory | arterial PaO2 50821 and FiO2 50816 by specimen; chart FiO2 223835 within 4h; SpO2/FiO2 only if PaO2/FiO2 is unavailable and SpO2 < 98%; support from `ventilation.sql`, the last charted status holding for up to 14h |
| Cardiovascular | MAP 220052, 220181, 225312; norepinephrine + epinephrine sum; other agents; ECMO, IABP, Impella, LVAD, TandemHeart |
| Liver | bilirubin 50885 |
| Kidney | creatinine 50912; urine from `urine_output.sql`; RRT from `rrt.sql` |
| Hemostasis | platelets 51265 |

Checked against the `mimic-code` source and the same: lab, MAP and urine items, the GU irrigant sign, active RRT items, the 37 ventilator modes and 13 Hamilton modes, FiO2 order and its 4h window.

Where this differs from `mimic-code`, and why:

| Item | `mimic-code` | Here | Reason |
|---|---|---|---|
| Sedated GCS | Defaults to 15 | Last GCS before sedation | Table 2 note c |
| Incomplete GCS | Fills components from the prior 6h, else normal | Motor score alone | Table 2 note d |
| Respiratory ratio | PaO2/FiO2 only | SpO2/FiO2 when PaO2 is unavailable | Table 2 note f |
| Vasopressors | Rate at each hour, no minimum duration | Continuous run of at least 60 minutes | Table 2 note j |
| Urine | 24h volume, SOFA-1 cutoffs | 6h, 12h and 24h mL/kg/h | Table 2 |
| End of a support status | Ends if the next row is 14h or more away | Holds for up to 14h | The next row is future data |
| Time window | Rolling 24h on a clock-hour grid | ICU day from `intime` | Windows align to the prediction time |
| Records used | All | Stored by the day's end | Avoids future data |

Done for the final labels:

1. RRT follows `rrt.sql` active items.
2. Respiratory support follows `ventilation.sql`.
3. `patientweight = 1` follows `norepinephrine.sql`.
4. A record counts for a day only if stored by that day's end.
5. SOFA-1 comparator uses the same evidence, with SpO2/FiO2 mapped by Pandharipande 2009 `[engineering]`.
6. The pilot's excess of higher SOFA-2 scores came mostly from a weaker SOFA-1 comparator. On the same evidence SOFA-2 is lower in 44.0%, the same in 29.0% and higher in 27.0%, per patient on day 1.

Project approximations to report `[engineering]`:

- Sedatives: propofol, midazolam, dexmedetomidine, ketamine, lorazepam infusions.
- Delirium drugs: haloperidol, quetiapine, olanzapine administrations.
- Intermittent RRT stays at 4 for 72h after the last session.
- Urine rolling calculation with a 6h maximum record gap.
- Vasopressors, note j: touching or overlapping records of one drug form one run, any gap ends it, and the run must reach 60 minutes by the window's end. A run may begin in the previous window.
- MAP is the daily minimum.
- Salt-versus-base norepinephrine is unknown. Halving the dose changes 4.0% of cardiovascular day scores. Report as a sensitivity analysis.
- Note p, RRT criteria without RRT: only the creatinine trigger is used. Adding oliguria changes 55 of 61,334 day scores (0.09%), so the labels keep the simpler form.
- Not implemented: excluding PaO2/FiO2 changes within 1 hour (note g), ceiling-of-treatment scoring (notes h, m), excluding nonrenal RRT (note o).

### Missing values

Every organ-day stores `score`, `evidence` (`real`, `presedation`, `carried`, `none`) and rule version.

- Organ delta: both ends `real`. Otherwise masked. A carried future value would make that organ's delta 0 by construction: on the 21,360 anchors, 66.5% of liver zeros and 28.1% of respiratory zeros under the official rule.
- Total delta, main: official SOFA-2 rule, Table 2 note b `[source]`. Missing on ICU day 1 scores 0; later missing values carry the last observation. 16,670 anchors, 83.2% real endpoints.
- Total delta, sensitivity: at most one carried window per organ (Lambden 2019) `[source]`, 4,910 anchors, and all six real, 2,227 anchors. Unchanged 29.0%, 28.4% and 26.9%; delta ≥ 2 13.3%, 14.3% and 15.5%.
- Ranzani 2025 recommends carry-forward for bedside use and says multiple imputation may be preferable for research, so results are reported under all three rules.

Pilot strict organ labels:

| Organ | Labels | Unchanged |
|---|---:|---:|
| Cardiovascular | 15,170 | 63.9% |
| Kidney | 15,120 | 70.5% |
| Hemostasis | 14,475 | 81.9% |
| Brain | 11,485 | 68.1% |
| Respiratory | 11,450 | 60.6% |
| Liver | 4,466 | 86.5% |

## 4. Ten-second waveform cut

Reuse the 10.14 joint CSFM preprocessing unchanged `[kept]`:

| Item | Setting |
|---|---|
| ECG | native 250 Hz, `ecg_clean` (neurokit, powerline 50), lead priority II, I, III, aVR, aVL, aVF, V1–V6, true lead ID kept |
| PPG | native 125 Hz, FFT to 250 Hz, `ppg_clean` (elgendi) |
| Resp | native 62.5 Hz, FFT to 250 Hz, `rsp_clean` (khodadad2018) |
| Output | float32 `[N,3,2500]`, per-channel z-score, modality mask, channel IDs; ECG 0–11, PPG 12, Resp 13 |
| Rules | complete 10s only; no stitching across gaps or admissions |

New in this run:

- **Grid:** slot `k` is `[intime + 10k s, intime + 10(k+1) s)` `[engineering]`, so 18 slots make 3min, 360 make 1h and 8,640 make 24h, all aligned to `t`.
- **Scope:** only ICU days that are anchors in section 2, regardless of whether SOFA can be computed for them.
- **Storage:** one `signal.npy` and one `meta.npz` per ICU day, replacing the 10.14 SQLite index. Each worker writes its own day; the meta file marks it finished, so a stopped run resumes by rerunning.
- **Code:** `waveform/icu_cut/`, importing the 10.14 `process_window` and `read_window` (`waveform/windows_10s/`) and the 10.13 header inventory (`waveform/native_index/`) unchanged.
- **Output path:** `waveform/icu_cut/output/`. Old 10.14 outputs are read only.

Result of the full cut, on the earlier plan of 22,126 days:

| Measure | Value |
|---|---:|
| Windows planned | 186,486,496 |
| At least one valid channel | 186,194,282 (99.8%) |
| All three channels | 181,474,055 (97.3%) |
| Rejected by signal QC | 292,214 |
| Read failures | 0 |
| Days passing 384 of 480 after QC | 21,360 |

The run took about 11.5 hours, at about 1,900 days per hour. Signals are about 5.1 TB.

**Index end.** Segments end at the `endtime` of `v0.2_tmp_starttime_index.csv`, which drops 8.1% of waveform hours that look valid (`waveform/coverage_audit/`). `plan` adds, after the index-end days, the 1,005 days that pass when records end at their master header; 981 pass QC. They have their own header inventory (`records_open_ended`) and segment numbers after the first 108,198. They were first cut into a separate folder and moved into the main cut on 2026-11-03, signals by hard link and meta files with shifted segment numbers; lead, status, rows, segment records and day starts were checked equal for all 8,551,254 planned slots. Together: 23,131 planned days and 195,037,750 planned windows.

**Verification, 2026-11-02,** in a new environment built from `requirements.txt`: labels rebuilt from raw MIMIC-IV gave identical extracts, `daily`, `anchors` and `cohort_report.json`; `plan` rebuilt from raw headers gave identical tables; `cut` on 3 ICU days gave byte-equal files; 200 windows matched the 10.14 output byte for byte (`waveform/icu_cut/validate_equivalence.py`). The full cut was not rerun.

Planned volume before the run: 186.5M windows, about 5.5 TB. Measured about 1,645 ICU days per hour with 64 workers, so about 13.5 hours. This step is CPU-bound in NeuroKit cleaning; a GPU rewrite would break byte equality with earlier data.

## 5. CSFM ten-second fine-tune

Goal: add Resp to the CSFM encoder with the paper's masked objective. The paper has no Resp channel and no self-supervised fine-tune step, so this stage as a whole is `[extension]`; each part below stays as close to Gu et al. 2026 as possible.

| Item | Setting | Source |
|---|---|---|
| Start | `pretrained_model_for_share/CSFM_base.pth` encoder and decoder | `[source]` |
| Input | 10s at 250 Hz, 0.1s patches (25 samples), per-channel z-score, NeuroKit cleaning | `[source]` |
| Modalities | all available channels of a window, flattened; absent modalities are not filled | `[source]` Fig. 6b |
| Masking | random 75% of signal tokens | `[source]` |
| Loss | MSE on masked patches only, against per-patch normalized targets `(x−mean)/sqrt(var+1e-6)` | `[source]` Eq. 2; normalization verified on the checkpoint, see below |
| Text | compared as arms A and B below | `[open]` |
| New parameters | Resp encoder channel embedding, decoder channel embedding, reconstruction head | `[extension]` |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.05, effective batch 256 (64 on each of 4 GPUs) | `[source]` |
| LR | base 1e-4 with layer-wise decay 0.75; new parameters at base LR | `[source]` downstream setting, reused here |
| Schedule | cosine with 5% warmup; validate every 2,500 steps; early stopping | `[source]` |
| Windows | train patients, one valid window per ICU hour, resampled each epoch | `[engineering]` |
| Selection | mean of ECG, PPG, Resp masked validation MSE on the same fixed signal-only windows in both arms, so text cannot drive selection | `[engineering]`; the paper says only "validation performance" |
| Saved | best overall and best Resp, each with decoder, plus last | `[engineering]` |

Pre-training check, done (`finetune/arms/check/check_mae.py`): the local MAE forward reproduces the checkpoint once targets are per-patch normalized. On 512 ICU windows, original-checkpoint MSE was ECG 0.92 raw versus 0.34 normalized and PPG 1.40 raw versus 0.18 normalized, against 1.0 for zero prediction. 10.17 trained against raw targets, which the pretrained decoder never used. The paper's Eq. 2 does not state this normalization, and the MAE forward is not public, so this implementation stays a reconstruction.

Code: `10.30/sofa2_icu_pipeline/csfm/`. `queue_gpus.py` starts both arms on the shared GPU 3 (batch 32 × accumulation 8) and moves them onto free GPUs as they appear, growing each arm from 1 to 2 to 4 GPUs. Moves checkpoint first, and effective batch stays 256. Other users' processes are never touched. NCCL peer-to-peer hangs on this host, so `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` are required.

### Text arms

10.17 text was mostly Heart Rhythm (90.1% of windows), 66.6% of windows had one field, and none had respiratory rate, so it offered Resp little to learn.

| Arm | Text | Windows |
|---|---|---|
| A | none; the paper allows signal-only samples (Eq. 1) | one random valid window per ICU hour |
| B | charted vital-sign snapshot `[extension]` | the valid 10s slot containing a charted snapshot, one per ICU hour |

Arm B uses real charted values only, as `<label>: <value>.` up to 64 tokens, with 50% token masking and masked-token cross entropy (Eq. 3) added to the signal MSE `[source]`.

| Modality | Fields (itemid) | Rows in waveform-patient stays |
|---|---|---:|
| ECG | Heart Rate (220045), Heart Rhythm (220048), Ectopy Type (224650) | 1.40M, 1.27M, 1.12M |
| PPG | SpO2 (220277), noninvasive mean BP (220181) | 1.37M, 0.92M |
| Resp | Respiratory Rate (220210), Breathing pattern (229322), O2 device (226732) | 1.38M, 0.25M, 0.34M |
| Ventilated patients | Ventilator mode (223849, 229314), FiO2 (223835), PEEP (220339), tidal volume (224685) | 0.11–0.18M |

HR, RR and SpO2 share a charttime 1.25M times, about hourly, before intersecting with waveform.

Arm B limits:

- The paper's text is ECG reports; vital-sign text is our adaptation.
- Numeric tokens must match exactly, so text accuracy may be low.
- The local BERT-uncased tokenizer is an unconfirmed assumption.
- Machine ECG reports match the paper's text type but only 4,806 fall inside ICU and waveform time, so they are excluded from this run.

### Encoder choice

Compare original, 10.17 epoch 17, arm A and arm B as frozen CLS features with the same XGBoost budget. The goal is to show that the original CSFM abilities are retained and that Resp tasks improve. Task types follow the CSFM paper, on our own continuous waveform and EHR data. SOFA is not a probe `[changed 2026-11-01]`. Code and method: `encoder_comparison/`.

Windows are one 10-second slot per ICU hour on each train or validation patient's first qualifying day, holding a charted vital when the hour has one. Patient tasks mean-pool the CLS over the day. Window tasks use up to 6 evenly spaced labeled windows per patient. Predictors fit on 90% of train patients, thresholds come from the other 10%, and scores come from validation patients with a paired patient bootstrap. The test split is not read.

| Part | Paper task type | Our tasks | Reference | Margin |
|---|---|---|---|---:|
| Retained | Rhythm classification | SR, ST, SB, AF, paced, macro-F1 | Original | 0.02 |
| Retained | Cardiac diagnosis | Eight ICD groups, macro-F1; PVC, PAC, first-degree block, AUROC; QTc, MAE | Original | 0.02; 3 ms |
| Retained | Demographics | Age, BMI, MAE; sex, AUROC | Original | 0.5 years; 0.25; 0.02 |
| Retained | Vital signs | Invasive SBP and DBP, SpO2, MAE, as scalar regression | Original | 1 mmHg; 0.25% |
| Retained | Clinical outcome | 1-year, 30-day and hospital death, AUROC | Original | 0.02 |
| Resp `[extension]` | None in the paper | Monitor RR, MAE; invasive support, AUROC; ICD respiratory diagnoses, macro-AUROC | Epoch 17 | See below |

Retention verdicts: lost when the point estimate is worse than the margin and the 95% CI excludes zero; retained when the whole CI is inside the margin; otherwise inconclusive. The margins are project choices.

An arm passes with no lost check, with RR and invasive support not worse than epoch 17 on all three channels, and with at least one of them better. Arm B must also not be worse on ICD respiratory diagnoses, because its training text holds RR, breathing pattern, O2 device and ventilator fields, so charted Resp labels favor it. Both pass: arm A, unless arm B is clearly better on RR or support. No arm passes: decided by the study team. Epoch 17 is reported against the same checks and is no longer the default, because 10.17 already showed lower 12-lead sex AUROC. If epoch 17 is chosen in the end, the existing 3min epoch 18 encoder can be reused and section 6 is skipped.

### Encoder choice on the new split `[changed 2026-11-06]`

**Chosen: arm A.** This departs from the rule above and is reported as a decision made after the results.

Arms A, B, B2 and B3 (`finetune/README.md`, section 3.10) were compared with original CSFM as the reference, on validation patients with a paired patient bootstrap of 1,000 draws. Results: `encoder_comparison/results/report.md`.

| | Original | A | B | B2 | B3 |
|---|---:|---:|---:|---:|---:|
| Monitor RR MAE, three channels | 4.21* | **3.06** | 3.81 | 3.81 | 4.00 |
| Monitor RR MAE, Resp alone | 4.21* | **3.06** | 3.23 | 3.17 | 3.36 |
| Invasive support AUROC, three channels | 0.738* | **0.810** | 0.779 | 0.789 | 0.778 |
| Heart rate MAE | 3.98 | **3.46** | 4.33 | 4.47 | 4.32 |
| Right bundle branch block AUROC | 0.610 | 0.569 | 0.726 | 0.792 | **0.830** |
| Sex from ECG AUROC | 0.684 | 0.652 | **0.708** | 0.703 | 0.678 |
| Retention checks lost, of 31 | Reference | 1 | 0 | 0 | 0 |

\* Original CSFM has no Resp channel; ECG + PPG.

The rule picks arm B, because arm A lost one check: sex from ECG alone, AUROC −0.032 (95% CI −0.062 to −0.002). Arm A is chosen instead because:

1. **The goal is Resp.** Arm A is the best encoder on every Resp task and on heart rate, and every text arm is worse than A on them.
2. **The text arms' gains are on text content.** Against A, they are better only on findings that ECG machine text describes (right bundle branch block, first-degree block, QTc, PAC). On tasks whose labels share nothing with the text (ICD groups, deaths, age, sex, BMI) they tie with A: B is better on 2, the same on 22 and worse on 1. Giving the final text arms their text does not change their signal reconstruction.
3. **The lost check is one of 31, without multiplicity correction,** and sex is neither an input nor an outcome of the SOFA-2 task.
4. **The rule compared A with B only.** B2 and B3 were added after the first results.

Cost, reported as limits: arm A is below original CSFM on sex from ECG (−0.032) and on right bundle branch block (−0.041, not a gate check).

### Rerun on the new split `[changed 2026-11-03]`

Everything above in this section describes the run behind the current results. The rerun changes the following; open items are marked `[open]`. The CSFM statements are checked against the published Methods, Gu et al., Nat Mach Intell 8, 220–233 (2026). The full design, its sources and the options not taken are in `finetune/README.md`.

| Item | Rerun | Source |
|---|---|---|
| Arm A | Never has text | `[source]` Eq. 1, signal-only samples |
| Arm B | Samples with and without text are mixed, as in Eq. 1. Windows are the same as arm A's, one random valid window per ICU hour, so the arms differ only in text | `[source]` Eq. 1; same windows `[engineering]` |
| When a window has text | A record within ±15 minutes for vitals and nursing assessments, ±12 hours for chest imaging, ±24 hours for ECG reports. 82.4% of training windows get text `[engineering]` | The paper pairs MIMIC-III ECG reports within one week and calls it a weak match `[source]` |
| Text sources | Charted vitals; charted nursing assessments (lung sounds, respiratory pattern and effort, cough, sputum, airway, ventilator mode, cardiovascular and neurological items); chest imaging reports from MIMIC-IV-Note; ECG machine reports from MIMIC-IV-ECG. Each sample carries one available source, drawn at random each epoch, cut to 64 tokens at a word boundary `[engineering]` | Sources `[extension]`; the paper's text is ECG reports |
| Text masking | Whole word masking as in BERT, below | `[source]` |
| Readout downstream | CLS token, also in the three levels | `[source]` Methods, "global class token" |
| Seeds | One first; the paper uses three for downstream results | `[engineering]` |
| Length | 60,000 updates for both arms, cosine decay to zero, no early stopping | `[engineering]` |
| Checkpoint | Probes on CLS for 10% of train patients held out from fine-tuning, every 2,500 updates. Keep checkpoints that hold ECG and PPG tasks within the comparison margins, then take the lowest respiratory rate error | `[engineering]` |
| Downstream tasks | RR, invasive support, ICD respiratory diagnoses, SpO2, SBP and DBP, rhythm, HR, 24-hour and hospital death. New: Resp reconstructed from ECG + PPG with Resp fully hidden, like the paper's cross-modal reconstruction | Tasks kept for their link to SOFA-2 organs `[engineering]` |

Cover of cohort ICU days by text type: chest imaging report on the same ICU day 56.0%, within 24 hours 84.4%; ECG machine report 12.5% and 25.5%; charted vitals in 98.3% of training hours. Discharge summaries are left out because they cover a whole admission and hold its outcome. ICD codes are left out because they are probe labels.

**Whole word masking.** Checked against `create_pretraining_data.py` in google-research/bert, commit `eedf571`, and its README entry of 2019-05-31. Code: `finetune/arms/train/finetune.py`, `whole_word_mask`.

- As in BERT: candidates are all tokens except `[CLS]` and `[SEP]`. A token starting with `##` joins the word before it. Words are shuffled. The budget is `max(1, round(len(tokens) × rate))`, where `len(tokens)` counts `[CLS]` and `[SEP]`. Words are taken in shuffled order, and a word that would pass the budget is skipped. Each masked piece is predicted on its own.
- Different from BERT, following CSFM: the rate is CSFM's 50%, not BERT's 15%. There is no 20-prediction cap. Masked tokens are dropped from the encoder as in CSFM, not replaced 80/10/10.

## 6. Decisions of 2026-11-02

The levels below were reset before training, after the encoder comparison and a label review.

- **Encoder:** arm B `[changed 2026-11-06: arm A on the new split, section 5, "Encoder choice on the new split"]`. It lost no retention check and improved RR and invasive support against original CSFM. The Resp reference moved from epoch 17 to original CSFM after the first results were seen, so this is reported as a post hoc choice.
- **Added waveform:** the index `endtime` drops 8.1% of waveform hours, and that data looks valid. `plan` adds the 1,005 days that pass with records ended at their master header to the same plan and cut, after the 22,126 index-end days, which are unchanged. The encoders were trained without them; the rerun on the new split uses them in fine-tuning and the encoder comparison too `[changed 2026-11-03]`.
- **Target:** ordinal future organ score, not direct delta regression.

## 7. Three training levels

Code: `hierarchy/`. Lower levels stay frozen. Only level 3 has a SOFA-2 loss.

`[changed 2026-11-05]` Levels 1 and 2 also have a masked-prediction loss in place of the paper's task loss: 75% of windows or clips hidden, a light decoder predicts the frozen features of the level below, centered within their clip or hour, MSE on hidden ones only. Level 2 and 3 read inner tokens. Stopping uses the validation masked loss. The current design, with sources, is `hierarchy/README.md`, section 3; the table below is the earlier version.

| Level | Input | Loss | Selection |
|---|---|---|---|
| 1, 3 minutes | 18 10-second slots over the frozen chosen encoder, arm A `[changed 2026-11-06]` | local + dynamic | Validation SSL loss |
| 2, 1 hour | 20 three-minute slots | local + dynamic; SSL needs at least 16 of 20 valid | Validation SSL loss |
| 3, 24 hours | 24 one-hour slots | SOFA-2 task + 0.1 local + 0.1 dynamic | Validation task loss |

Level 1 follows the structure and losses of Tang et al. 2026, Eqs. 1 to 3 `[source]`, used without its task loss from Eq. 4 `[extension]`. The paper trains these losses only beside a task loss and gives no numerical settings. Ours are `[engineering]`: loss weights 1, 8 prompts, 18 global positions, 256-d projection heads, temperature 0.1, high-affinity fraction 0.5, AdamW LR 3e-4, weight decay 0.01, pool 16 (batch 8 on 2 GPUs), accumulation 2. Eq. 3 is implemented as displayed, on high-affinity slots only. A clip allows one bad slot: its tokens are dropped and the other slots keep their positions `[extension]`. Training uses clean 18-slot clips; features cover all valid clips.

Level 1 budget `[engineering]`: at most 40,000 clips per epoch from different ICU hours, redrawn each epoch, 30 epochs, warmup 2, stop after 5 epochs with less than 0.1% validation improvement. Measured speed on GPUs 3 and 7 is about 6 clips per second, so one clip per ICU hour would take 17 hours per epoch. Teachers are computed online, not cached. Gloo timeout is raised to 30 minutes, which fixes the failure in the last run.

Levels 2 and 3 use the 10.24 temporal model: 256 hidden, 2 blocks, 8 heads, 768-d token readout, learned positions, training-only input normalization, explicit masks for missing slots. AdamW LR 3e-4, weight decay 0.01, batch 32, at most 100 epochs, warmup 7, patience 10.

## 8. SOFA-2 in the model

`[changed 2026-11-05]` The head is now CORN organ outputs plus a composite output, with the E2 baseline path (previous-day organ scores, ICU day, hourly HR, RR, SpO2 and MAP); the Huber total term is dropped. `L_SOFA` is the organ mean of CORN plus composite cross entropy, with 0.1 L_local and 0.1 L_dyn. Current design: `hierarchy/README.md`, section 3.6, and `labels/README.md`, section 7. The table below is the earlier version.

| Item | Setting |
|---|---|
| Output | P(score > k), k = 0 to 3, for each of six organs on day `t` to `t+24h` `[extension]` |
| Delta | Expected future score, the sum of those probabilities, minus baseline |
| Baseline input | Six organ scores divided by 4 plus their masks. Missing organs are zero with a false mask |
| Structure | Baseline path plus a waveform correction on the same logits. The correction starts at zero `[extension]` |
| Organ loss | Binary cross entropy on the cumulative targets, masked to organs with a real baseline and a future score, averaged per organ |
| Total loss | Huber on the summed change of the organs labeled on that anchor |
| Anchors | All. Death and discharge are scored as in the fate table above |
| Sampling | Each patient has equal total weight `[engineering]`, since anchors range from 1 to 69 per patient |

The official total is an evaluation target, not a training target. Its carried values force part of the change to 0, so it does not describe true change. Only 13% of anchors have six real organs.

Controls on identical masks: persistence (delta 0), baseline only, waveform only. Before level 2, `staged_check.py` tests current score, past change and future change from mean-pooled three-minute features, so a missing signal is found early.

Input-length comparison stays planned: 3 minutes, 5 minutes, 12 hours, 24 hours `[extension]`.

## 9. Evaluation `[open: task 2]`

Primary and secondary metrics are decided by task 2. Provisional reporting:

- Organ and total: MAE, RMSE, and MAE on changed anchors only. MAE alone rewards predicting zero.
- Worsening delta ≥ 2 and the composite: AUROC, AUPRC, calibration.
- Direction (down, same, up): balanced accuracy.
- Patient-level bootstrap with 1,000 draws for every interval.
- Liver is reported with its 28.2% coverage and not used for selection until its label count is reviewed.
- Provisional checkpoint selection: validation masked Huber over organs and total.

## 10. Run order

| Step | Work | Output | Gate before continuing |
|---|---|---|---|
| 1 | Finalize labels (section 3) | `labels/` | SOFA-1 comparison explained |
| 2 | Lock cohort, assign new patients | `labels/` | Attrition table reviewed |
| 3 | Cut waveform (section 4) | `waveform/` | Coverage and QC audited |
| 4 | CSFM arms A and B, then probes | `finetune/`, `encoder_comparison/` | Encoder chosen |
| 5 | 3min SSL and features | `hierarchy/` | Stable validation loss, non-degenerate features |
| 6 | Staged check with mean-pooled features | `hierarchy/` | Signal for current organ score |
| 7 | 1h SSL and features | `hierarchy/` | Stable validation loss |
| 8 | 24h task and controls | `hierarchy/` | Validation comparison |
| 9 | Lock setting, run test once | `final/` | None |

Step 6 sits before the 1h and 24h levels so a missing signal is found before more training.

## 11. References

- SOFA-2, Ranzani et al., JAMA 2025: https://jamanetwork.com/journals/jama/fullarticle/2840822
- Lambden et al., Crit Care 2019: https://doi.org/10.1186/s13054-019-2663-7
- de Grooth et al., Crit Care 2017: https://doi.org/10.1186/s13054-017-1609-1
- AdrenOSS-2 trial protocol v4.2, 2019, NCT03085758: https://cdn.clinicaltrials.gov/large-docs/58/NCT03085758/Prot_000.pdf
- Helleberg et al., Crit Care 2026: https://doi.org/10.1186/s13054-026-06093-8
- Seymour et al., JAMA 2016: https://jamanetwork.com/journals/jama/fullarticle/2492875
- Octogenarian ICU study, PLOS ONE: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8328283/
- Tang et al., 2026: https://arxiv.org/abs/2605.16975
- Waveform-derived features for septic shock onset: https://pmc.ncbi.nlm.nih.gov/articles/PMC10977921
- MIT-LCP mimic-code: https://github.com/MIT-LCP/mimic-code
- CSFM, Gu et al., Nature Machine Intelligence 2026: https://doi.org/10.1038/s42256-026-01180-5
- BERT whole word masking, google-research/bert README and `create_pretraining_data.py`, commit eedf571: https://github.com/google-research/bert
- CSFM code and fine-tune tutorial: https://github.com/guxiao0822/Cardiac-Sensing-FM

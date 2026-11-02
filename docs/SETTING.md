# ICU waveform → SOFA-2 change: full pipeline setting

Version: draft 3, 2026-10-30. Status: labelled SOFA-2 pilot in section 3 done; waveform planning done; full waveform cut started (section 4). Any-ICU-stay inclusion and the CSFM text comparison are confirmed.

Location: paths in this document are the original working folders. In this repository, the same code is under `labels/`, `waveform/`, `finetune/` and `encoder_comparison/`.

Tags used below:

- `[文献]` taken from a published source
- `[沿用]` kept from an existing project experiment
- `[工程]` engineering choice with no scientific source
- `[扩展]` our own method extension, to be stated as such in any paper
- `[待定]` open; the default here is provisional

## 1. Task

Predict, from ECG/PPG/Resp only plus baseline SOFA-2, the change in each of the six organ scores and the total over the next 24 hours, using ICU data only.

For prediction time `t`:

| Item | Interval | Note |
|---|---|---|
| Waveform input | `[t−24h, t)` | Length is compared in section 8 `[扩展]` |
| Baseline SOFA-2 | worst per organ in `[t−24h, t)` | SOFA-2 Table 2 `[文献]` |
| Future SOFA-2 | worst per organ in `[t, t+24h)` | same |
| Target | future − baseline | fixed-interval delta, de Grooth 2017 `[文献]` |
| Worsening | total or organ delta ≥ 2 | Sepsis-3 threshold `[文献]` |

`t = intime + 24k` hours, `k ≥ 1`, so ICU days are `[intime+24d, intime+24(d+1))` as in Helleberg 2026 `[文献]`. Denser anchors are a later sensitivity analysis.

There is no gap between input end and future window. Baseline scores given to the model use only records with `storetime ≤ t`. Target scores use `storetime ≤ t+24h`.

## 2. Cohort

Sources:

- EHR: `/home/public/mimiciv/3.1/` (`icu/`, `hosp/`)
- Waveform: `/home/public/MIMIC_waveform/mimic_iv_waveform_v0_2_prerelease/v0.2`

Rules, applied in this order with counts reported at each step:

1. Patient has a waveform record: 7,306 patients with ICU stays.
2. Age ≥ 18, any ICU stay: 13,604 stays. A later stay in the same admission is allowed; patient-level splitting and bootstrap handle repeated stays. Sensitivity: first ICU stay of each admission.
3. ICU day `d` is complete and `t < outtime`: 47,730 days.
4. WFDB headers can still reach ≥ 384 of 480 three-minute slots `[沿用: 80% rule]`: 22,126 days, 6,527 stays, 5,497 patients. This is an upper bound until step 5.
5. After signal QC, ≥ 384 of 480 three-minute slots are valid (section 6). 80% is total coverage, not one unbroken stretch; missing slots stay in place and are masked. In 400 sampled 12h windows, total coverage kept 82.2% and one unbroken 80% stretch kept 25.5%.

Not cut by CXR. CXR is neither an anchor nor an input.

Fate of each anchor over `[t, t+24h)` (header plan, before signal QC and the added days; final counts are in `COHORT.md`):

| Fate | Anchors | Share | Use |
|---|---:|---:|---|
| Stayed in ICU | 16,337 | 73.8% | delta regression and classification |
| Left ICU alive | 5,161 | 23.3% | excluded from delta; reported; sensitivity analysis |
| Died | 628 | 2.8% | counted as worsening in the composite; excluded from delta regression |

Primary handling follows the exclusion-with-sensitivity pattern of the octogenarian ICU study. The composite "delta ≥ 2 or death within 24h" adapts the worst-rank idea and is `[扩展]`. Sensitivity: death = 24, discharge = 0 or last value (de Grooth 2017) `[文献]`.

### Split

- Keep existing patient assignment from `10.16/csfm_ehr_text_coverage/output/pairs.csv` `[沿用]`.
- Patients without an assignment are assigned once by `SHA256(subject_id)` to 70/15/15 `[工程]`, before any model sees them. Planned days: train 15,581, val 3,268, test 3,277.
- No patient crosses splits. All encoder and SSL training uses train patients only.
- SOFA test results are computed once, after this setting is locked.

## 3. SOFA-2 labels

Labels are final as of 2026-11-01: code in `labels/`, patient-level tables in `labels/private/`, aggregates in `labels/cohort_report.json`, summary in `COHORT.md`. Numbers from the earlier first-stay pilot further down this section are superseded by `COHORT.md`.

Thresholds and notes: SOFA-2 Table 2 (Ranzani, JAMA 2025) `[文献]`. The 4-point cardiovascular row prints ">.02", also in the published version; use 0.2 to match the 3-point row.

Extraction follows MIT-LCP `mimic-code` `[文献]`:

| Organ | Sources |
|---|---|
| Brain | GCS 220739, 223900, 223901; motor-only if domains are unavailable; last pre-sedation GCS if sedated |
| Respiratory | arterial PaO2 50821 and FiO2 50816 by specimen; chart FiO2 223835 within 4h; SpO2/FiO2 only if PaO2/FiO2 is unavailable and SpO2 < 98%; support from `ventilation.sql` |
| Cardiovascular | MAP 220052, 220181, 225312; norepinephrine + epinephrine sum; other agents; ECMO, IABP, Impella, LVAD, TandemHeart |
| Liver | bilirubin 50885 |
| Kidney | creatinine 50912; urine from `urine_output.sql`; RRT from `rrt.sql` |
| Hemostasis | platelets 51265 |

Done for the final labels:

1. RRT follows `rrt.sql` active items.
2. Respiratory support follows `ventilation.sql`.
3. `patientweight = 1` follows `norepinephrine.sql`.
4. A record counts for a day only if stored by that day's end.
5. SOFA-1 comparator uses the same evidence, with SpO2/FiO2 mapped by Pandharipande 2009 `[工程]`.
6. The pilot's excess of higher SOFA-2 scores came mostly from a weaker SOFA-1 comparator. On the same evidence SOFA-2 is lower in 44.0%, the same in 28.7% and higher in 27.3%, per patient on day 1.

Project approximations to report `[工程]`:

- Sedatives: propofol, midazolam, dexmedetomidine, ketamine, lorazepam infusions.
- Delirium drugs: haloperidol, quetiapine, olanzapine administrations.
- Intermittent RRT stays at 4 for 72h after the last session.
- Urine rolling calculation with a 6h maximum record gap.
- Vasopressor fragments are joined per drug; total ≥ 60 minutes in the window.
- MAP is the daily minimum.
- Salt-versus-base norepinephrine is unknown. Halving the dose changes 4.0% of cardiovascular day scores. Report as a sensitivity analysis.
- Note p, RRT criteria without RRT: only the creatinine trigger is used. Adding oliguria changes 55 of 61,334 day scores (0.09%), so the labels keep the simpler form.
- Not implemented: excluding PaO2/FiO2 changes within 1 hour (note g), ceiling-of-treatment scoring (notes h, m), excluding nonrenal RRT (note o).

### Missing values

Every organ-day stores `score`, `evidence` (`real`, `presedation`, `carried`, `none`) and rule version.

- Organ delta: both ends `real`. Otherwise masked. A carried future value would make that organ's delta 0 by construction: on the 21,360 anchors, 66.5% of liver zeros and 28.1% of respiratory zeros under the official rule.
- Total delta, main: official SOFA-2 rule, Table 2 note b `[文献]`. Missing on ICU day 1 scores 0; later missing values carry the last observation. 16,670 anchors, 83.2% real endpoints.
- Total delta, sensitivity: at most one carried window per organ (Lambden 2019) `[文献]`, 4,910 anchors, and all six real, 2,227 anchors. Unchanged 28.7%, 28.2% and 27.0%; delta ≥ 2 13.4%, 14.5% and 15.6%.
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

Reuse the 10.14 joint CSFM preprocessing unchanged `[沿用]`:

| Item | Setting |
|---|---|
| ECG | native 250 Hz, `ecg_clean` (neurokit, powerline 50), lead priority II, I, III, aVR, aVL, aVF, V1–V6, true lead ID kept |
| PPG | native 125 Hz, FFT to 250 Hz, `ppg_clean` (elgendi) |
| Resp | native 62.5 Hz, FFT to 250 Hz, `rsp_clean` (khodadad2018) |
| Output | float32 `[N,3,2500]`, per-channel z-score, modality mask, channel IDs; ECG 0–11, PPG 12, Resp 13 |
| Rules | complete 10s only; no stitching across gaps or admissions |

New in this run:

- **Grid:** slot `k` is `[intime + 10k s, intime + 10(k+1) s)` `[工程]`, so 18 slots make 3min, 360 make 1h and 8,640 make 24h, all aligned to `t`.
- **Scope:** only ICU days that are anchors in section 2, regardless of whether SOFA can be computed for them.
- **Storage:** one `signal.npy` and one `meta.npz` per ICU day, replacing the 10.14 SQLite index. Each worker writes its own day; the meta file marks it finished, so a stopped run resumes by rerunning.
- **Code:** `10.30/sofa2_icu_pipeline/waveform_cut/`, importing the 10.14 `process_window` and `read_window` and the 10.13 header inventory unchanged. 200 sampled 10.14 windows re-cut through this code were byte-equal.
- **Output path:** `10.30/sofa2_icu_pipeline/waveform/`. Old 10.14 outputs are read only.

Planned volume: 186.5M windows, about 5.5 TB. Measured about 1,645 ICU days per hour with 64 workers, so about 13.5 hours. This step is CPU-bound in NeuroKit cleaning; a GPU rewrite would break byte equality with earlier data.

## 5. CSFM ten-second fine-tune

Goal: add Resp to the CSFM encoder with the paper's masked objective. The paper has no Resp channel and no self-supervised fine-tune step, so this stage as a whole is `[扩展]`; each part below stays as close to Gu et al. 2026 as possible.

| Item | Setting | Source |
|---|---|---|
| Start | `pretrained_model_for_share/CSFM_base.pth` encoder and decoder | `[文献]` |
| Input | 10s at 250 Hz, 0.1s patches (25 samples), per-channel z-score, NeuroKit cleaning | `[文献]` |
| Modalities | all available channels of a window, flattened; absent modalities are not filled | `[文献]` Fig. 6b |
| Masking | random 75% of signal tokens | `[文献]` |
| Loss | MSE on masked patches only, against per-patch normalized targets `(x−mean)/sqrt(var+1e-6)` | `[文献]` Eq. 2; normalization verified on the checkpoint, see below |
| Text | compared as arms A and B below | `[待定]` |
| New parameters | Resp encoder channel embedding, decoder channel embedding, reconstruction head | `[扩展]` |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.05, effective batch 256 (64 on each of 4 GPUs) | `[文献]` |
| LR | base 1e-4 with layer-wise decay 0.75; new parameters at base LR | `[文献]` downstream setting, reused here |
| Schedule | cosine with 5% warmup; validate every 2,500 steps; early stopping | `[文献]` |
| Windows | train patients, one valid window per ICU hour, resampled each epoch | `[工程]` |
| Selection | mean of ECG, PPG, Resp masked validation MSE on the same fixed signal-only windows in both arms, so text cannot drive selection | `[工程]`; the paper says only "validation performance" |
| Saved | best overall and best Resp, each with decoder, plus last | `[工程]` |

Pre-training check, done (`csfm/check_mae.py`): the local MAE forward reproduces the checkpoint once targets are per-patch normalized. On 512 ICU windows, original-checkpoint MSE was ECG 0.92 raw versus 0.34 normalized and PPG 1.40 raw versus 0.18 normalized, against 1.0 for zero prediction. 10.17 trained against raw targets, which the pretrained decoder never used. The paper's Eq. 2 does not state this normalization, and the MAE forward is not public, so this implementation stays a reconstruction.

Code: `10.30/sofa2_icu_pipeline/csfm/`. `queue_gpus.py` starts both arms on the shared GPU 3 (batch 32 × accumulation 8) and moves them onto free GPUs as they appear, growing each arm from 1 to 2 to 4 GPUs. Moves checkpoint first, and effective batch stays 256. Other users' processes are never touched. NCCL peer-to-peer hangs on this host, so `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` are required.

### Text arms

10.17 text was mostly Heart Rhythm (90.1% of windows), 66.6% of windows had one field, and none had respiratory rate, so it offered Resp little to learn.

| Arm | Text | Windows |
|---|---|---|
| A | none; the paper allows signal-only samples (Eq. 1) | one random valid window per ICU hour |
| B | charted vital-sign snapshot `[扩展]` | the valid 10s slot containing a charted snapshot, one per ICU hour |

Arm B uses real charted values only, as `<label>: <value>.` up to 64 tokens, with 50% token masking and masked-token cross entropy (Eq. 3) added to the signal MSE `[文献]`.

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

Compare original, 10.17 epoch 17, arm A and arm B as frozen CLS features with the same XGBoost budget. The goal is to show that the original CSFM abilities are retained and that Resp tasks improve. Task types follow the CSFM paper, on our own continuous waveform and EHR data. SOFA is not a probe `[修改 2026-11-01]`. Code and method: `encoder_comparison/`.

Windows are one 10-second slot per ICU hour on each train or validation patient's first qualifying day, holding a charted vital when the hour has one. Patient tasks mean-pool the CLS over the day. Window tasks use up to 6 evenly spaced labeled windows per patient. Predictors fit on 90% of train patients, thresholds come from the other 10%, and scores come from validation patients with a paired patient bootstrap. The test split is not read.

| Part | Paper task type | Our tasks | Reference | Margin |
|---|---|---|---|---:|
| Retained | Rhythm classification | SR, ST, SB, AF, paced, macro-F1 | Original | 0.02 |
| Retained | Cardiac diagnosis | Eight ICD groups, macro-F1; PVC, PAC, first-degree block, AUROC; QTc, MAE | Original | 0.02; 3 ms |
| Retained | Demographics | Age, BMI, MAE; sex, AUROC | Original | 0.5 years; 0.25; 0.02 |
| Retained | Vital signs | Invasive SBP and DBP, SpO2, MAE, as scalar regression | Original | 1 mmHg; 0.25% |
| Retained | Clinical outcome | 1-year, 30-day and hospital death, AUROC | Original | 0.02 |
| Resp `[扩展]` | None in the paper | Monitor RR, MAE; invasive support, AUROC; ICD respiratory diagnoses, macro-AUROC | Epoch 17 | See below |

Retention verdicts: lost when the point estimate is worse than the margin and the 95% CI excludes zero; retained when the whole CI is inside the margin; otherwise inconclusive. The margins are project choices.

An arm passes with no lost check, with RR and invasive support not worse than epoch 17 on all three channels, and with at least one of them better. Arm B must also not be worse on ICD respiratory diagnoses, because its training text holds RR, breathing pattern, O2 device and ventilator fields, so charted Resp labels favor it. Both pass: arm A, unless arm B is clearly better on RR or support. No arm passes: decided by the study team. Epoch 17 is reported against the same checks and is no longer the default, because 10.17 already showed lower 12-lead sex AUROC. If epoch 17 is chosen in the end, the existing 3min epoch 18 encoder can be reused and section 6 is skipped.

## 6. Decisions of 2026-11-02

The levels below were reset before training, after the encoder comparison and a label review.

- **Encoder:** arm B. It lost no retention check and improved RR and invasive support against original CSFM. The Resp reference moved from epoch 17 to original CSFM after the first results were seen, so this is reported as a post hoc choice.
- **Added waveform:** the index `endtime` drops 8.1% of waveform hours, and that data looks valid. 1,005 added days were planned with the master header end and cut to `waveform/icu_cut/output_tail/`. The original 22,126 days are unchanged, and the encoders are not retrained.
- **Target:** ordinal future organ score, not direct delta regression.

## 7. Three training levels

Code: `hierarchy/`. Lower levels stay frozen. Only level 3 has a SOFA-2 loss.

| Level | Input | Loss | Selection |
|---|---|---|---|
| 1, 3 minutes | 18 10-second slots over frozen arm B | local + dynamic | Validation SSL loss |
| 2, 1 hour | 20 three-minute slots | local + dynamic; SSL needs at least 16 of 20 valid | Validation SSL loss |
| 3, 24 hours | 24 one-hour slots | SOFA-2 task + 0.1 local + 0.1 dynamic | Validation task loss |

Level 1 follows Tang et al. 2026 `[文献]`, used without a task loss `[扩展]`: 8 prompts, 18 global positions, 256-d projection heads, temperature 0.1, high-affinity fraction 0.5, AdamW LR 3e-4, weight decay 0.01, pool 16 (batch 8 on 2 GPUs), accumulation 2. A clip allows one bad slot: its tokens are dropped and the other slots keep their positions `[扩展]`. Training uses clean 18-slot clips; features cover all valid clips.

Level 1 budget `[工程]`: at most 40,000 clips per epoch from different ICU hours, redrawn each epoch, 30 epochs, warmup 2, stop after 5 epochs with less than 0.1% validation improvement. Measured speed on GPUs 3 and 7 is about 6 clips per second, so one clip per ICU hour would take 17 hours per epoch. Teachers are computed online, not cached. Gloo timeout is raised to 30 minutes, which fixes the failure in the last run.

Levels 2 and 3 use the 10.24 temporal model: 256 hidden, 2 blocks, 8 heads, 768-d token readout, learned positions, training-only input normalization, explicit masks for missing slots. AdamW LR 3e-4, weight decay 0.01, batch 32, at most 100 epochs, warmup 7, patience 10.

## 8. SOFA-2 in the model

| Item | Setting |
|---|---|
| Output | P(score > k), k = 0 to 3, for each of six organs on day `t` to `t+24h` `[扩展]` |
| Delta | Expected future score, the sum of those probabilities, minus baseline |
| Baseline input | Six organ scores divided by 4 plus their masks. Missing organs are zero with a false mask |
| Structure | Baseline path plus a waveform correction on the same logits. The correction starts at zero `[扩展]` |
| Organ loss | Binary cross entropy on the cumulative targets, masked to organs with real values at both ends, averaged per organ |
| Total loss | Huber on the summed change of the organs labeled on that anchor |
| Anchors | Stayed in ICU. Death and discharge add no task loss |
| Sampling | Each patient has equal total weight `[工程]`, since anchors range from 1 to 69 per patient |

The official total is an evaluation target, not a training target. Its carried values force part of the change to 0, so it does not describe true change. Only 13% of anchors have six real organs.

Controls on identical masks: persistence (delta 0), baseline only, waveform only. Before level 2, `staged_check.py` tests current score, past change and future change from mean-pooled three-minute features, so a missing signal is found early.

Input-length comparison stays planned: 3 minutes, 5 minutes, 12 hours, 24 hours `[扩展]`.

## 9. Evaluation `[待定: task 2]`

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
| 2 | Lock cohort, assign new patients | `cohort/` | Attrition table reviewed |
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
- Helleberg et al., Crit Care 2026: https://doi.org/10.1186/s13054-026-06093-8
- Seymour et al., JAMA 2016: https://jamanetwork.com/journals/jama/fullarticle/2492875
- Octogenarian ICU study, PLOS ONE: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8328283/
- Tang et al., 2026: https://arxiv.org/abs/2605.16975
- Waveform-derived features for septic shock onset: https://pmc.ncbi.nlm.nih.gov/articles/PMC10977921
- MIT-LCP mimic-code: https://github.com/MIT-LCP/mimic-code
- CSFM, Gu et al., Nature Machine Intelligence 2026: https://doi.org/10.1038/s42256-026-01180-5
- CSFM code and fine-tune tutorial: https://github.com/guxiao0822/Cardiac-Sensing-FM

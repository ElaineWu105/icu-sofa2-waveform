# Cohort summary: ICU waveform to SOFA-2 change

Date: 2026-11-02. Data: MIMIC-IV 3.1 and the MIMIC-IV Waveform Database v0.2 prerelease. ICU data only.

Regenerate the numbers with `labels/run.sh`; aggregate output is in `labels/cohort_report.json`. Full method: `SETTING.md`.

## Definition

**Unit of analysis:** one prediction time `t` in one ICU stay, called an anchor. `t = intime + 24k` hours with `k ≥ 1`.

| Item | Interval |
|---|---|
| Input | ECG, PPG and Resp in `[t−24h, t)`, plus baseline SOFA-2 |
| Baseline | Worst SOFA-2 per organ in `[t−24h, t)` |
| Target | Worst SOFA-2 per organ in `[t, t+24h)`, minus baseline |

Anchors are repeated within patients. Splits and uncertainty are therefore patient-level.

## Inclusion and exclusion

**Included:**

1. Patient has a waveform record.
2. Adult ICU stay, any stay within an admission.
3. Complete 24h ICU day ending before ICU discharge.
4. At least 384 of 480 three-minute slots valid in that day.

A three-minute slot needs 18 consecutive 10-second windows with all three channels from one native segment and one ECG lead. One bad 10-second window is allowed. The 80% is total coverage, not one unbroken stretch.

**Excluded:**

- Stays with no usable waveform segment or too short for an anchor.
- Waveform records failing clock checks: 215 with a nonzero master counter, 161 with a UTC offset change.
- Delta regression only: anchors where the patient died or left the ICU within 24h. They are counted below.

CXR plays no part in cohort selection.

## Attrition

| Step | Count |
|---|---:|
| ICU stays in MIMIC-IV 3.1 | 94,458 |
| Adult ICU stays of patients with waveform records | 13,604 (7,306 patients) |
| Candidate ICU days (complete, ending before discharge) | 47,730 |
| Days cut, within the index end time | 22,126 (6,527 stays, 5,497 patients) |
| Days added from data past the index end time | 1,005 |
| Days below 384 slots after signal QC | 790 |
| **Final anchors** | **22,341 (6,416 stays, 5,406 patients)** |

The waveform index `endtime` is sometimes early and drops 8.1% of waveform hours. Monitor heart rate in the dropped data matches charted heart rate as well as the earlier data does, so those days were added. Encoder fine-tuning and the encoder comparison used the 21,360 anchors before this addition, and those anchors did not change.

Fate over the next 24h:

| Fate | Anchors | Share | Use |
|---|---:|---:|---|
| Stayed in ICU | 16,670 | 74.6% | SOFA-2 change |
| Left ICU alive | 5,044 | 22.6% | Sensitivity analysis |
| Died | 627 | 2.8% | Composite worsening endpoint |

Death uses the hospital death time, so it can follow ICU discharge.

## Splits

Split is by patient. Existing assignments were kept. Patients without one were assigned once by `SHA256(subject_id)` to 70/15/15.

| Split | Patients | Stays | Anchors | Age, median (IQR) | Female | ICU stay, days, median (IQR) |
|---|---:|---:|---:|---|---:|---|
| Train | 3,799 | 4,531 | 15,731 | 68 (57–77) | 41.9% | 2.95 (1.83–5.77) |
| Validation | 814 | 937 | 3,312 | 67 (56–76) | 41.3% | 2.89 (1.81–5.85) |
| Test | 793 | 948 | 3,298 | 67 (56–77) | 43.6% | 2.95 (1.81–5.57) |
| All | 5,406 | 6,416 | 22,341 | 67 (57–77) | 42.1% | 2.95 (1.82–5.79) |

## Waveform missingness

| Measure | Value |
|---|---:|
| Windows planned | 195,037,750 |
| With at least one valid channel | 194,735,202 (99.8%) |
| With all three channels | 189,839,093 (97.3%) |
| Read failures | 0 |
| Valid three-minute slots per day, median (IQR) | 465 (447–474) of 480 |

Missing slots keep their time position and are masked. Nothing is filled or stitched.

## SOFA-2 scoring

Scored from SOFA-2 Table 2 (Ranzani, JAMA 2025), with MIMIC variables following MIT-LCP `mimic-code`. Day scores come from `labels/score.py` and deltas from `labels/finalize.py`.

1. An ICU day is `[intime+24d, intime+24(d+1))`, counted from ICU admission rather than the calendar date.
2. Each organ scores 0 to 4 from its worst value in that day. A record counts only if stored by the day's end.
3. The total is the sum of the six organs, 0 to 24.
4. For an anchor at `t`, baseline is the day before `t` and future is the day after `t`. Delta is future minus baseline, so a positive delta means worse.

| Organ and measure | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Brain: GCS | 15 | 13–14 | 9–12 | 6–8 | 3–5 |
| Respiratory: PaO2/FiO2 | >300 | ≤300 | ≤225 | ≤150 with support | ≤75 with support |
| Respiratory: SpO2/FiO2 | >300 | ≤300 | ≤250 | ≤200 with support | ≤120 with support |
| Cardiovascular: norepinephrine + epinephrine, mcg/kg/min | None, MAP ≥70 | None, MAP <70 | ≤0.2 | >0.2 to 0.4, or ≤0.2 with another agent | >0.4, or >0.2 with another agent |
| Cardiovascular: neither of those two | | | Another agent, or dopamine ≤20 alone | Dopamine >20 to 40 alone | Dopamine >40 alone |
| Liver: bilirubin, mg/dL | ≤1.2 | >1.2 to 3 | >3 to 6 | >6 to 12 | >12 |
| Kidney: creatinine, mg/dL | ≤1.2 | >1.2 to 2 | >2 to 3.5 | >3.5 | |
| Kidney: urine, mL/kg/h | | <0.5 for 6h | <0.5 for 12h | <0.3 for 24h, or none for 12h | |
| Hemostasis: platelets, ×10⁹/L | >150 | ≤150 | ≤100 | ≤80 | ≤50 |

A ratio that meets a 3- or 4-point threshold without advanced support scores 2. The 4-point cardiovascular row of Table 2 prints ">.02"; we use 0.2 to match the 3-point row.

These rules override the table:

- **Brain:** at least 1 on a day with a delirium drug.
- **Respiratory:** 4 on any ECMO day.
- **Cardiovascular:** 4 on a day with IABP, Impella, LVAD, RVAD, TandemHeart or VA-ECMO.
- **Kidney:** 4 on a day with RRT. Also 4 if creatinine is >1.2 and either potassium is ≥6, or pH is ≤7.2 with bicarbonate ≤12.

How each value is read from MIMIC:

| Organ | Rule |
|---|---|
| Brain | Worst GCS outside sedation. If the patient is intubated or a GCS component is missing, motor score alone: 6 scores 0, 5 scores 1, 4 scores 2, 3 scores 3, 1–2 score 4. GCS during a propofol, midazolam, dexmedetomidine, ketamine or lorazepam infusion is not used. A day with no other GCS takes the last GCS before sedation, or 0 if there is none. Delirium drugs are haloperidol, quetiapine and olanzapine |
| Respiratory | Worst ratio of the day. PaO2 comes from arterial gases only, and FiO2 from the same gas or else charted in the prior 4h. If the day has an arterial gas, PaO2/FiO2 is used. Otherwise SpO2/FiO2 is used with SpO2 <98%, and FiO2 charted in the prior 4h or 0.21 if the device was charted as none. Advanced support is an endotracheal tube, BiPAP, CPAP, high-flow oxygen or a ventilator mode charted in the prior 4h |
| Cardiovascular | Daily minimum MAP, 10–200 mmHg. A drug counts only with at least 60 minutes of continuous infusion in the day. Norepinephrine and epinephrine doses are summed by hour, and the daily peak is scored. Other agents are dobutamine, vasopressin, phenylephrine, milrinone and angiotensin II, and dopamine when given with another drug |
| Liver | Highest bilirubin of the day |
| Kidney | Highest creatinine and worst urine window of the day. Urine is summed hourly over rolling windows, needs a weight of 30–300 kg, and allows at most a 6h gap between records. RRT comes from procedures, active dialysis charting and CRRT drugs, and scores 4 until 72h after the last session. A stay flagged as chronic dialysis scores 4 on every day |
| Hemostasis | Lowest platelets of the day |

## SOFA-2 labels

**Evidence** means the organ was actually measured or recorded in that 24h window, so its score rests on data from that window. A carried value or a pre-sedation GCS is not evidence.

Counts below use the 16,670 stayed anchors. Baseline and Future evidence are the share of these anchors with evidence in the window before and after `t`. An organ label needs evidence in both windows and is masked otherwise. Down, Same, Up and Absolute delta ≥ 2 are shares of that row's labels.

| Organ | Baseline evidence | Future evidence | Labels | Patients | Down | Same | Up | Absolute delta ≥ 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cardiovascular | 99.9% | 99.9% | 16,643 | 3,545 | 23.5% | 63.7% | 12.9% | 6.4% |
| Kidney | 99.8% | 99.5% | 16,562 | 3,508 | 15.3% | 69.8% | 14.9% | 9.3% |
| Hemostasis | 97.4% | 96.1% | 15,705 | 3,453 | 9.7% | 82.6% | 7.6% | 1.4% |
| Brain | 80.6% | 83.1% | 12,587 | 3,287 | 17.5% | 68.2% | 14.3% | 7.0% |
| Respiratory | 82.9% | 81.0% | 12,281 | 2,816 | 21.9% | 61.3% | 16.8% | 11.4% |
| Liver | 41.7% | 36.8% | 4,688 | 1,376 | 7.5% | 86.5% | 6.0% | 0.3% |

Total delta uses the official rule from SOFA-2 Table 2, note b: a value missing on ICU day 1 scores 0, and a value missing later carries the last observation. The other two rules are sensitivity analyses. Coverage is the share of the 16,670 anchors usable under that rule. Real endpoints is the share of the values behind those totals, six organs at two ends, that have evidence.

| Rule | Anchors | Coverage | Patients | Real endpoints | Down | Same | Up | Absolute delta ≥ 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Official SOFA-2 rule (main)** | **16,670** | **100.0%** | **3,549** | **83.2%** | **41.4%** | **28.7%** | **29.9%** | **33.9%** |
| At most one carried window | 4,910 | 29.5% | 1,768 | 94.3% | 42.4% | 28.2% | 29.4% | 37.2% |
| All six real | 2,227 | 13.4% | 853 | 100.0% | 41.6% | 27.0% | 31.3% | 38.2% |

Organ labels use evidence only because a carried future value makes that organ's delta 0 by construction. On the 21,360 anchors, liver would read 94.9% unchanged under the official rule rather than 86.5%, and 66.5% of those zeros would be forced.

## Validity check

First ICU day, normal-value imputation, ICU mortality, on the first ICU stay of each of the 7,306 waveform patients, not limited to cut days:

| Measure | This cohort | Published |
|---|---:|---:|
| SOFA-2 AUROC | 0.816 | 0.79–0.81 (JAMA 2025), 0.829 (Bai 2026) |
| SOFA-1 AUROC | 0.821 | 0.77–0.80 (JAMA 2025), 0.796 (Bai 2026) |
| ICU mortality | 6.8% | 7.3% (Bai 2026) |

Against SOFA-1 on the same evidence, SOFA-2 was lower in 44.0%, the same in 28.7% and higher in 27.3% of stays. Published comparisons differ: 40% lower and 11% higher in JAMA 2025, 60.2% lower and 15.3% higher in Helleberg 2026, 15.0% lower and 62.7% higher in Bai 2026.

Using SpO2/FiO2 for SOFA-1 is our choice, and the SOFA-1 AUROC and this comparison depend on it.

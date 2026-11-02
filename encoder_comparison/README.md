# CSFM downstream comparison

This compares four frozen encoders to choose one for the SOFA-2 pipeline: original CSFM, 10.17 epoch 17, arm A and arm B. It asks whether the original abilities are retained and whether Resp tasks improve. Task types follow the CSFM paper, on our own continuous waveform and EHR data. SOFA is not used. The rule is in `docs/SETTING.md`, section "Encoder choice".

## Run

Run from this folder. GPU 3 is the default because it has free memory beside another user's job.

```bash
bash run.sh test                 # unit tests, a few seconds
bash run.sh labels               # windows and labels, CPU
GPU=3 bash run.sh features       # frozen CLS features; resumes if stopped
bash run.sh probes               # XGBoost, bootstrap and decision, CPU
```

`bash run.sh all` runs the last three in order. Use `tmux` or `nohup` so a closed terminal does not stop it.

Expected time for the full run, from a 120-patient check: `labels` about 5 minutes, `features` about 30 minutes, `probes` about 2 hours at 16 threads. Features take about 6 GB on disk.

| Step | Output |
|---|---|
| `probe_labels/build.py` | `private/windows.pkl`, `private/patients.pkl`, `probe_labels/labels_summary.json` |
| `features/extract.py` | `private/features/` |
| `probes/evaluate.py` | `private/predictions/`, `results/report.md`, `results/metrics.csv`, `results/comparisons.csv`, `results/decision.json` |

Read `results/report.md` first.

If `features` runs out of GPU memory, lower `--batch-size`. A finished step does not need rerunning: `probes` reuses saved predictions, so delete `private/predictions/` after changing a task.

## Method

- **Patients:** train and validation only, one first qualifying ICU day each. The test split is never read.
- **Windows:** one 10-second slot per ICU hour with all three channels, holding a charted vital when the hour has one. Selection never uses a label value.
- **Encoders:** eval mode, no gradients, no masking and no text. Original CSFM receives no Resp input.
- **Inputs:** ECG, PPG, and ECG + PPG for all encoders; all three channels and Resp alone for the three with a Resp channel.
- **Predictor:** XGBoost with 300 trees, depth 6 and learning rate 0.05, as in the 10.17 and 10.28 probes. One seed is used because these settings give identical trees across seeds.
- **Splits:** fit on 90% of train patients, choose thresholds on the other 10%, score validation patients.
- **Uncertainty:** 1,000 paired draws of validation patients. Each draw keeps all rows of a patient.

## Labels

| Source | Labels | Notes |
|---|---|---|
| Monitor numerics | RR, HR, SpO2, invasive SBP and DBP, QTc | Median inside the 10 seconds. `numerics.csv.gz` was aligned to the waveform on 800 ICU days, see `label_audit/` |
| Charted items | Rhythm, ectopy, breathing pattern, respiratory support | Within 30 minutes of the window. Conflicting or unmapped values are excluded |
| Charted RR | RR | Charted inside the 10 seconds |
| Hospital tables | Age, sex, BMI, ICD groups, deaths | BMI is the nearest outpatient value within a year. Ages from 89 are excluded |

## Reading the results

- `gain` is positive when the candidate is better than the reference.
- Both parts compare with original CSFM. In part 2 it uses ECG + PPG because it has no Resp channel. This reference replaced epoch 17 after the first results were seen.
- `gate` marks checks that count towards the decision. Tasks with too few positives are listed as skipped.
- Many checks are made with no multiplicity correction, so one lost check needs a look at its size and CI before it is believed.

## Limits

- **Leads:** continuous waveform keeps one monitor lead, so the precordial 12-lead ability measured in the paper is not tested.
- **False alarms:** MIMIC has no alarm labels, so the paper's ICU false-alarm task has no equivalent here. Death within 24 hours is reported instead, outside the decision.
- **Blood pressure:** the paper reconstructs the ABP waveform. Here CLS predicts SBP and DBP values.
- **ICD labels:** coded after discharge for the whole admission, so they do not describe the window's moment.
- **Resp labels:** the monitor computes RR from the same Resp signal, and arm B's training text holds RR, breathing pattern, O2 device and ventilator fields. ICD respiratory diagnoses are the task that does not favor it.
- **Margins:** these are project choices, not published thresholds.

## Patient data

`private/` and `label_audit/private/` hold patient-level data and must not be shared or committed. `results/` and `label_audit/numerics_audit.json` are aggregates. `probe_labels/labels_summary.json` and `label_audit/ehr_audit.json` are aggregates with very small counts, so they stay out of git.

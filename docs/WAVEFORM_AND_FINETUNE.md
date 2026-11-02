# Waveform cutting and CSFM fine-tuning in detail

How the monitor waveforms became 10-second ICU windows, and how the CSFM encoder was fine-tuned on them with a new Resp channel. Commands are in the main README. The cohort is in `COHORT.md`.

## Process

| Step | Folder | What happens | Result |
|---|---|---|---|
| 1 | `waveform/native_index/` | Read WFDB headers and check clocks | 11,989 records |
| 2 | `waveform/icu_cut/` `plan` | Plan ICU days from headers, without reading samples | 22,126 days, 186.5M windows |
| 3 | `waveform/windows_10s/` | Read, clean and normalize one 10-second window | Used by step 4 |
| 4 | `waveform/icu_cut/` `cut` | Cut every planned day onto the ICU time grid | 5.1 TB, 21,360 days pass QC |
| 5 | `finetune/arms/` `text` | Build charted vital-sign text for arm B | 634,226 snapshots |
| 6 | `finetune/arms/` `check` | Confirm the MAE forward matches the checkpoint | Targets are per-patch normalized |
| 7 | `finetune/arms/` `train` | Fine-tune arm A (signal only) and arm B (signal + text) | Two encoders |
| 8 | `encoder_comparison/` | Compare encoders on downstream tasks | Choice still open |

`finetune/csfm_mae/` holds the model and tokenizer that steps 5 to 7 reuse. 
## Waveform cut

**Input:** MIMIC-IV Waveform Database v0.2 prerelease and MIMIC-IV 3.1. The patient split comes from a fixed file, `waveform/icu_cut/inputs/pairs.csv`. Patients without an assignment get one from `SHA256(subject_id)`, 70/15/15.

**Time grid:** ICU day `d` is `[intime + 24d, intime + 24(d+1))`. Slot `k` is the 10 seconds from `intime + 10k`. So 18 slots make 3 minutes and 8,640 make a day, all aligned to the prediction time.

**Planning:** a day is planned if it is complete, ends before ICU discharge, and its headers could still reach 384 of 480 valid three-minute slots.

| Step | Count |
|---|---:|
| ICU stays in MIMIC-IV 3.1 | 94,458 |
| Adult stays of patients with waveform records | 13,604 |
| Candidate days | 47,730 |
| Planned days | 22,126 (6,527 stays, 5,497 patients) |
| Days passing 384 of 480 after signal QC | 21,360 |

Fine-tuning and the encoder comparison used these 21,360 days. A later cut added 1,005 days that the waveform index end time had dropped, of which 981 passed, giving the final 22,341 anchors in `COHORT.md`.

**Cleaning:** the same code as the 10.14 preprocessing.

| Channel | Native rate | Processing |
|---|---:|---|
| ECG | 250 Hz | `ecg_clean` (NeuroKit, powerline 50). One lead per segment, priority II, I, III, aVR, aVL, aVF, V1 to V6 |
| PPG | 125 Hz | FFT to 250 Hz, `ppg_clean` (Elgendi) |
| Resp | 62.5 Hz | FFT to 250 Hz, `rsp_clean` (Khodadad 2018) |

Each channel is z-scored. Windows are complete 10 seconds, with nothing stitched across gaps.

**Result:**

| Measure | Value |
|---|---:|
| Windows planned | 186,486,496 |
| At least one valid channel | 186,194,282 (99.8%) |
| All three channels | 181,474,055 (97.3%) |
| Rejected by signal QC | 292,214 |
| Read failures | 0 |

**Storage:** one `day_NNN_signal.npy` (`float32 [N,3,2500]`) and one `day_NNN_meta.npz` per ICU day. The meta file is written last, so a stopped run resumes by rerunning the command. The run took about 11.5 hours, at about 1,900 days per hour.

## Fine-tuning

**Start:** `CSFM_base.pth`, encoder and decoder. Three things are new: a Resp encoder channel embedding, a Resp decoder channel embedding and a Resp reconstruction head.

| Item | Setting |
|---|---|
| Masking | Random 75% of signal tokens |
| Signal loss | MSE on masked patches, against per-patch normalized targets |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.05 |
| Learning rate | 1e-4, layer-wise decay 0.75, cosine with 5% warmup |
| Effective batch | 256 |
| Validation | Every 2,500 updates, on the same 77,571 signal-only windows for both arms |
| Selection | Lowest mean of ECG, PPG and Resp validation MSE |
| Stopping | 10 validations without improvement after warmup |

| Arm | Text | Training windows |
|---|---|---|
| A | None | One random valid window per ICU hour, 369,685 |
| B | Charted vitals, 50% of tokens masked, cross entropy added to the signal loss | The window holding a charted snapshot, one per ICU hour, 365,153 |

Arm B's text uses real charted values only, as `<label>: <value>.`, up to 64 tokens: heart rate and rhythm, ectopy, SpO2, noninvasive mean BP, respiratory rate, breathing pattern, O2 device and ventilator settings.

**Why the target changed from 10.17:** the check in step 6 showed the pretrained decoder reconstructs per-patch normalized targets. The 10.17 fine-tune trained against raw targets, which that decoder never used, and its best epoch was 17 of 27. That code is not part of this repository.

**How the arms ran:** both shared GPU 3 for the whole run, with batch 32 and accumulation 8, one worker process each. `queue_gpus.py` was written to move them to free GPUs, but no GPU became free. Arm A was restarted once at update 120 and resumed from its checkpoint.

**Result:** validation MSE at the selected update, lower is better. The same update was best for mean and for Resp in both arms.

| | Original CSFM | Arm A | Arm B |
|---|---:|---:|---:|
| ECG | 0.343 | 0.3002 | 0.2967 |
| PPG | 0.184 | 0.1355 | 0.1331 |
| Resp | No channel | 0.2427 | 0.2266 |
| Mean of three | | 0.2261 | 0.2188 |
| Selected / stopped update | | 22,500 / 47,500 | 40,000 / 65,000 |
| Training time | | About 24 hours | About 31 hours |

Original CSFM was scored on 512 train windows, not the validation set, so that column is only a rough reference.

**Lower MSE did not mean a better encoder.** Arm B had the lower Resp MSE, but arm A did better on downstream Resp tasks. See below.

## Encoder comparison

Frozen CLS features with XGBoost on validation patients, on 106,508 windows from 4,561 patients. Code, method and full tables are in `encoder_comparison/`.

| | Original CSFM | Epoch 17 | Arm A | Arm B |
|---|---:|---:|---:|---:|
| Respiratory rate MAE | 4.29 | 3.67 | 3.14 | 4.06 |
| Invasive support AUROC | 0.720 | 0.765 | 0.805 | 0.780 |
| Retention checks lost, of 31 | Reference | 2 | 1 | 0 |

Original CSFM has no Resp channel, so it uses ECG + PPG here and the others use all three channels.

**The encoder is not chosen yet.** With epoch 17 as the Resp reference, as in `encoder_comparison/results/report.md`, no arm passes. With original CSFM as the reference, arm B passes. That reference was changed after the results were seen.

## Limits

- **ECG leads:** one monitor lead per segment, so 12-lead ability is not trained or tested here.
- **Selection:** only days with at least 80% valid slots enter the cohort, so this is not all ICU patients.
- **Index end:** segments are cut at the `endtime` in `v0.2_tmp_starttime_index.csv`, which drops 8.1% of waveform hours. That data looks valid, and using it would add about 1,005 planned days and 57 patients. See `waveform/coverage_audit/`. This run did not use it.
- **Early stopping:** both arms stopped with the learning rate still high, at about 8e-5 and 6e-5. Resp validation error was noisy and drove the stop, while ECG and PPG were still improving slowly.
- **Arm A Resp:** its Resp validation MSE rose from 0.2427 to 0.2674 after update 22,500.
- **Fine-tuning itself:** the CSFM paper has no Resp channel and no self-supervised fine-tune step, so this stage is our own extension. The MAE forward is not public, so ours is a reconstruction.
- **Tokenizer:** BERT uncased is an assumption. The authors have not confirmed the original mapping.
- **Text:** vital-sign text replaces the ECG reports used in the paper.

## Verification

Checked on 2026-11-02, in a new environment built from `requirements.txt`:

| Check | Result |
|---|---|
| Unit tests | All 104 pass |
| Labels rebuilt from raw MIMIC-IV | All 6 extracts, `daily`, `anchors` and `cohort_report.json` identical |
| `plan` rebuilt from raw headers | `days`, `stays`, `segments` and `records` identical |
| `cut` on 3 ICU days | Signal and meta files byte-equal |
| 200 windows against the 10.14 output | All byte-equal |
| `text` rebuilt from chartevents | All arrays identical, 634,226 snapshots |
| `check` | Identical to the original code on the same data |
| Encoder features, 128 windows, every encoder and input | Maximum difference 0 from the reported run |
| Probes, PVC task | Metrics identical to the reported run |
| Training, 3 updates on real data | Runs and validates |

The full 5.1 TB cut and full training were not rerun. `finetune.py` and `data.py` are byte-identical to the versions that trained the arms.

## Left out

- **`queue_gpus.py`:** a scheduling wrapper that only starts `run.sh train` and moves arms between GPUs.
- **Old pipelines:** the 10.13 and 10.14 SQLite preprocessors and the 10.17 launcher. They are not part of this repository.
- **`10.30/sofa2_icu_pipeline/pilot/`:** one-off feasibility scripts. Their findings are recorded in `SETTING.md`, such as total coverage keeping 82.2% of sampled 12-hour windows against 25.5% for one unbroken stretch.
- **Later steps:** 3-minute SSL and the 24-hour task, which depend on the encoder choice.

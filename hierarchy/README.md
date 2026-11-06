# Three training levels: 3 minutes, 1 hour, 24 hours

This folder builds a 24-hour waveform representation on top of the fine-tuned 10-second CSFM encoder (`finetune/README.md`) and predicts the next day's SOFA-2 organ scores from it. Three levels are trained one after the other, each on the frozen level below: 3 minutes, 1 hour, then 24 hours.

The three levels are one model, the long-horizon CSFM extension of Tang et al. 2026 [1], applied three times with a larger unit each time: 10-second windows make a 3-minute clip, 3-minute clips make an hour, and hours make a day. Each level trains with the paper's objective, a task loss plus a local and a dynamic loss. Levels 1 and 2 have no labels, so a self-supervised masked-prediction loss takes the place of the task loss. Only level 3 has a SOFA-2 loss.

This page covers what the paper does, what we do, which data we use, where we depart from the paper and why, and the options we considered and did not take.

**Status.** The code holds the design of section 3, written and tested on 2026-11-06: masked prediction at levels 1 and 2, inner tokens with two-term positions at levels 2 and 3, and the CORN and composite head on E2 at level 3, with the controls and endpoints of `labels/README.md`, section 7. A 12-day run went through all three levels and the endpoints. No level has been trained on the new split yet. The encoder is arm A, chosen on 2026-11-06 (`docs/SETTING.md`, "Encoder choice"); the index was rebuilt for the new split the same day (section 8). An earlier version, with the local and dynamic losses only, ran on the earlier split until 2026-11-03 (section 7).

Tags: `[source]` taken from a cited work; `[engineering]` our choice with no scientific source; `[extension]` our own method extension, to be stated as such in any paper; `[open]` not yet fixed.

## Contents

- [Three training levels: 3 minutes, 1 hour, 24 hours](#three-training-levels-3-minutes-1-hour-24-hours)
  - [Contents](#contents)
  - [1. What the long-horizon paper does](#1-what-the-long-horizon-paper-does)
  - [2. Data](#2-data)
  - [3. Our method](#3-our-method)
    - [3.1 One model at every level](#31-one-model-at-every-level)
    - [3.2 Level 1 structure](#32-level-1-structure)
    - [3.3 Level 1 losses](#33-level-1-losses)
    - [3.4 Level 1 training](#34-level-1-training)
    - [3.5 Level 2](#35-level-2)
    - [3.6 Level 3](#36-level-3)
    - [3.7 Masking](#37-masking)
    - [3.8 Stopping and selection](#38-stopping-and-selection)
  - [4. Differences from the paper](#4-differences-from-the-paper)
  - [5. Options considered and not used](#5-options-considered-and-not-used)
  - [6. Evaluation](#6-evaluation)
  - [7. The earlier run and what it showed](#7-the-earlier-run-and-what-it-showed)
  - [8. Implementation status](#8-implementation-status)
  - [9. Known limits](#9-known-limits)
  - [10. Files](#10-files)
  - [11. References](#11-references)

## 1. What the long-horizon paper does

All statements below come from Tang et al., "Extending Pretrained 10-Second ECG Foundation Models to Longer Horizons", arXiv:2605.16975v1 (2026) [1]. Tang, Han and Gu are also authors of CSFM [2].

| Item | Paper |
|---|---|
| Problem | 10-second foundation models "have no built-in way to combine information across time" |
| Backbone | A pretrained transformer, kept frozen. CSFM-Tiny, CSFM-Base, MERL and ECG-JEPA are tested |
| Input length | Trained on L = 180 s and evaluated on 180, 120, 60, 30 and 10 s "without retraining" |
| Positions, Eq. 1 | `E[L][t] = E[10s][t mod N] + E_global[floor(t / N)]`, with N patches per 10 s. The pretrained 10-second positions are repeated, not interpolated. `E_global` is a new learnable matrix, one row per 10-second segment |
| Soft prompts | Learnable tokens prepended to the input and "inserted at intermediate layers", following visual prompt tuning [3] |
| Attention | All patches of the whole recording pass through the frozen backbone together, so 10-second segments attend to each other |
| Teacher | The frozen backbone applied to each 10-second segment on its own |
| L_local, Eq. 2 | Rank-based soft contrastive loss [4]: for anchor segment i, if the teacher ranks j above k, the student should too. Similarity is, for example, negative L2. A projector `g_local` sits on the student |
| L_dyn, Eq. 3 | Select high-affinity segments `I_ha` "by ranking across the affinity matrix" of the teacher. Their projected student features `g_dyn(z_i)` are pulled to the stop-gradient mean `z̄` of all student segment features, by cosine distance. The stop-gradient follows SimSiam [5] |
| Readout | `z̄`, the mean of the student's segment features |
| Loss, Eq. 4 | `L = L_task + λ1 L_local + λ2 L_dyn`, trained end to end. `L_task` is "the standard classification loss" |
| Tasks | VTaC true versus false VT alarms (last 3 minutes before the alarm), MC-MED ED disposition and triage acuity, CPSC2021 paroxysmal AF |
| Baselines | Token and logit pooling over 10-second windows; full fine-tuning, linear probing, partial and bias tuning, adapters, VPT, and an LSTM over snapshot features |

What the paper does not state: λ1, λ2, the temperature, the number of prompts, the high-affinity fraction, how a segment's student feature is read out, the optimizer and schedule. It has no appendix and no public code. The text says "using both groups consistently outperforms using either group alone" for L_dyn, but Eq. 3 shows only the high-affinity group. **The losses are never trained without L_task.**

Results that matter for our design:

| Result | Values |
|---|---|
| Patch-level attention against a sequence model on frozen snapshot features (Table 1, VTaC, CSFM-Base, 180 s, AUC %) | Paper's method 91.23; LSTM tuning 86.93; token pooling 77.94 |
| The same on CPSC2021 and MC-MED ED (Table 1, CSFM-Base, 180 s) | 99.56 against 98.74, and 72.89 against 72.03: small gaps |
| Ablation (Table 3, CSFM on VTaC, 10 s / 180 s) | Global positions alone 78.13 / 88.92; plus L_local 83.62 / 89.00; plus L_dyn 80.01 / 88.90; both 84.10 / 89.50 |
| Affinity-guided against random segment dropping (Table 4) | Affinity-guided is higher in 19 of 20 settings, by up to 3.5 points |
| Other signals (Fig. 4, CSFM-Tiny, VTaC) | ECG only, PPG only and ECG + PPG all beat the baselines |

So most of the gain at 180 s comes from the structure. The two losses help most when the input is short. The conclusion names longer horizons "under practical memory and computational constraints" as an open challenge.

## 2. Data

| Input | From | Notes |
|---|---|---|
| Cut 10-second windows | `waveform/icu_cut/output/cut/days`, see `docs/SETTING.md`, section 4 | ECG, PPG and Resp at 250 Hz, 8,640 slots per ICU day |
| 10-second encoder | `finetune/arms/runs/arm_<A or B>/selected.pt`, set in `common.py`, `ENCODER` | Frozen at every level |
| Anchors and SOFA-2 labels | `labels/private/anchors.pkl`, see `docs/COHORT.md` | One row per anchor day: baseline scores, next-day scores, fate |

**Patients.** Patients are split 70/15/15 by `SHA256(subject_id)` (`docs/SETTING.md`, section "Split"). All three levels train on train patients. Validation patients choose checkpoints and give the reported results. No script reads test patients; `build_index.py --test` exists for the single final run.

**Time.** Every level uses only the 24 hours before an anchor, the anchor day in `anchors.pkl`. Levels 1 and 2 use no labels, so using later hours of train patients would also be allowed; we keep one time frame for all levels `[engineering]`.

**Clips and hours.**

| Unit | Slots | Valid when | Code |
|---|---|---|---|
| 3-minute clip | 18 ten-second slots | All three channels present in at least 17 of the 18 slots, one ECG lead and one record segment, by `valid_clips` in `waveform/icu_cut/common.py`. A bad slot is dropped and the others keep their place `[extension]` | `build_index.py` |
| 1 hour | 20 clips | At least 16 valid clips for level 2 training; any valid clip for feature extraction `[engineering]` | `day_data.py`, `hour_rows` |
| 24 hours | 24 hours | Any valid hour | `train_day.py` |

On the earlier split, the index held 8,702,870 valid clips, 7,188,751 of them from train patients; 94.5% had all 18 slots. The index must be rebuilt on the new split (section 8).

## 3. Our method

### 3.1 One model at every level

The paper extends a model of 10-second segments to a recording of many segments. We do the same three times. At each level the unit of the level below becomes the segment: level 1 treats a 10-second window as the segment of a 3-minute clip, level 2 treats a 3-minute clip as the segment of an hour, and level 3 treats an hour as the segment of a day. Each level is trained, frozen, and used to compute the inputs of the next.

**Each segment keeps its inner tokens.** In the paper a 10-second segment enters as its 300 patch tokens, in order, not as one summary. We keep that one level up. An hour enters level 2 as the 18 window features of each of its 20 clips, and a day enters level 3 as the 20 clip features of each of its 24 hours. Each level therefore sees both the order of segments and the order and change inside each segment `[extension]`.

Every level uses Eq. 4 of the paper [1]:

`L = L_task + λ1 L_local + λ2 L_dyn`

| Part of the paper's model | Level 1, 3 minutes | Level 2, 1 hour | Level 3, 24 hours |
|---|---|---|---|
| Segment | 10-second window | 3-minute clip | 1 hour |
| Segments per recording | 18 | 20 | 24 |
| Tokens per segment | 300 patch tokens (3 channels × 100) | 18, level 1's window features of that clip | 20, level 2's clip features of that hour |
| Tokens per recording | 5,400 | 360 | 480 |
| Backbone | CSFM transformer, pretrained and frozen `[source]` | A transformer trained from scratch | A transformer trained from scratch |
| Within-segment positions, Eq. 1 first term | CSFM's 10-second positions, 100 per channel, repeated in each window `[source]` | 18, one per 10-second slot of a clip, repeated in each clip | 20, one per 3-minute slot of an hour, repeated in each hour |
| Global segment positions, Eq. 1 second term | 18, learned `[source]` | 20, learned | 24, learned |
| Soft prompts | 8 per layer `[source]` | None: the backbone itself is trained | None |
| Segment feature `z_i` | Mean of the window's patch tokens | Mean of the clip's 18 output tokens | Mean of the hour's 20 output tokens |
| Teacher | CSFM on each window alone `[source]` | Level 1 on each clip alone: its clip feature | Level 2 on each hour alone: its hour feature |
| L_task | **Masked segment prediction** `[extension]` | **Masked segment prediction** `[extension]` | SOFA-2 loss, with an ordinal head `[extension]` |
| L_local and L_dyn | As the paper, λ1 = λ2 = 1 | As the paper, λ1 = λ2 = 1 | As the paper, λ1 = λ2 = 0.1 |
| Readout | Mean of segment features, `z̄` `[source]` | `z̄` | `z̄`, then the SOFA-2 head |
| Passed up | The 18 window features `z_i` of each clip, and `z̄` | The 20 clip features `z_i` of each hour, and `z̄` | |

**The one difference between levels is the backbone.** The paper's backbone is a transformer that was pretrained on single segments and is kept frozen; soft prompts then adapt it. For 10-second windows that transformer exists: CSFM. For sequences of window or clip features, no pretrained transformer exists, so levels 2 and 3 train their transformer from scratch `[extension]`. A trained backbone needs no prompts. Everything else is the paper's model.

**Each level looks one level down, not to the patches.** Level 2 sees windows, level 3 sees clips. Going further down is not possible: the frozen 3-minute model as the backbone of an hour would attend over 20 × 5,400 = 108,000 patch tokens, and a day over about 2.6 million. The paper names longer horizons under memory limits as an open problem [1].

**Why a masked-prediction loss stands in for L_task.** In the paper, L_task is the only term that makes `z̄` carry information. Without it, L_dyn is met by a projector that maps every segment to the mean, and L_local is met by a student that copies its teacher. Our earlier level 1 had exactly this: L_dyn fell from 1.0 to 0.0065 in 5,000 updates (section 7). Masked prediction needs no labels and can only be solved from context. Its target is a frozen feature, so it cannot collapse: the target does not move with the trained weights.

Levels 1 and 2 follow the masked autoencoder, as CSFM itself does [2, 9]: 75% of segments are masked at random and removed, the encoder sees the visible segments only, a light decoder adds mask tokens, and the loss is the mean squared error on masked segments only. The target is the frozen feature of the level below rather than the raw input, as in MaskFeat, which gains when its targets are frozen features [8]. At level 1 the raw input of a masked segment is a whole 10-second waveform, whose beat timing cannot be inferred from the neighbouring windows, and I-JEPA found raw-input targets much worse than feature targets for linear probing [11]. At level 2 the input already is the level-below feature, so the target is the masked input itself.


### 3.2 Level 1 structure

As in `long_model.py` and `clip_model.py`.

| Item | Setting | Tag |
|---|---|---|
| Backbone | The chosen 10-second encoder, all weights frozen and in eval mode | `[source]` |
| Input | 18 windows × 3 channels × 100 patches = 5,400 patch tokens plus CLS. Channel IDs: the window's ECG lead, PPG 12, Resp 13 | Our CSFM input, `finetune/README.md` |
| Positions | Pretrained 10-second positions repeated in each window, plus a learned global position per window, 18 rows, initialized N(0, 0.02²) | Eq. 1 `[source]`; initialization `[engineering]` |
| Dropped slot | Its patches are left out; the other windows keep their global positions | `[extension]` |
| Soft prompts | 8 per layer at all 12 layers, initialized N(0, 0.02²). Each layer's prompts replace the previous layer's, so the sequence does not grow | Deep prompts `[source]` [3]; number and replacement `[engineering]` |
| Window feature `z_i` | Mean of the window's patch tokens after the last norm. The joint pass has one CLS for the whole clip, so no per-window CLS exists | `[engineering]`; the paper does not say |
| Clip feature `z̄` | Mean of the window features | `[source]` |
| Teacher `t_i` | The frozen encoder on window i alone, mean of its patch tokens, matched to `z_i` | `[engineering]`; using its CLS, the readout chosen in `finetune/`, is `[open]` |
| Precision | fp16 autocast, gradient checkpointing per layer | `[engineering]` |

The trainable parts are the global positions, the prompts, the two projectors and the masked-prediction decoder.

### 3.3 Level 1 losses

**L_task, masked segment prediction** `[extension]`

1. Of the windows present in a clip, 75% are masked, drawn at random, so `max(1, round(present × 0.25))` stay visible (section 3.7).
2. The masked windows' patches are removed from the student input. The others keep their global positions. CSFM was pretrained with its encoder seeing only 25% of patches [2], so a partial input is familiar to the frozen backbone. A mask token fed into the frozen backbone would not be.
3. A small decoder, 2 transformer layers of width 256, gets the visible window features `z_i` projected to 256, plus one learned mask vector at each masked window. Each position adds its own learned decoder position.
4. A linear layer predicts the teacher feature `t_i` of each masked window, centered within the clip and standardized: the target is `t_i` minus the mean `t` of the clip's windows, divided per dimension by train-patient statistics of these centered targets. Without centering, the mean of the 4 visible windows' teachers already reached a standardized error of 0.10 in a short run, because the windows of one clip are nearly alike; after centering it was 0.44, the level of a model that learns nothing. Centering a target by its own statistics follows MAE's per-patch normalized pixel targets [9] `[source]`; centering by clip is `[extension]`.
5. The loss is the mean squared error over masked windows only. HuBERT found the loss on masked frames only to be the most robust choice [7].
6. The decoder is dropped after training. Feature extraction masks nothing.

The decoder width, depth and standardization are `[engineering]`. The structure follows MAE and CSFM: encoder on visible tokens, light decoder with mask tokens [2, 9].

**L_local**, Eq. 2 `[source]`, in `rank_local_loss`:

| Item | Setting | Tag |
|---|---|---|
| Similarity | Negative L2, for both teacher and student | Example given in the paper |
| Pool | All visible windows of all clips in the batch on all GPUs, gathered before the loss | `[engineering]` |
| Ties | Teacher ties never enter each other's denominators; self is excluded | `[engineering]` |
| Projector | `g_local`: 768 → 256 → 256 | `[source]` for a projector; size `[engineering]` |
| Temperature | 0.1 | `[engineering]` |
| Reduction | Sum over anchors and positives, divided by K² | `[engineering]` |

**L_dyn**, Eq. 3 `[source]`, in `dynamics_loss`:

| Item | Setting | Tag |
|---|---|---|
| High-affinity windows | The 50% of visible windows in a clip with the highest mean teacher affinity (negative L2 to the others) | `[engineering]`; the paper gives no fraction |
| Target | Stop-gradient `z̄` of the same pass, over visible windows | `[source]` |
| Projector | `g_dyn`: 768 → 256 → 768, on the student only | `[source]` for a one-sided projector; size `[engineering]` |
| Distance | 1 − cosine similarity | `[source]` |
| Second group | Not used. The paper's text mentions both groups but gives an equation for one | Displayed equation only |

L_local and L_dyn use visible windows only, since a masked window has no student feature.

**Weights.** 1 for L_task, λ1 = 1, λ2 = 1 `[engineering]`. The paper gives no values.

### 3.4 Level 1 training

As in `train_three_min.py`.

| Item | Setting | Tag |
|---|---|---|
| Clips per epoch | One clean clip per ICU hour of train patients, redrawn each epoch, at most 40,000 | `[engineering]` |
| Validation | A fixed set of 4,000 clean clips from validation patients, with fixed masks | `[engineering]` |
| Batch | 8 clips per GPU on 2 GPUs, accumulation 2, so 32 clips per update. The local loss pools the 16 clips of one step across GPUs | `[engineering]` |
| Optimizer | AdamW, learning rate 3e-4, weight decay 0.01, gradient clip 1 | `[engineering]` |
| Schedule | 30 epochs at most, 2 warmup epochs, then cosine | `[engineering]` |
| Stopping | Section 3.8 | |
| Teacher | Computed online on each batch, not cached | `[engineering]` |
| Speed | About 22 clips per second on one GPU with 75% of windows removed, in a short run on 2026-11-06; the earlier run without masking managed 6 to 8 on two GPUs. An epoch of 40,000 clips then takes about 30 minutes on one GPU | |

Afterwards `extract_three_min.py` encodes every valid clip of every anchor day into `private/three_min/`, without masking. Per day, `<stay>_<day>_windows.npy` holds the 18 window features `z_i` of each of the 480 clips in fp16, about 13 MB, and `<stay>_<day>.npz` holds the clip features `z̄`, the clip and window masks, and the input of control M (section 6): the mean CLS of the frozen 10-second encoder over the first present window of each valid clip `[engineering]`. For about 22,000 days the window files take about 290 GB. Extraction took about 24 seconds per day on one GPU in the short run, so about 150 GPU hours, or about 40 hours on four GPUs.

### 3.5 Level 2

As in `temporal_model.py` and `train_one_hour.py`.

| Item | Setting | Tag |
|---|---|---|
| Input | 20 clips × 18 window features = 360 tokens per hour, with a mask for missing windows and missing clips | `[extension]` |
| Normalization | Per-dimension mean and standard deviation from 200,000 train window features, fixed before training | `[engineering]` |
| Backbone | Linear 768 → 256, 2 pre-norm transformer layers with 8 heads, GELU, dropout 0.1; then LayerNorm, linear 256 → 768, LayerNorm. Trained from scratch | The 10.24 temporal model `[engineering]` |
| Positions | Token t gets `E_within[t mod 18] + E_global[floor(t / 18)]`: 18 learned within-clip positions repeated in each clip, plus 20 learned clip positions. Both initialized N(0, 0.02²) | Eq. 1 `[source]`; learned rather than reused, since this backbone is new `[engineering]` |
| Missing windows and clips | Key padding mask; their outputs are set to zero | `[engineering]` |
| Clip feature `z_i` | Mean of the clip's present output tokens, after the 256 → 768 projection | As level 1's window feature `[engineering]` |
| Hour feature `z̄` | Mean over present clips | `[source]` |
| Teacher `t_i` | The clip feature given as input: frozen level 1 on that clip alone, as CSFM on one window is level 1's teacher | `[source]` for the role; `[extension]` one level up |

**L_task, masked segment reconstruction** `[extension]`, as at level 1:

1. Of the clips present in an hour, 75% are masked at random (section 3.7). All 18 tokens of a masked clip are removed from the encoder input; the others keep both position terms.
2. A decoder of 2 transformer layers, width 256, gets the visible clip features `z_i` projected to 256, plus one learned mask token at each masked clip, each with a learned decoder position.
3. A linear layer predicts the teacher of each masked clip, level 1's clip feature, centered within the hour (minus the mean over the hour's present clips) and standardized per dimension with train-patient statistics, as at level 1. Without centering, the mean of the visible clips reached a standardized error of 0.54 in a short run; after centering, 1.38.
4. The loss is the mean squared error over masked clips only. The decoder is dropped after training; extraction masks nothing.

The decoder size and standardization are `[engineering]`; the structure is MAE's [9].

**L_local and L_dyn** as at level 1, over visible present clips: projectors 768 → 256 → 256 and 768 → 256 → 768, temperature 0.1, high-affinity fraction 0.5. λ1 = λ2 = 1.

| Item | Setting | Tag |
|---|---|---|
| Hours used | At least 16 of 20 valid clips | `[engineering]` |
| Hours per epoch | At most 40,000 train hours, redrawn each epoch, as at level 1 | `[engineering]` |
| Validation | A fixed set of 4,000 validation hours with fixed masks | `[engineering]` |
| Batch, optimizer | 32 hours; AdamW, learning rate 3e-4, weight decay 0.01, gradient clip 1 | `[engineering]` |
| Schedule | Up to 100 epochs, 7 warmup epochs, then cosine; patience 10 | `[engineering]` |
| Extraction | Every hour with at least one valid clip, into `private/one_hour/`: `clip_features.npy`, the 20 clip features `z_i` per hour; `hour_features.npy`, the hour feature `z̄`; `index.npz`, the masks and day keys. About 16 GB in fp16 | `[engineering]` |

### 3.6 Level 3

Code: `train_day.py`, `sofa_head.py` and `evaluate_day.py`. Labels, endpoints and the E2 inputs are defined in `labels/README.md`, sections 3 to 7; the E2 table comes from `labels/e2/e2.py`.

| Item | Setting | Tag |
|---|---|---|
| Input | 24 hours × 20 clip features from level 2 = 480 tokens per day, with a mask for missing clips and hours | `[extension]` |
| Backbone | As level 2, trained from scratch | `[engineering]` |
| Positions | Token t gets `E_within[t mod 20] + E_global[floor(t / 20)]`: 20 learned within-hour positions repeated in each hour, plus 24 learned hour positions | Eq. 1 `[source]`; learned `[engineering]` |
| Hour feature `z_i` and day feature `z̄` | Mean of the hour's present output tokens; mean over present hours. `z̄` goes to the SOFA-2 head | `[source]` |
| L_task | `L_SOFA = mean over organs of the CORN loss + BCE of the composite output` (below) | CORN [12]; one binary output per endpoint [13] `[source]`; equal weights `[engineering]` |
| Head | A baseline path on the E2 inputs, plus a waveform correction on the same logits from `z̄` and the same inputs, its last layer initialized to zero (below) | `[extension]` |
| L_local and L_dyn | On hour features `z_i`, teacher = level 2's hour feature `z̄` of that hour, weights 0.1 and 0.1 | `[source]` for the terms; weights `[engineering]` |
| Death and discharge | Main analysis: death scored 24 (each organ 4), discharge 0, after de Grooth et al. 2017 [10]. Sensitivity: `locf`, `partial_discharge`, `no_death`, `stayed_only` | `[source]` |
| Sampling | Each patient carries the same total weight | `[engineering]` |
| Optimizer and schedule | As level 2 | `[engineering]` |
| Methods | `e0` and `e2`, the baseline path alone on E0 or E2; `m`, E2 with the mean frozen 10-second CLS as the waveform input; `hierarchy`, E2 with level 3; `wave_only`, level 3 alone. Three seeds each, 42, 43 and 44 | `[engineering]` |
| Endpoints | `evaluate_day.py` averages each method's seeds into one probability and scores the endpoints of `labels/README.md`, section 7, with 1,000 paired patient draws; per-seed primary endpoints are also reported. Output: `runs/day/<handling>/endpoints.json` | `[engineering]` |

Level 3 has no masked-prediction loss, because the paper's Eq. 4 has one L_task and here it is SOFA-2:

`L3 = L_SOFA + 0.1 L_local + 0.1 L_dyn`

Levels 1 and 2 are frozen and their features precomputed, so the SOFA-2 loss trains level 3 and the head only.

**Head.** Two paths give the same logits, which are added:

| Path | Input | Output |
|---|---|---|
| Baseline | The E2 inputs (`labels/README.md`, section 7.4): baseline and previous-day organ scores with missing flags, ICU day, and hourly median heart rate, respiratory rate, SpO2 and MAP of the 24 hours before `t` with missing flags | 6 organs × 4 CORN logits, and 1 composite logit |
| Waveform correction | `z̄` and the same inputs | The same 25 logits, added. Its last layer starts at zero, so training starts from the baseline model |

The gap between H and its baseline-only control is then what the waveform adds beyond charted EHR. With an E0 baseline, a gain could come from reading charted heart rate or SpO2 back out of the waveform.

**Loss.**

1. **Organs, CORN** [12]. For organ o and k = 0 to 3, logit `f_{o,k}` gives P(future > k | future > k − 1). Output k is trained with binary cross entropy only on anchors whose labeled future score is above k − 1 (all anchors for k = 0). Each organ's loss is the mean over its eligible anchor-threshold pairs, among anchors where that organ is labeled (`labels/README.md`, section 6). Cumulative probabilities are running products of the sigmoids, so they cannot cross.
2. **Composite.** Binary cross entropy of the composite output against total change ≥ 2 or death in `[t, t + 24h)`, under the anchor's handling, on anchors with an official total.
3. `L_SOFA` is the mean of the six organ losses plus the composite loss. Organs weigh equally whatever their label counts. Every patient carries the same total weight, as now.

The earlier Huber term on the summed expected change is dropped. It had no source and trained the expected change, while the endpoint needs the probability of a change of 2 or more.

**Stopping.** On the validation `L_SOFA`, not on the endpoints. Death and discharge handlings are separate runs (`labels/README.md`, section 5).

### 3.7 Masking

Random masking of 75% of segments, as in MAE and CSFM [2, 9] `[source]`:

| Level | Segments | Masked | Visible |
|---|---|---|---|
| 1 | 18 windows of a clip | 75% of present windows, at random | `max(1, round(present × 0.25))`, 4 of 18 |
| 2 | 20 clips of an hour | 75% of present clips, at random | 5 of 20 |

Adjacent segments are alike, so a masked segment next to a visible one could be filled by copying it. MAE's high ratio limits this: it argues that masking most of the input removes the redundancy that makes such shortcuts possible [9]. With 4 of 18 windows visible, most masked windows have no visible neighbour. A high ratio was not enough here: segments of one clip or hour are so alike that the mean of the visible ones nearly gives every masked one. Targets are therefore centered within their clip or hour (sections 3.3 and 3.5), so that only the change between segments is left to predict. Other ratios are compared with probes (section 3.8), not with the masked loss, which rises with the ratio by itself.

L_local and L_dyn use the visible segments only, so they see 4 or 5 per clip or hour. L_local pools all visible segments of a batch, so its pool stays large `[engineering]`.

### 3.8 Stopping and selection

| Level | Within one run | Between settings, such as the masking ratio |
|---|---|---|
| 1 and 2 | Validation masked-prediction loss, the term in the place of L_task `[engineering]` | Linear probes from the level's feature to charted heart rate, respiratory rate, SpO2 and MAP of the same period, on validation patients `[open]` |
| 3 | Validation `L_SOFA` | Validation endpoints of `labels/README.md`, section 7 |

Each level stops on its L_task. L_local and L_dyn can fall while the feature learns nothing (section 7), so they do not choose checkpoints. Masked losses of different settings cannot be compared, since masking more raises the loss by itself. SOFA-2 is not used before level 3, so that it stays a test of the waveform representation, as in the encoder comparison.

## 4. Differences from the paper

| Item | Paper | Here | Reason |
|---|---|---|---|
| L_task at levels 1 and 2 | Classification loss on labels | Masked segment prediction | No labels at these levels. Without a task loss, L_dyn becomes trivial (section 7) `[extension]` |
| Horizon | 180 s | 3 minutes, then 1 hour and 24 hours | Our task needs a day `[extension]` |
| Segment at levels 2 and 3 | 10-second ECG as its patch tokens | A 3-minute clip as its 18 window features, then an hour as its 20 clip features | The same model one level up, keeping order and change inside each segment `[extension]` |
| Within-segment positions at levels 2 and 3 | The frozen backbone's own, repeated | Learned, repeated | A new backbone has no positions of its own `[engineering]` |
| Backbone at levels 2 and 3 | Pretrained and frozen, with soft prompts | Trained from scratch, no prompts | No pretrained model of 3-minute or hourly features exists `[extension]` |
| Teacher at levels 2 and 3 | Frozen backbone on each segment | The frozen level below on each segment | Same role one level up `[extension]` |
| Masked windows at level 1 | Not used | Removed from the input | Masked prediction needs them hidden |
| Dropped bad slot | Not discussed | Left out, other windows keep positions | Real ICU data has gaps `[extension]` |
| Channels | ECG, or ECG and PPG (Fig. 4) | ECG, PPG and Resp | Resp is added in `finetune/` |
| Backbone | Pretrained CSFM, MERL or ECG-JEPA | CSFM fine-tuned with Resp | `finetune/README.md` |
| Segment readout | Not stated | Mean of patch tokens per window | No per-window CLS in a joint pass `[engineering]` |
| L_dyn groups | Text mentions both; Eq. 3 shows one | Eq. 3 only | Displayed equation |
| Hyperparameters | Not stated | Sections 3.2 to 3.5 | `[engineering]` |
| Input lengths | Trained at 180 s, evaluated at shorter lengths | Trained and used at full length; gaps from missing data only | Our inputs have fixed length |

## 5. Options considered and not used

| Option | What it would do | Why not now |
|---|---|---|
| The paper's losses without L_task | The earlier code | L_dyn fell to 0.0024 and L_local fell 18% in 17 epochs; nothing forces `z̄` to hold information (section 7). Kept as an ablation |
| Frozen lower level as backbone at every level | 1 hour over the frozen 3-minute model attending over all its patches, 24 hours over the frozen 1-hour model | 108,000 patch tokens per hour and 2.6 million per day |
| One token per segment | Level 2 sees 20 clip means, level 3 sees 24 hour means; the earlier code | Loses the order and change inside each clip and hour. Kept as an ablation (section 6) |
| Reusing the lower level's global positions as within-segment positions | Closer to Eq. 1, which reuses the backbone's own 10-second positions | Level 1's global positions live in CSFM's input space, while level 2 is a new 256-d model with no use for them |
| Frozen CSFM layers as the backbone of levels 2 and 3 | The same frozen weights at all three levels: a new input projection, global positions and prompts feed the 3-minute or hourly features to CSFM's transformer layers | Fewer trained parameters, which may help level 3 with 15,615 training anchors. But CSFM's layers have only seen 0.1-second patch embeddings, and there is no evidence they work on other inputs. Candidate comparison |
| One token-level 3-minute model reused at every level | Level 1 on cached 10-second features; levels 2 and 3 as frozen copies with new prompts and positions | Level 1 would lose patch-level attention, the paper's main gain (91.23 against 86.93 on VTaC, Table 1). Candidate comparison at level 1 |
| Caching 10-second features for level 1 | Faster training | Only useful if level 1 works on features. The teacher could still be cached; not needed yet |
| Mask tokens in the frozen backbone | Keep the sequence length | The frozen encoder never saw mask tokens in its input |
| Uncentered targets | Predict each masked segment's teacher as it is | The mean of the visible segments already gives a low error (0.10 at level 1, 0.54 at level 2), so the task can be met without learning change between segments. Decided on 2026-11-06 |
| Centering by the visible segments only | Target minus the mean of the visible segments' teachers | Same effect; the clip mean is the segment's own recording statistic, as MAE normalizes a patch by its own statistics |
| Loss on all positions | Also reconstruct visible windows | HuBERT found masked-only more robust [7]; visible windows are already constrained by L_local |
| EMA teacher, as in I-JEPA | Learned targets that improve during training | I-JEPA needs an exponential moving average target encoder to avoid collapse [11]. Our frozen targets cannot collapse and need no EMA |
| Contrastive future prediction (CPC) | Predict the next segments against negatives | Segments of the same patient are easy to tell from other patients' segments, so the model may learn patient identity |
| Masked prediction at level 3 too | Self-supervision beside SOFA-2 | Eq. 4 has a single L_task, here SOFA-2 |
| Independent sigmoids on P(score > k) with a Huber term on the summed change | The earlier head | The sigmoids can cross, and the Huber term has no source; CORN and a composite output replace them (section 3.6) |
| Baseline path on E0 only | Simpler | Would credit the waveform with charted values (section 3.6) |
| SOFA-2 probes to choose level 1 and 2 settings | Choose by the final task | SOFA-2 would then shape the self-supervised levels |
| Span masking, as in wav2vec 2.0 and HuBERT [6, 7] | About half of the segments masked in spans of at least 3, set by a lag check | MAE's 75% random masking is CSFM's own scheme and limits neighbour copying without a span length to tune (section 3.7) |
| JEPA with block masking, as in I-JEPA [11] | Predict masked blocks of a target encoder's features | With a frozen target this is the same structure as ours; the masking and loss follow MAE so that all levels match CSFM |
| Raw-waveform targets at level 1 | Reconstruct the masked windows' 0.1-second patches, as CSFM does inside one window | A whole masked 10-second window cannot be rebuilt beat by beat from its neighbours; raw-input targets were much worse than feature targets in I-JEPA [11]. Candidate comparison if needed |

## 6. Evaluation

**Checks of the self-supervised levels**, on validation patients:

| Check | Pass when |
|---|---|
| Masked loss against copying the nearest visible window, and against the mean of visible windows | Clearly lower than both |
| L_dyn without its projector: cosine distance of high-affinity window features to `z̄` | Not near zero; a near-zero L_dyn with a large distance here means the projector solved it |
| Staged check, `staged_check.py`: XGBoost on mean-pooled clip features per day for current organ score, past change and future change | Better than a constant, before levels 2 and 3 are run |

**Level 3**, on validation patients, with a patient bootstrap of 1,000 draws. Provisional metrics, from `docs/SETTING.md`, section 9:

- organ and total MAE and RMSE, and MAE on anchors whose score changed;
- worsening by 2 or more, and the composite of worsening by 2 or death: AUROC, AUPRC, calibration;
- direction (down, same, up): balanced accuracy.

Primary endpoints are `[open]`. Liver is reported with its coverage and not used for selection.

**Controls and ablations**, on identical anchors and masks:

| Model | Input | Question | State |
|---|---|---|---|
| Persistence | Next score = today's | Floor | In the code (`evaluate_day.py`, MAE and RPS) |
| E0 | Six baseline scores | Value of current status alone | In the code, `e0` |
| Waveform only | Waveform, no baseline | Waveform alone | In the code, `wave_only` |
| E2 | Baseline + previous day + ICU day + hourly HR, RR, SpO2 and MAP | Does the waveform only recompute monitor numbers | In the code, `e2` |
| M | E2 + mean of frozen 10-second CLS, one window per valid clip | Is the hierarchy better than averaging | In the code, `m` |
| Hierarchy | E2 inputs + levels 1 to 3 | Our method | In the code, `hierarchy` |
| Hierarchy without self-supervision | Same structure, levels 1 and 2 trained with SOFA-2 end to end | Is the self-supervised step needed | To do |
| Level 1 without L_task, without L_local, without L_dyn | Paper's ablation (Table 3) on our setting | What each term adds | To do |
| One token per segment at levels 2 and 3 | Clip means and hour means only | Does order and change inside a segment help | To do |

## 7. The earlier run and what it showed

Level 1 ran on the earlier split with the earlier arm B encoder, with L_local and L_dyn only, on 2 GPUs. It was stopped on 2026-11-03, before levels 2 and 3, because the encoders and the split are being redone.

| Update | L_local | L_dyn |
|---:|---:|---:|
| 20 | 4.56 | 1.00 |
| 960 | 4.01 | 0.050 |
| 4,960 | 3.85 | 0.0065 |
| 20,960 | 3.65 | 0.0024 |

Validation after the first epoch: L_local 4.05, L_dyn 0.025. After epoch 17: 3.72 and 0.0024.

- **L_dyn:** reached nearly zero within 5,000 updates and then stayed. The one-sided projector can map all windows to the mean, so the term no longer constrains the features.
- **L_local:** fell about 18% over 17 epochs. Only prompts and global positions learn, and the loss asks the student to keep the teacher's ranking, which the unchanged teacher features already almost do.
- **Speed:** 6 to 8 clips per second, 1,250 updates per epoch of 40,000 clips.

This is what led to the masked-prediction loss in the place of L_task.

## 8. Implementation status

| Part | State |
|---|---|
| Level 1 structure, L_local, L_dyn | In the code, `long_model.py` and `clip_model.py`, with tests |
| Levels 2 and 3, death and discharge handling | In the code, with tests |
| Level 3 head: CORN organ outputs, composite output, E2 baseline path; `L_SOFA` without the Huber term | In the code, `sofa_head.py` and `train_day.py`, with tests |
| Endpoints with eligibility, calibration, RPS, MAE and the paired bootstrap | In the code, `evaluate_day.py`, with tests |
| MAE loss at levels 1 and 2, 75% random masking, decoders | In the code, `clip_model.py`, `temporal_model.py`, with tests |
| Inner tokens: level 1 stores 18 window features per clip, level 2 reads 360 tokens and stores 20 clip features per hour, level 3 reads 480 tokens; two-term positions at levels 2 and 3 | In the code, with tests |
| Stopping on the masked loss at levels 1 and 2 | In the code |
| Copy and mean baselines; L_dyn without projector | In the code, logged at every validation of levels 1 and 2 |
| End-to-end run | 12 days of the earlier index through all three levels and the endpoints on 2026-11-06, one short epoch each; a code check only |
| Index on the new split | Done 2026-11-06: 15,615 train and 3,284 validation days; 7,136,020 and 1,499,451 clips, 472,534 with one bad slot |
| `ENCODER` in `common.py` | Arm A, `finetune/arms/runs/arm_A/selected.pt`, set on 2026-11-06 |
| E2 table | `labels/e2/e2.py`, run by `labels/run.sh vitals` and `e2` |
| E2 and M controls | In the code, `train_day.py` |
| No-self-supervision control | To do. Training levels 1 and 2 end to end with SOFA-2 would back-propagate through 24 hours of patches; a cheaper version is `[open]` |
| `staged_check.py` on the endpoints | To do; it reports MAE of mean-pooled level 1 features |

## 9. Known limits

- **No reference implementation:** the paper gives no code or numbers, so the structure and every setting are our reconstruction.
- **Untested use of the paper's losses:** the paper never trains them without a task loss, and never beyond 180 s.
- **Backbones of levels 2 and 3 are not pretrained:** the paper's model relies on a pretrained frozen backbone, and here only level 1 has one.
- **Clips need all three channels:** windows without Resp, or with a lead change, are not used, so hours with partial monitoring lose coverage.
- **Teacher readout:** the teacher is a patch mean, while the encoder was chosen on its CLS. Whether that matters is `[open]`.
- **Cost:** level 1 training runs the encoder on 1,200 tokens per clip for the student plus 18 windows alone for the teacher; extraction runs it on 5,400 tokens per clip and takes about 150 GPU hours. The stored window features take about 290 GB.
- **One seed:** each level is trained once first.

## 10. Files

| File | Role |
|---|---|
| `run.sh` | `index`, `three_min`, `features`, `check`, `one_hour`, `day`, `endpoints`, `test` |
| `common.py` | Paths, clip reading, the chosen encoder |
| `data/build_index.py` | Valid 3-minute clips on every anchor day, `private/clips.npz` and `days.pkl` |
| `data/day_data.py` | Day arrays, SOFA-2 targets and the composite event, death and discharge handling, patient weights, E0 and E2 head inputs |
| `models/long_model.py` | Frozen CSFM with global positions and prompts; L_local and L_dyn |
| `models/clip_model.py` | Level 1 with a dropped slot; its losses |
| `models/temporal_model.py` | Temporal transformer over inner tokens, its L_local and L_dyn, the masked decoder and its reference checks, for all levels |
| `models/sofa_head.py` | CORN and composite head, `L_SOFA` |
| `models/distributed_cpu.py` | Multi-GPU helpers |
| `level1_three_min/train_three_min.py` | Level 1 training |
| `level1_three_min/extract_three_min.py` | Level 1 window and clip features for every clip, and the control M input, `private/three_min/` |
| `level1_three_min/staged_check.py` | Signal check on mean-pooled level 1 features |
| `level2_one_hour/train_one_hour.py` | Level 2 training and features, `private/one_hour/` |
| `level3_day/train_day.py` | Level 3 and its controls, three seeds each |
| `level3_day/evaluate_day.py` | Endpoints and the paired patient bootstrap, `endpoints.json` |
| `tests/test_hierarchy.py` | Tests |

## 11. References

1. Tang W, Han J, Cui K, Carletti M, Gustafsson FK, Gowda SN, Palo P, Thakur A, Clifton L, Morel J-M, Chan RH, Clifton DA, Gu X. Extending pretrained 10-second ECG foundation models to longer horizons. arXiv:2605.16975v1 (2026). https://arxiv.org/abs/2605.16975
2. Gu X, Tang W, Han J, et al. Cardiac health assessment across scenarios and devices using a multimodal foundation model pretrained on data from 1.7 million individuals. *Nature Machine Intelligence* 8, 220–233 (2026). https://doi.org/10.1038/s42256-026-01180-5
3. Jia M, Tang L, Chen B-C, Cardie C, Belongie S, Hariharan B, Lim S-N. Visual prompt tuning. *ECCV*, 709–727 (2022).
4. Zha K, Cao P, Son J, Yang Y, Katabi D. Rank-N-Contrast: learning continuous representations for regression. *NeurIPS* 36, 17882–17903 (2023).
5. Chen X, He K. Exploring simple Siamese representation learning. *CVPR*, 15750–15758 (2021).
6. Baevski A, Zhou H, Mohamed A, Auli M. wav2vec 2.0: A framework for self-supervised learning of speech representations. *NeurIPS* 33 (2020). arXiv:2006.11477
7. Hsu W-N, Bolte B, Tsai Y-HH, Lakhotia K, Salakhutdinov R, Mohamed A. HuBERT: Self-supervised speech representation learning by masked prediction of hidden units. *IEEE/ACM Transactions on Audio, Speech, and Language Processing* (2021). arXiv:2106.07447
8. Wei C, Fan H, Xie S, Wu C-Y, Yuille A, Feichtenhofer C. Masked feature prediction for self-supervised visual pre-training. *CVPR* (2022). arXiv:2112.09133
9. He K, Chen X, Xie S, Li Y, Dollár P, Girshick R. Masked autoencoders are scalable vision learners. *CVPR*, 16000–16009 (2022).
10. de Grooth H-J, Geenen IL, Girbes AR, Vincent J-L, Parienti J-J, Oudemans-van Straaten HM. SOFA and mortality endpoints in randomized controlled trials: a systematic review and meta-regression analysis. *Critical Care* 21, 38 (2017). https://doi.org/10.1186/s13054-017-1609-1
11. Assran M, Duval Q, Misra I, Bojanowski P, Vincent P, Rabbat M, LeCun Y, Ballas N. Self-supervised learning from images with a joint-embedding predictive architecture. *CVPR* (2023). arXiv:2301.08243
12. Shi X, Cao W, Raschka S. Deep neural networks for rank-consistent ordinal regression based on conditional probabilities. *Pattern Analysis and Applications* 26, 941–955 (2023). https://doi.org/10.1007/s10044-023-01181-9. arXiv:2111.08851
13. Yèche H, Kuznetsova R, et al. HiRID-ICU-Benchmark: a comprehensive machine learning benchmark on high-resolution ICU data. *NeurIPS Datasets and Benchmarks* (2021). arXiv:2111.08536. https://arxiv.org/abs/2111.08536

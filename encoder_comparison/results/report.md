# Encoder comparison

Validation patients only. See README.md for the method.

The Resp reference was changed from epoch 17 to original CSFM after the first results were seen, to match the stated goal.

## Decision

- arm_a: FAIL. Lost 1, inconclusive 13 of 31 retention checks. All three channels against original CSFM on ECG + PPG: RR better, invasive support better, ICD respiratory same.
  - lost: death_1y on ecg, auroc gain -0.033 [-0.0614, -0.0052]
- arm_b: PASS. Lost 0, inconclusive 9 of 31 retention checks. All three channels against original CSFM on ECG + PPG: RR better, invasive support better, ICD respiratory same.
- epoch17 for reference: lost 2 of 31 retention checks.
- Chosen encoder: arm_b.

## Primary metric per task

| part | task | input | metric | constant | spectral | original | epoch17 | arm_a | arm_b |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | rhythm | ecg | macro_f1 | 0.1536 |  | 0.8025 | 0.7966 | 0.8092 | 0.7915 |
| 1 | icd_cardiac | ecg | macro_f1 | 0.0 |  | 0.4057 | 0.4045 | 0.4061 | 0.4212 |
| 1 | pvc | ecg | auroc | 0.5 |  | 0.7433 | 0.746 | 0.7301 | 0.7442 |
| 1 | pac | ecg | auroc | 0.5 |  | 0.7251 | 0.7414 | 0.7164 | 0.7436 |
| 1 | avb1 | ecg | auroc | 0.5 |  | 0.9436 | 0.942 | 0.9265 | 0.9391 |
| 1 | qtc | ecg | mae | 31.9974 |  | 23.3865 | 23.0231 | 22.4784 | 22.843 |
| 1 | age | ecg | mae | 12.3165 |  | 9.3079 | 9.4676 | 9.3758 | 9.3144 |
| 1 | age | ppg | mae | 12.3165 |  | 9.332 | 9.2733 | 9.1183 | 8.7879 |
| 1 | age | ecg_ppg | mae | 12.3165 |  | 8.6275 | 8.5067 | 8.3989 | 8.2759 |
| 1 | sex | ecg | auroc | 0.5 |  | 0.7224 | 0.7163 | 0.7061 | 0.7305 |
| 1 | sex | ppg | auroc | 0.5 |  | 0.7225 | 0.7271 | 0.7232 | 0.7365 |
| 1 | sex | ecg_ppg | auroc | 0.5 |  | 0.7591 | 0.7536 | 0.767 | 0.7594 |
| 1 | bmi | ecg | mae | 5.4297 |  | 5.4889 | 5.4299 | 5.3507 | 5.4294 |
| 1 | bmi | ppg | mae | 5.4297 |  | 5.3062 | 5.4802 | 5.3826 | 5.4263 |
| 1 | bmi | ecg_ppg | mae | 5.4297 |  | 5.4531 | 5.4987 | 5.4579 | 5.2857 |
| 1 | sbp | ecg | mae | 14.7892 |  | 14.5615 | 14.4963 | 14.7305 | 14.4294 |
| 1 | sbp | ppg | mae | 14.7892 |  | 14.4456 | 14.4621 | 14.1669 | 14.2606 |
| 1 | sbp | ecg_ppg | mae | 14.7892 |  | 14.3388 | 14.0151 | 13.6848 | 13.8187 |
| 1 | dbp | ecg | mae | 8.4888 |  | 8.4884 | 8.5077 | 8.5491 | 8.5669 |
| 1 | dbp | ppg | mae | 8.4888 |  | 8.1232 | 8.0951 | 8.0535 | 7.97 |
| 1 | dbp | ecg_ppg | mae | 8.4888 |  | 8.3121 | 8.0158 | 8.0495 | 7.9302 |
| 1 | spo2 | ppg | mae | 2.5514 |  | 2.5192 | 2.5047 | 2.4691 | 2.4899 |
| 1 | death_1y | ecg | auroc | 0.5 |  | 0.7047 | 0.666 | 0.6717 | 0.7025 |
| 1 | death_1y | ppg | auroc | 0.5 |  | 0.6856 | 0.6662 | 0.6804 | 0.6769 |
| 1 | death_1y | ecg_ppg | auroc | 0.5 |  | 0.7344 | 0.7081 | 0.7115 | 0.7224 |
| 1 | death_hospital | ecg | auroc | 0.5 |  | 0.7013 | 0.6721 | 0.6995 | 0.7152 |
| 1 | death_hospital | ppg | auroc | 0.5 |  | 0.677 | 0.6966 | 0.6982 | 0.7154 |
| 1 | death_hospital | ecg_ppg | auroc | 0.5 |  | 0.6984 | 0.7123 | 0.7482 | 0.7437 |
| 1 | death_30d | ecg | auroc | 0.5 |  | 0.6885 | 0.674 | 0.6859 | 0.6919 |
| 1 | death_30d | ppg | auroc | 0.5 |  | 0.6678 | 0.6523 | 0.6784 | 0.667 |
| 1 | death_30d | ecg_ppg | auroc | 0.5 |  | 0.7087 | 0.6821 | 0.7021 | 0.7284 |
| 1 | death_24h | ecg | auroc | 0.5 |  | 0.7805 | 0.725 | 0.8651 | 0.8205 |
| 1 | death_24h | ppg | auroc | 0.5 |  | 0.7757 | 0.7937 | 0.7268 | 0.7383 |
| 1 | death_24h | ecg_ppg | auroc | 0.5 |  | 0.7894 | 0.7397 | 0.8041 | 0.7969 |
| 1 | rbbb | ecg | auroc | 0.5 |  | 0.7373 | 0.855 | 0.6859 | 0.7945 |
| 1 | hr | ecg | mae | 14.2231 |  | 3.9954 | 4.4043 | 3.6597 | 4.26 |
| 2 | rr_monitor | ecg_ppg | mae | 4.5052 | 4.9871 | 4.288 | 4.2952 | 4.2934 | 4.2757 |
| 2 | rr_monitor | all3 | mae | 4.5052 | 4.9871 | 4.288 | 3.6707 | 3.1431 | 4.0604 |
| 2 | rr_monitor | resp | mae | 4.5052 | 4.9871 | 4.288 | 3.3424 | 3.1611 | 3.4074 |
| 2 | support | ecg_ppg | auroc | 0.5 |  | 0.7203 | 0.7539 | 0.7721 | 0.776 |
| 2 | support | all3 | auroc | 0.5 |  | 0.7203 | 0.7649 | 0.805 | 0.7802 |
| 2 | support | resp | auroc | 0.5 |  | 0.7203 | 0.7192 | 0.7391 | 0.7169 |
| 2 | breathing | ecg_ppg | auroc | 0.5 |  | 0.5652 | 0.5393 | 0.5841 | 0.5944 |
| 2 | breathing | all3 | auroc | 0.5 |  | 0.5652 | 0.6078 | 0.6052 | 0.585 |
| 2 | breathing | resp | auroc | 0.5 |  | 0.5652 | 0.5895 | 0.6139 | 0.586 |
| 2 | rr_charted | ecg_ppg | mae | 4.4193 | 5.0488 | 4.2564 | 4.2459 | 4.241 | 4.2371 |
| 2 | rr_charted | all3 | mae | 4.4193 | 5.0488 | 4.2564 | 3.7153 | 3.1972 | 4.0313 |
| 2 | rr_charted | resp | mae | 4.4193 | 5.0488 | 4.2564 | 3.3724 | 3.214 | 3.4162 |
| 2 | icd_respiratory | ecg_ppg | macro_auroc | 0.5 |  | 0.6305 | 0.636 | 0.6397 | 0.639 |
| 2 | icd_respiratory | all3 | macro_auroc | 0.5 |  | 0.6305 | 0.6388 | 0.6431 | 0.6514 |
| 2 | icd_respiratory | resp | macro_auroc | 0.5 |  | 0.6305 | 0.5775 | 0.5905 | 0.5792 |

## Paired comparisons

`gain` is positive when the candidate is better than the reference.

| task | input | metric | reference | candidate | gain | ci_low | ci_high | margin | gate | verdict | draws |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rhythm | ecg | macro_f1 | original | epoch17 | -0.0059 | -0.022 | 0.011 | 0.02 | True | inconclusive | 1000 |
| rhythm | ecg | macro_f1 | original | arm_a | 0.0068 | -0.0085 | 0.0225 | 0.02 | True | retained | 1000 |
| rhythm | ecg | macro_f1 | original | arm_b | -0.0109 | -0.0255 | 0.003 | 0.02 | True | inconclusive | 1000 |
| icd_cardiac | ecg | macro_f1 | original | epoch17 | -0.0012 | -0.0182 | 0.0138 | 0.02 | True | retained | 1000 |
| icd_cardiac | ecg | macro_f1 | original | arm_a | 0.0003 | -0.023 | 0.0248 | 0.02 | True | inconclusive | 1000 |
| icd_cardiac | ecg | macro_f1 | original | arm_b | 0.0154 | -0.0027 | 0.0355 | 0.02 | True | retained | 1000 |
| pvc | ecg | auroc | original | epoch17 | 0.0027 | -0.0138 | 0.0186 | 0.02 | True | retained | 1000 |
| pvc | ecg | auroc | original | arm_a | -0.0133 | -0.0328 | 0.0056 | 0.02 | True | inconclusive | 1000 |
| pvc | ecg | auroc | original | arm_b | 0.0008 | -0.0167 | 0.0176 | 0.02 | True | retained | 1000 |
| pac | ecg | auroc | original | epoch17 | 0.0163 | -0.0114 | 0.0422 | 0.02 | True | retained | 1000 |
| pac | ecg | auroc | original | arm_a | -0.0086 | -0.0389 | 0.0232 | 0.02 | True | inconclusive | 1000 |
| pac | ecg | auroc | original | arm_b | 0.0186 | -0.0191 | 0.0573 | 0.02 | True | retained | 1000 |
| avb1 | ecg | auroc | original | epoch17 | -0.0017 | -0.0121 | 0.0095 | 0.02 | True | retained | 1000 |
| avb1 | ecg | auroc | original | arm_a | -0.0171 | -0.0353 | -0.0018 | 0.02 | True | inconclusive | 1000 |
| avb1 | ecg | auroc | original | arm_b | -0.0046 | -0.018 | 0.0101 | 0.02 | True | retained | 1000 |
| qtc | ecg | mae | original | epoch17 | 0.3634 | -0.0799 | 0.7876 | 3.0 | True | retained | 1000 |
| qtc | ecg | mae | original | arm_a | 0.9081 | 0.4525 | 1.312 | 3.0 | True | retained | 1000 |
| qtc | ecg | mae | original | arm_b | 0.5435 | 0.0685 | 0.9667 | 3.0 | True | retained | 1000 |
| age | ecg | mae | original | epoch17 | -0.1597 | -0.4485 | 0.0929 | 0.5 | True | retained | 1000 |
| age | ecg | mae | original | arm_a | -0.0679 | -0.3735 | 0.2124 | 0.5 | True | retained | 1000 |
| age | ecg | mae | original | arm_b | -0.0066 | -0.2839 | 0.2913 | 0.5 | True | retained | 1000 |
| age | ppg | mae | original | epoch17 | 0.0587 | -0.2012 | 0.3272 | 0.5 | True | retained | 1000 |
| age | ppg | mae | original | arm_a | 0.2137 | -0.0773 | 0.4903 | 0.5 | True | retained | 1000 |
| age | ppg | mae | original | arm_b | 0.5441 | 0.2314 | 0.85 | 0.5 | True | retained | 1000 |
| age | ecg_ppg | mae | original | epoch17 | 0.1208 | -0.1591 | 0.4221 | 0.5 | True | retained | 1000 |
| age | ecg_ppg | mae | original | arm_a | 0.2286 | -0.0796 | 0.543 | 0.5 | True | retained | 1000 |
| age | ecg_ppg | mae | original | arm_b | 0.3516 | 0.0426 | 0.6644 | 0.5 | True | retained | 1000 |
| sex | ecg | auroc | original | epoch17 | -0.0061 | -0.0313 | 0.0174 | 0.02 | True | inconclusive | 1000 |
| sex | ecg | auroc | original | arm_a | -0.0163 | -0.046 | 0.0122 | 0.02 | True | inconclusive | 1000 |
| sex | ecg | auroc | original | arm_b | 0.0081 | -0.0176 | 0.0357 | 0.02 | True | retained | 1000 |
| sex | ppg | auroc | original | epoch17 | 0.0046 | -0.0174 | 0.0268 | 0.02 | True | retained | 1000 |
| sex | ppg | auroc | original | arm_a | 0.0007 | -0.0219 | 0.0237 | 0.02 | True | inconclusive | 1000 |
| sex | ppg | auroc | original | arm_b | 0.014 | -0.0073 | 0.035 | 0.02 | True | retained | 1000 |
| sex | ecg_ppg | auroc | original | epoch17 | -0.0055 | -0.0318 | 0.0205 | 0.02 | True | inconclusive | 1000 |
| sex | ecg_ppg | auroc | original | arm_a | 0.0079 | -0.0172 | 0.0339 | 0.02 | True | retained | 1000 |
| sex | ecg_ppg | auroc | original | arm_b | 0.0003 | -0.024 | 0.0273 | 0.02 | True | inconclusive | 1000 |
| bmi | ecg | mae | original | epoch17 | 0.059 | -0.1098 | 0.2472 | 0.25 | True | retained | 1000 |
| bmi | ecg | mae | original | arm_a | 0.1382 | -0.0447 | 0.3321 | 0.25 | True | retained | 1000 |
| bmi | ecg | mae | original | arm_b | 0.0596 | -0.1236 | 0.2367 | 0.25 | True | retained | 1000 |
| bmi | ppg | mae | original | epoch17 | -0.174 | -0.3545 | 0.018 | 0.25 | True | inconclusive | 1000 |
| bmi | ppg | mae | original | arm_a | -0.0764 | -0.2824 | 0.1302 | 0.25 | True | inconclusive | 1000 |
| bmi | ppg | mae | original | arm_b | -0.1201 | -0.3192 | 0.0621 | 0.25 | True | inconclusive | 1000 |
| bmi | ecg_ppg | mae | original | epoch17 | -0.0455 | -0.2277 | 0.1373 | 0.25 | True | retained | 1000 |
| bmi | ecg_ppg | mae | original | arm_a | -0.0048 | -0.2073 | 0.1824 | 0.25 | True | retained | 1000 |
| bmi | ecg_ppg | mae | original | arm_b | 0.1675 | -0.0146 | 0.357 | 0.25 | True | retained | 1000 |
| sbp | ecg | mae | original | epoch17 | 0.0652 | -0.2595 | 0.342 | 1.0 | True | retained | 1000 |
| sbp | ecg | mae | original | arm_a | -0.169 | -0.4681 | 0.093 | 1.0 | True | retained | 1000 |
| sbp | ecg | mae | original | arm_b | 0.1321 | -0.1436 | 0.3979 | 1.0 | True | retained | 1000 |
| sbp | ppg | mae | original | epoch17 | -0.0165 | -0.3072 | 0.2525 | 1.0 | True | retained | 1000 |
| sbp | ppg | mae | original | arm_a | 0.2787 | -0.0226 | 0.5564 | 1.0 | True | retained | 1000 |
| sbp | ppg | mae | original | arm_b | 0.1851 | -0.1043 | 0.4592 | 1.0 | True | retained | 1000 |
| sbp | ecg_ppg | mae | original | epoch17 | 0.3237 | 0.0348 | 0.6407 | 1.0 | True | retained | 1000 |
| sbp | ecg_ppg | mae | original | arm_a | 0.6541 | 0.3055 | 0.9968 | 1.0 | True | retained | 1000 |
| sbp | ecg_ppg | mae | original | arm_b | 0.5201 | 0.1855 | 0.8658 | 1.0 | True | retained | 1000 |
| dbp | ecg | mae | original | epoch17 | -0.0193 | -0.2259 | 0.1775 | 1.0 | True | retained | 1000 |
| dbp | ecg | mae | original | arm_a | -0.0606 | -0.2591 | 0.1182 | 1.0 | True | retained | 1000 |
| dbp | ecg | mae | original | arm_b | -0.0784 | -0.2625 | 0.1111 | 1.0 | True | retained | 1000 |
| dbp | ppg | mae | original | epoch17 | 0.0282 | -0.143 | 0.2026 | 1.0 | True | retained | 1000 |
| dbp | ppg | mae | original | arm_a | 0.0697 | -0.1111 | 0.2323 | 1.0 | True | retained | 1000 |
| dbp | ppg | mae | original | arm_b | 0.1532 | -0.018 | 0.3405 | 1.0 | True | retained | 1000 |
| dbp | ecg_ppg | mae | original | epoch17 | 0.2963 | 0.1192 | 0.4776 | 1.0 | True | retained | 1000 |
| dbp | ecg_ppg | mae | original | arm_a | 0.2626 | 0.066 | 0.4448 | 1.0 | True | retained | 1000 |
| dbp | ecg_ppg | mae | original | arm_b | 0.382 | 0.2031 | 0.5715 | 1.0 | True | retained | 1000 |
| spo2 | ppg | mae | original | epoch17 | 0.0145 | -0.0097 | 0.0411 | 0.25 | True | retained | 1000 |
| spo2 | ppg | mae | original | arm_a | 0.0501 | 0.0198 | 0.08 | 0.25 | True | retained | 1000 |
| spo2 | ppg | mae | original | arm_b | 0.0293 | -0.0007 | 0.0581 | 0.25 | True | retained | 1000 |
| death_1y | ecg | auroc | original | epoch17 | -0.0387 | -0.0682 | -0.0116 | 0.02 | True | lost | 1000 |
| death_1y | ecg | auroc | original | arm_a | -0.033 | -0.0614 | -0.0052 | 0.02 | True | lost | 1000 |
| death_1y | ecg | auroc | original | arm_b | -0.0022 | -0.0309 | 0.025 | 0.02 | True | inconclusive | 1000 |
| death_1y | ppg | auroc | original | epoch17 | -0.0193 | -0.0437 | 0.0047 | 0.02 | True | inconclusive | 1000 |
| death_1y | ppg | auroc | original | arm_a | -0.0052 | -0.0305 | 0.022 | 0.02 | True | inconclusive | 1000 |
| death_1y | ppg | auroc | original | arm_b | -0.0086 | -0.0341 | 0.0162 | 0.02 | True | inconclusive | 1000 |
| death_1y | ecg_ppg | auroc | original | epoch17 | -0.0263 | -0.0477 | -0.0032 | 0.02 | True | lost | 1000 |
| death_1y | ecg_ppg | auroc | original | arm_a | -0.0229 | -0.0507 | 0.0035 | 0.02 | True | inconclusive | 1000 |
| death_1y | ecg_ppg | auroc | original | arm_b | -0.012 | -0.0366 | 0.0107 | 0.02 | True | inconclusive | 1000 |
| death_hospital | ecg | auroc | original | epoch17 | -0.0292 | -0.0683 | 0.0148 | 0.02 | True | inconclusive | 1000 |
| death_hospital | ecg | auroc | original | arm_a | -0.0018 | -0.0433 | 0.0409 | 0.02 | True | inconclusive | 1000 |
| death_hospital | ecg | auroc | original | arm_b | 0.0139 | -0.0267 | 0.0552 | 0.02 | True | inconclusive | 1000 |
| death_hospital | ppg | auroc | original | epoch17 | 0.0196 | -0.0089 | 0.0507 | 0.02 | True | retained | 1000 |
| death_hospital | ppg | auroc | original | arm_a | 0.0211 | -0.0119 | 0.0557 | 0.02 | True | retained | 1000 |
| death_hospital | ppg | auroc | original | arm_b | 0.0384 | 0.003 | 0.0726 | 0.02 | True | retained | 1000 |
| death_hospital | ecg_ppg | auroc | original | epoch17 | 0.014 | -0.0231 | 0.0533 | 0.02 | True | inconclusive | 1000 |
| death_hospital | ecg_ppg | auroc | original | arm_a | 0.0499 | 0.0127 | 0.0881 | 0.02 | True | retained | 1000 |
| death_hospital | ecg_ppg | auroc | original | arm_b | 0.0453 | 0.0068 | 0.0848 | 0.02 | True | retained | 1000 |
| death_30d | ecg | auroc | original | epoch17 | -0.0145 | -0.0454 | 0.0193 | 0.02 | True | inconclusive | 1000 |
| death_30d | ecg | auroc | original | arm_a | -0.0026 | -0.0436 | 0.0338 | 0.02 | True | inconclusive | 1000 |
| death_30d | ecg | auroc | original | arm_b | 0.0033 | -0.0322 | 0.0404 | 0.02 | True | inconclusive | 1000 |
| death_30d | ppg | auroc | original | epoch17 | -0.0154 | -0.0447 | 0.0127 | 0.02 | True | inconclusive | 1000 |
| death_30d | ppg | auroc | original | arm_a | 0.0106 | -0.0205 | 0.0418 | 0.02 | True | inconclusive | 1000 |
| death_30d | ppg | auroc | original | arm_b | -0.0007 | -0.031 | 0.0285 | 0.02 | True | inconclusive | 1000 |
| death_30d | ecg_ppg | auroc | original | epoch17 | -0.0267 | -0.058 | 0.0014 | 0.02 | True | inconclusive | 1000 |
| death_30d | ecg_ppg | auroc | original | arm_a | -0.0066 | -0.0384 | 0.0229 | 0.02 | True | inconclusive | 1000 |
| death_30d | ecg_ppg | auroc | original | arm_b | 0.0196 | -0.011 | 0.0497 | 0.02 | True | retained | 1000 |
| death_24h | ecg | auroc | original | epoch17 | -0.0555 | -0.1658 | 0.0553 | 0.02 | False | inconclusive | 1000 |
| death_24h | ecg | auroc | original | arm_a | 0.0847 | 0.0171 | 0.1615 | 0.02 | False | retained | 1000 |
| death_24h | ecg | auroc | original | arm_b | 0.0401 | -0.0286 | 0.1161 | 0.02 | False | inconclusive | 1000 |
| death_24h | ppg | auroc | original | epoch17 | 0.018 | -0.0613 | 0.0995 | 0.02 | False | inconclusive | 1000 |
| death_24h | ppg | auroc | original | arm_a | -0.0488 | -0.1381 | 0.0348 | 0.02 | False | inconclusive | 1000 |
| death_24h | ppg | auroc | original | arm_b | -0.0373 | -0.1429 | 0.0565 | 0.02 | False | inconclusive | 1000 |
| death_24h | ecg_ppg | auroc | original | epoch17 | -0.0498 | -0.1333 | 0.0296 | 0.02 | False | inconclusive | 1000 |
| death_24h | ecg_ppg | auroc | original | arm_a | 0.0147 | -0.1076 | 0.1396 | 0.02 | False | inconclusive | 1000 |
| death_24h | ecg_ppg | auroc | original | arm_b | 0.0074 | -0.0907 | 0.1006 | 0.02 | False | inconclusive | 1000 |
| rbbb | ecg | auroc | original | epoch17 | 0.1176 | 0.0395 | 0.1403 | 0.02 | False | retained | 945 |
| rbbb | ecg | auroc | original | arm_a | -0.0514 | -0.0761 | 0.0052 | 0.02 | False | inconclusive | 945 |
| rbbb | ecg | auroc | original | arm_b | 0.0572 | -0.0416 | 0.0852 | 0.02 | False | inconclusive | 945 |
| hr | ecg | mae | original | epoch17 | -0.409 | -0.5644 | -0.2711 | 1.0 | False | retained | 1000 |
| hr | ecg | mae | original | arm_a | 0.3356 | 0.1999 | 0.4636 | 1.0 | False | retained | 1000 |
| hr | ecg | mae | original | arm_b | -0.2646 | -0.4012 | -0.1313 | 1.0 | False | retained | 1000 |
| rr_monitor | ecg_ppg | mae | original | epoch17 | -0.0072 | -0.0528 | 0.0383 | 0.0 | True | same | 1000 |
| rr_monitor | ecg_ppg | mae | original | arm_a | -0.0054 | -0.0537 | 0.0391 | 0.0 | True | same | 1000 |
| rr_monitor | ecg_ppg | mae | original | arm_b | 0.0123 | -0.0339 | 0.0592 | 0.0 | True | same | 1000 |
| rr_monitor | ecg_ppg | mae | arm_a | arm_b | 0.0177 | -0.0289 | 0.0662 | 0.0 | True | same | 1000 |
| rr_monitor | all3 | mae | original | epoch17 | 0.6173 | 0.5392 | 0.6939 | 0.0 | True | better | 1000 |
| rr_monitor | all3 | mae | original | arm_a | 1.1449 | 1.0415 | 1.2503 | 0.0 | True | better | 1000 |
| rr_monitor | all3 | mae | original | arm_b | 0.2276 | 0.1702 | 0.2807 | 0.0 | True | better | 1000 |
| rr_monitor | all3 | mae | arm_a | arm_b | -0.9173 | -1.0139 | -0.8271 | 0.0 | True | worse | 1000 |
| rr_monitor | resp | mae | original | epoch17 | 0.9456 | 0.8495 | 1.053 | 0.0 | True | better | 1000 |
| rr_monitor | resp | mae | original | arm_a | 1.1269 | 1.018 | 1.2364 | 0.0 | True | better | 1000 |
| rr_monitor | resp | mae | original | arm_b | 0.8806 | 0.7833 | 0.9825 | 0.0 | True | better | 1000 |
| rr_monitor | resp | mae | arm_a | arm_b | -0.2463 | -0.2993 | -0.1943 | 0.0 | True | worse | 1000 |
| support | ecg_ppg | auroc | original | epoch17 | 0.0336 | 0.0174 | 0.049 | 0.0 | True | better | 1000 |
| support | ecg_ppg | auroc | original | arm_a | 0.0518 | 0.0328 | 0.0684 | 0.0 | True | better | 1000 |
| support | ecg_ppg | auroc | original | arm_b | 0.0557 | 0.0366 | 0.0749 | 0.0 | True | better | 1000 |
| support | ecg_ppg | auroc | arm_a | arm_b | 0.0039 | -0.0076 | 0.0164 | 0.0 | True | same | 1000 |
| support | all3 | auroc | original | epoch17 | 0.0446 | 0.0259 | 0.0618 | 0.0 | True | better | 1000 |
| support | all3 | auroc | original | arm_a | 0.0847 | 0.0629 | 0.1057 | 0.0 | True | better | 1000 |
| support | all3 | auroc | original | arm_b | 0.0599 | 0.0409 | 0.0783 | 0.0 | True | better | 1000 |
| support | all3 | auroc | arm_a | arm_b | -0.0248 | -0.0404 | -0.0089 | 0.0 | True | worse | 1000 |
| support | resp | auroc | original | epoch17 | -0.0011 | -0.0311 | 0.0298 | 0.0 | True | same | 1000 |
| support | resp | auroc | original | arm_a | 0.0188 | -0.012 | 0.0475 | 0.0 | True | same | 1000 |
| support | resp | auroc | original | arm_b | -0.0034 | -0.0347 | 0.0273 | 0.0 | True | same | 1000 |
| support | resp | auroc | arm_a | arm_b | -0.0222 | -0.0363 | -0.0077 | 0.0 | True | worse | 1000 |
| breathing | ecg_ppg | auroc | original | epoch17 | -0.0259 | -0.0957 | 0.0499 | 0.0 | False | same | 1000 |
| breathing | ecg_ppg | auroc | original | arm_a | 0.0189 | -0.0475 | 0.0833 | 0.0 | False | same | 1000 |
| breathing | ecg_ppg | auroc | original | arm_b | 0.0292 | -0.0446 | 0.1 | 0.0 | False | same | 1000 |
| breathing | ecg_ppg | auroc | arm_a | arm_b | 0.0103 | -0.0484 | 0.0699 | 0.0 | False | same | 1000 |
| breathing | all3 | auroc | original | epoch17 | 0.0426 | -0.0078 | 0.1009 | 0.0 | False | same | 1000 |
| breathing | all3 | auroc | original | arm_a | 0.0401 | -0.028 | 0.1084 | 0.0 | False | same | 1000 |
| breathing | all3 | auroc | original | arm_b | 0.0198 | -0.0695 | 0.0994 | 0.0 | False | same | 1000 |
| breathing | all3 | auroc | arm_a | arm_b | -0.0202 | -0.109 | 0.0534 | 0.0 | False | same | 1000 |
| breathing | resp | auroc | original | epoch17 | 0.0243 | -0.0471 | 0.0872 | 0.0 | False | same | 1000 |
| breathing | resp | auroc | original | arm_a | 0.0487 | -0.0281 | 0.1339 | 0.0 | False | same | 1000 |
| breathing | resp | auroc | original | arm_b | 0.0209 | -0.05 | 0.0946 | 0.0 | False | same | 1000 |
| breathing | resp | auroc | arm_a | arm_b | -0.0278 | -0.0914 | 0.0387 | 0.0 | False | same | 1000 |
| rr_charted | ecg_ppg | mae | original | epoch17 | 0.0105 | -0.0312 | 0.0493 | 0.0 | False | same | 1000 |
| rr_charted | ecg_ppg | mae | original | arm_a | 0.0154 | -0.0264 | 0.059 | 0.0 | False | same | 1000 |
| rr_charted | ecg_ppg | mae | original | arm_b | 0.0193 | -0.0241 | 0.063 | 0.0 | False | same | 1000 |
| rr_charted | ecg_ppg | mae | arm_a | arm_b | 0.0039 | -0.0357 | 0.0462 | 0.0 | False | same | 1000 |
| rr_charted | all3 | mae | original | epoch17 | 0.5411 | 0.4748 | 0.6098 | 0.0 | False | better | 1000 |
| rr_charted | all3 | mae | original | arm_a | 1.0591 | 0.9614 | 1.1687 | 0.0 | False | better | 1000 |
| rr_charted | all3 | mae | original | arm_b | 0.2251 | 0.1732 | 0.2768 | 0.0 | False | better | 1000 |
| rr_charted | all3 | mae | arm_a | arm_b | -0.834 | -0.9208 | -0.7534 | 0.0 | False | worse | 1000 |
| rr_charted | resp | mae | original | epoch17 | 0.884 | 0.7922 | 0.9874 | 0.0 | False | better | 1000 |
| rr_charted | resp | mae | original | arm_a | 1.0423 | 0.9448 | 1.1517 | 0.0 | False | better | 1000 |
| rr_charted | resp | mae | original | arm_b | 0.8402 | 0.7519 | 0.94 | 0.0 | False | better | 1000 |
| rr_charted | resp | mae | arm_a | arm_b | -0.2022 | -0.2487 | -0.1536 | 0.0 | False | worse | 1000 |
| icd_respiratory | ecg_ppg | macro_auroc | original | epoch17 | 0.0056 | -0.0166 | 0.0269 | 0.0 | True | same | 1000 |
| icd_respiratory | ecg_ppg | macro_auroc | original | arm_a | 0.0092 | -0.0121 | 0.0304 | 0.0 | True | same | 1000 |
| icd_respiratory | ecg_ppg | macro_auroc | original | arm_b | 0.0085 | -0.0157 | 0.0322 | 0.0 | True | same | 1000 |
| icd_respiratory | ecg_ppg | macro_auroc | arm_a | arm_b | -0.0007 | -0.0228 | 0.0206 | 0.0 | True | same | 1000 |
| icd_respiratory | all3 | macro_auroc | original | epoch17 | 0.0083 | -0.017 | 0.0334 | 0.0 | True | same | 1000 |
| icd_respiratory | all3 | macro_auroc | original | arm_a | 0.0127 | -0.0158 | 0.0394 | 0.0 | True | same | 1000 |
| icd_respiratory | all3 | macro_auroc | original | arm_b | 0.0209 | -0.0011 | 0.0443 | 0.0 | True | same | 1000 |
| icd_respiratory | all3 | macro_auroc | arm_a | arm_b | 0.0082 | -0.0145 | 0.0345 | 0.0 | True | same | 1000 |
| icd_respiratory | resp | macro_auroc | original | epoch17 | -0.0529 | -0.0916 | -0.0174 | 0.0 | True | worse | 1000 |
| icd_respiratory | resp | macro_auroc | original | arm_a | -0.04 | -0.0772 | -0.0047 | 0.0 | True | worse | 1000 |
| icd_respiratory | resp | macro_auroc | original | arm_b | -0.0512 | -0.0862 | -0.0161 | 0.0 | True | worse | 1000 |
| icd_respiratory | resp | macro_auroc | arm_a | arm_b | -0.0113 | -0.0384 | 0.0169 | 0.0 | True | same | 1000 |

## Skipped

- lbbb on ecg: positives: fit 33, validation 0

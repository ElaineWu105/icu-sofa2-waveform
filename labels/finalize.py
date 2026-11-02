"""Join final SOFA-2 days to cut waveform days; write anchors and aggregate reports.

An anchor needs a cut day passing 384/480 valid clips as baseline and input.
Private anchors hold patient identifiers; cohort_report.json is aggregate only.
"""
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PRIVATE = HERE / "private"
WAVEFORM = HERE.parent / "waveform/icu_cut/output"
TAIL = HERE.parent / "waveform/icu_cut/output_tail"
SOURCES = [root for root in (WAVEFORM, TAIL) if (root / "plan/plan.sqlite").is_file()]
E = os.environ["MIMIC_IV_DIR"].rstrip("/") + "/"
ORGANS = ("brain", "resp", "cv", "liver", "kidney", "hemo")
MIN_CLIPS = 384


def total(frame, suffix="s2"):
    return frame[[f"{organ}_{suffix}" for organ in ORGANS]].fillna(0).sum(axis=1)


def versus(left, right):
    return {"same_pct": round(float((left == right).mean()) * 100, 1), "higher_pct": round(float((left > right).mean()) * 100, 1),
            "lower_pct": round(float((left < right).mean()) * 100, 1)}


def distribution(delta):
    delta = delta.dropna()
    if not len(delta):
        return {"n": 0}
    return {"n": len(delta), "down_pct": round(float((delta < 0).mean()) * 100, 1), "same_pct": round(float((delta == 0).mean()) * 100, 1),
            "up_pct": round(float((delta > 0).mean()) * 100, 1), "up_ge2_pct": round(float((delta >= 2).mean()) * 100, 1),
            "abs_ge2_pct": round(float((delta.abs() >= 2).mean()) * 100, 1), "abs_ge3_pct": round(float((delta.abs() >= 3).mean()) * 100, 1)}


def day_one(table):
    first = table[table.day == 0].copy()
    died = first.deathtime.notna() & (first.deathtime <= first.outtime + pd.Timedelta(hours=6))
    two, one = total(first), total(first, "s1")
    report = {"stays": len(first), "icu_deaths": int(died.sum()), "icu_mortality_pct": round(float(died.mean()) * 100, 2),
              "auroc_sofa2": round(roc_auc_score(died, two), 4), "auroc_sofa1": round(roc_auc_score(died, one), 4),
              "sofa2_median_iqr": two.quantile([.5, .25, .75]).tolist(), "sofa1_median_iqr": one.quantile([.5, .25, .75]).tolist(),
              "sofa2_versus_sofa1": versus(two, one),
              "organ_mean_sofa2": {o: round(float(first[f"{o}_s2"].fillna(0).mean()), 3) for o in ORGANS},
              "organ_mean_sofa1": {o: round(float(first[f"{o}_s1"].fillna(0).mean()), 3) for o in ORGANS},
              "evidence_pct": {o: round(float(first[f"{o}_fresh"].mean()) * 100, 1) for o in ORGANS}}
    variants = {}
    for name, replaced in (("no_urine_or_rrt_in_kidney", ("kidney_s2", "kidney_s2_creatinine")),
                           ("no_rrt_in_kidney", ("kidney_s2", "kidney_s2_no_rrt")), ("abg_only_resp", ("resp_s2", "resp_s2_abg"))):
        changed = first.copy()
        changed[replaced[0]] = changed[replaced[1]]
        value = total(changed)
        variants[name] = {**versus(value, one), "auroc": round(roc_auc_score(died, value), 4)}
    report["sofa2_variants"] = variants
    return report


def main():
    daily = pd.read_pickle(PRIVATE / "daily.pkl").sort_values(["stay_id", "day"])
    plans, clips = [], []
    for root in SOURCES:
        db = sqlite3.connect(f"file:{root / 'plan/plan.sqlite'}?mode=ro", uri=True)
        plans.append(pd.read_sql("SELECT stay_id, day, split, fate FROM days", db))
        stays = pd.read_sql("SELECT stay_id, subject_id, hadm_id, split FROM stays", db)
        db.close()
        for path in (root / "cut").glob("days/*/*_meta.npz"):
            with np.load(path, allow_pickle=False) as meta:
                summary = json.loads(str(meta["summary"]))
            clips.append((summary["stay_id"], summary["day"], summary["valid_clips"], summary["all_three"], summary["planned"]))
    plan = pd.concat(plans, ignore_index=True)
    clips = pd.DataFrame(clips, columns=["stay_id", "day", "valid_clips", "all_three_windows", "planned_windows"])
    for organ in ORGANS:
        real = daily[f"{organ}_s2"].where(daily[f"{organ}_fresh"])
        daily[f"{organ}_real"] = real
        daily[f"{organ}_carry1"] = real.fillna(real.groupby(daily.stay_id).shift(1))
        full = daily[f"{organ}_s2"].groupby(daily.stay_id).ffill()
        daily[f"{organ}_full"] = full.where(~((daily.day == 0) & full.isna()), 0.).groupby(daily.stay_id).ffill().fillna(0.)
    columns = ["stay_id", "day"] + [f"{o}_{v}" for o in ORGANS for v in ("real", "carry1", "full")]
    future = daily[columns].copy()
    future["day"] -= 1
    anchors = (plan.merge(clips, on=["stay_id", "day"]).merge(stays[["stay_id", "subject_id", "hadm_id"]], on="stay_id")
               .merge(daily[columns], on=["stay_id", "day"]).merge(future, on=["stay_id", "day"], how="left", suffixes=("_base", "_next")))
    report = {"day_one_all_stays": day_one(daily)}
    first_stays = stays.sort_values("stay_id")
    intimes = daily[daily.day == 0][["stay_id", "start"]].merge(stays, on="stay_id").sort_values("start")
    report["day_one_first_stay_per_patient"] = day_one(daily[daily.stay_id.isin(intimes.groupby("subject_id").head(1).stay_id)])
    passing = anchors[anchors.valid_clips >= MIN_CLIPS].copy()
    stayed = passing[passing.fate == "stayed"]
    labels = {}
    for organ in ORGANS:
        delta = stayed[f"{organ}_real_next"] - stayed[f"{organ}_real_base"]
        passing[f"{organ}_delta"] = passing[f"{organ}_real_next"] - passing[f"{organ}_real_base"]
        labels[organ] = {**distribution(delta), "coverage_pct": round(float(delta.notna().mean()) * 100, 1),
                         "patients": int(stayed[delta.notna()].subject_id.nunique()),
                         "by_split": delta.notna().groupby(stayed.split).sum().astype(int).to_dict(),
                         "baseline_evidence_pct": round(float(stayed[f"{organ}_real_base"].notna().mean()) * 100, 1),
                         "future_evidence_pct": round(float(stayed[f"{organ}_real_next"].notna().mean()) * 100, 1)}
    totals = {}
    for name, title in (("real", "six_real_both_ends"), ("carry1", "carry_at_most_one_window"), ("full", "full_sofa2_rule")):
        base = stayed[[f"{o}_{name}_base" for o in ORGANS]]
        next_ = stayed[[f"{o}_{name}_next" for o in ORGANS]]
        usable = base.notna().all(axis=1) & next_.notna().all(axis=1)
        delta = pd.Series(next_.values.sum(axis=1) - base.values.sum(axis=1), index=stayed.index)[usable]
        real = np.concatenate([stayed.loc[usable, [f"{o}_real_{side}" for o in ORGANS]].notna().values for side in ("base", "next")])
        totals[title] = {**distribution(delta), "coverage_pct": round(float(usable.mean()) * 100, 1),
                         "patients": int(stayed[usable].subject_id.nunique()), "real_endpoints_pct": round(float(real.mean()) * 100, 1),
                         "by_split": usable.groupby(stayed.split).sum().astype(int).to_dict()}
        passing[f"total_delta_{name}"] = delta.reindex(passing.index)
    pairs = pd.DataFrame({o: stayed[f"{o}_real_base"].notna() & stayed[f"{o}_real_next"].notna() for o in ORGANS}).sum(axis=1)

    patients = pd.read_csv(E + "hosp/patients.csv.gz", usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
    icu = pd.read_csv(E + "icu/icustays.csv.gz", usecols=["stay_id", "intime", "los"], parse_dates=["intime"])
    people = passing.drop_duplicates("stay_id").merge(icu, on="stay_id").merge(patients, on="subject_id")
    people["age"] = people.anchor_age + people.intime.dt.year - people.anchor_year
    demographics = {}
    for split, group in list(people.groupby("split")) + [("all", people)]:
        unique = group.sort_values("intime").drop_duplicates("subject_id")
        demographics[split] = {"patients": len(unique), "stays": len(group), "age_median_iqr": unique.age.quantile([.5, .25, .75]).tolist(),
                               "female_pct": round(float((unique.gender == "F").mean()) * 100, 1),
                               "icu_los_days_median_iqr": [round(x, 2) for x in group.los.quantile([.5, .25, .75])]}
    report.update(
        anchors_planned=len(anchors), anchors_passing=len(passing), anchors_stayed=len(stayed),
        passing_by_split=passing.split.value_counts().to_dict(), passing_by_fate=passing.fate.value_counts().to_dict(),
        passing_patients=int(passing.subject_id.nunique()), passing_stays=int(passing.stay_id.nunique()),
        patients_by_split=passing.groupby("split").subject_id.nunique().to_dict(),
        organ_delta_real=labels, total_delta=totals, paired_real_organs_per_anchor={("at_most_2" if k == 2 else int(k)): int(v) for k, v in pairs.clip(lower=2).value_counts().sort_index().items()},
        waveform={"all_three_pct_of_slots_median": round(float((passing.all_three_windows / 8640).median()) * 100, 1),
                  "valid_clips_median_iqr": passing.valid_clips.quantile([.5, .25, .75]).tolist()},
        demographics=demographics,
        rrt_days_pct=round(float(daily.kidney_rrt.eq(True).mean()) * 100, 1), rrt_exact_days_pct=round(float(daily.kidney_rrt_exact.eq(True).mean()) * 100, 1),
        ne_half_dose_changes_cv_pct=round(float((daily.cv_s2 != daily.cv_s2_nehalf)[daily.cv_s2.notna()].mean()) * 100, 1))
    passing.to_pickle(PRIVATE / "anchors.pkl")
    (HERE / "cohort_report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()

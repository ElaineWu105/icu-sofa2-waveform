"""Shared paths and definitions for the CSFM downstream comparison."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRIVATE = Path(os.environ.get("CSFM_DOWNSTREAM_PRIVATE", HERE / "private"))
RESULTS = PRIVATE.parent / "results"
WAVEFORM = ROOT / "waveform/icu_cut/output"
PLAN = WAVEFORM / "plan/plan.sqlite"
CUT = WAVEFORM / "cut/days"
ANCHORS = ROOT / "labels/private/anchors.pkl"
CHART = HERE / "label_audit/private/chart.pkl"
EHR = os.environ["MIMIC_IV_DIR"].rstrip("/") + "/"
ORIGINAL = ROOT / "pretrained_model_for_share/CSFM_base.pth"
ARMS = {"arm_a": ROOT / "finetune/arms/runs/arm_A/selected.pt",
        "arm_b": ROOT / "finetune/arms/runs/arm_B/selected.pt",
        "arm_b2": ROOT / "finetune/arms/runs/arm_B2/selected.pt",
        "arm_b3": ROOT / "finetune/arms/runs/arm_B3/selected.pt"}
ENCODERS = ("original", *ARMS)
# Arms trained with text. B2 and B3 were added on 2026-11-06, after the first results.
TEXT_ARMS = ("arm_b", "arm_b2", "arm_b3")
# Tasks whose label is not charted vitals, nursing fields or ECG machine text, so the text arms'
# training text does not hold it. Arm B's chest reports can still mention respiratory diagnoses.
TEXT_FREE_TASKS = ("icd_cardiac", "icd_respiratory", "age", "sex", "bmi",
                   "death_1y", "death_hospital", "death_30d", "death_24h")
# Positions in the cut signal: ECG, PPG, Resp.
INPUTS = {"ecg": (0,), "ppg": (1,), "ecg_ppg": (0, 1), "all3": (0, 1, 2), "resp": (2,)}
RESP_INPUTS = ("all3", "resp")
ROW_BYTES = 3 * 2500 * 4
SLOT_US, DAY_US, SLOTS_PER_HOUR = 10_000_000, 86_400_000_000, 360
READY, ALL_THREE = 1, 7
# ICD-10 and ICD-9 prefixes per group.
ICD = {"atrial_fibrillation_flutter": (("I48",), ("42731", "42732")),
       "heart_failure": (("I50",), ("428",)),
       "coronary_artery_disease": (("I25",), ("414",)),
       "acute_mi": (("I21", "I22"), ("410",)),
       "valve_disease": (("I05", "I06", "I07", "I08", "I34", "I35", "I36", "I37"), ("394", "395", "396", "397", "424")),
       "cardiomyopathy": (("I42", "I43"), ("425",)),
       "conduction_disorder": (("I44", "I45"), ("426",)),
       "ventricular_arrhythmia": (("I472", "I490"), ("4271", "4274")),
       "hypertension": (("I10", "I11", "I12", "I13", "I15"), ("401", "402", "403", "404", "405")),
       "stroke": (("I60", "I61", "I62", "I63"), ("430", "431", "432", "433", "434")),
       "ckd": (("N18",), ("585",)),
       "diabetes": (("E10", "E11", "E13"), ("250",)),
       "copd": (("J44",), ("4912", "496")),
       "pneumonia": (("J12", "J13", "J14", "J15", "J16", "J17", "J18"), ("480", "481", "482", "483", "484", "485", "486")),
       "acute_respiratory_failure": (("J960",), ("51881",)),
       "ards": (("J80",), ("51882",)),
       "sleep_apnea": (("G473",), ("32723",)),
       "pulmonary_embolism": (("I26",), ("4151",)),
       "sepsis": (("A40", "A41", "R652"), ("038", "99591", "99592"))}
ICD_CARDIAC = ("atrial_fibrillation_flutter", "heart_failure", "coronary_artery_disease", "acute_mi", "valve_disease",
               "cardiomyopathy", "conduction_disorder", "ventricular_arrhythmia")
ICD_RESP = ("acute_respiratory_failure", "pneumonia", "copd", "sleep_apnea")

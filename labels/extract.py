"""Filter MIMIC-IV 3.1 rows for SOFA-2 to waveform patients, with storetime. Read-only."""
import os
import sys
import pandas as pd

E = os.environ["MIMIC_IV_DIR"].rstrip("/") + "/"
OUT = sys.argv[2]
W = os.environ["MIMIC_WAVEFORM_DIR"].rstrip("/") + "/v0.2_tmp_starttime_index.csv"
SUBJECTS = set(pd.read_csv(W, dtype=str).subject_id.str.lstrip('p').astype(int))
CHART = [220739, 223900, 223901, 220052, 220181, 225312, 220277, 223835, 226732, 223849, 229314,
         224660, 229268, 229270, 229277, 228193, 224322, 225982, 228154, 229671, 229254, 228198, 220128, 229897,
         226512, 224639, 225126,
         # rrt.sql dialysis_active
         226499, 224154, 225183, 227438, 224191, 225806, 225807, 228004, 228005, 228006, 224144, 224145,
         224153, 226457, 225965]
LAB = [50821, 50816, 52033, 50820, 50882, 50971, 50822, 50885, 50912, 51265]
URINE = [226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227488, 227489]
PROCEDURE = [225441, 225802, 225803, 225805, 225809, 225955]
DRUG = [221906, 221289, 229617, 221662, 221653, 221749, 229630, 229631, 229632, 222315, 221986, 229709, 229764,
        222168, 221668, 225150, 229420, 221712, 221385, 221824, 227536, 227525]


def run(path, columns, keep, name, chunk=4_000_000):
    parts, seen = [], 0
    for frame in pd.read_csv(E + path, usecols=columns, chunksize=chunk, low_memory=False, keep_default_na=name != 'chart'):
        seen += len(frame)
        frame = frame[keep(frame)]
        if len(frame):
            parts.append(frame)
        print(name, seen, flush=True)
    pd.concat(parts).to_pickle(f'{OUT}/{name}.pkl')
    print(name, 'DONE', sum(map(len, parts)), flush=True)


task = sys.argv[1]
if task == 'chart':
    run('icu/chartevents.csv.gz', ['subject_id', 'stay_id', 'charttime', 'storetime', 'itemid', 'value', 'valuenum'],
        lambda c: c.itemid.isin(CHART) & c.subject_id.isin(SUBJECTS), 'chart')
if task == 'lab':
    run('hosp/labevents.csv.gz', ['subject_id', 'specimen_id', 'charttime', 'storetime', 'itemid', 'valuenum', 'value'],
        lambda c: c.itemid.isin(LAB) & c.subject_id.isin(SUBJECTS), 'lab')
if task == 'emar':
    run('hosp/emar.csv.gz', ['subject_id', 'charttime', 'storetime', 'medication', 'event_txt'],
        lambda c: c.subject_id.isin(SUBJECTS) & c.medication.fillna('').str.contains(
            'haloperidol|haldol|quetiapine|seroquel|olanzapine|zyprexa', case=False, regex=True), 'emar')
if task == 'small':
    run('icu/inputevents.csv.gz', ['subject_id', 'stay_id', 'starttime', 'endtime', 'storetime', 'itemid', 'amount', 'rate',
                                   'rateuom', 'patientweight', 'statusdescription', 'ordercategorydescription'],
        lambda c: c.itemid.isin(DRUG) & c.subject_id.isin(SUBJECTS), 'input')
    run('icu/outputevents.csv.gz', ['subject_id', 'stay_id', 'charttime', 'storetime', 'itemid', 'value'],
        lambda c: c.itemid.isin(URINE) & c.subject_id.isin(SUBJECTS), 'urine')
    run('icu/procedureevents.csv.gz', ['subject_id', 'stay_id', 'starttime', 'endtime', 'storetime', 'itemid', 'value'],
        lambda c: c.itemid.isin(PROCEDURE) & c.subject_id.isin(SUBJECTS), 'proc')

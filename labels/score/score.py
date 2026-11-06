"""SOFA-2 (Ranzani, JAMA 2025, Table 2) on ICU days for every adult ICU stay of waveform patients.

Read-only on MIMIC. Patient-level tables stay in labels/private; printed output is aggregate.
Support, RRT, urine, FiO2 and vasopressor rates follow MIT-LCP mimic-code. A record counts
for a day only if stored by that day's end. SOFA-1 uses the same evidence as a comparator.
Each ICU day d is [intime+24d, intime+24(d+1)). An anchor t=intime+24(d+1) has
baseline=day d, future=day d+1, and waveform input=day d.
"""
import os
import numpy as np, pandas as pd, pickle, sys
from sklearn.metrics import roc_auc_score

from infusion import day_doses

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'private', '')
E = os.environ["MIMIC_IV_DIR"].rstrip("/") + "/"
W = os.environ["MIMIC_WAVEFORM_DIR"].rstrip("/") + "/v0.2_tmp_starttime_index.csv"
ORG = ('brain', 'resp', 'cv', 'liver', 'kidney', 'hemo')
DAY = pd.Timedelta(hours=24)
HOUR = pd.Timedelta(hours=1)
MINUTE = pd.Timedelta(minutes=1)

# ---------- cohort ----------
wave = pd.read_csv(W, dtype=str)
wave['subject_id'] = wave.subject_id.str.lstrip('p').astype(int)
for c in ('starttime', 'endtime'):
    wave[c] = pd.to_datetime(wave[c].str.slice(0, 19))
wave = wave.dropna(subset=['starttime', 'endtime'])
wave = wave[wave.endtime > wave.starttime]
ic = pd.read_csv(E + 'icu/icustays.csv.gz', parse_dates=['intime', 'outtime'])
ad = pd.read_csv(E + 'hosp/admissions.csv.gz', usecols=['hadm_id', 'deathtime'], parse_dates=['deathtime'])
pt = pd.read_csv(E + 'hosp/patients.csv.gz', usecols=['subject_id', 'anchor_age', 'anchor_year'])
ic = ic[ic.subject_id.isin(set(wave.subject_id))].merge(ad, on='hadm_id', how='left').merge(pt, on='subject_id', how='left')
ic['age'] = ic.anchor_age + ic.intime.dt.year - ic.anchor_year
ic = ic[(ic.age >= 18) & ic.outtime.notna()].sort_values('intime')
st = ic.set_index('stay_id')
st['ndays'] = np.ceil((st.outtime - st.intime) / DAY).astype(int)
print('cohort stays', len(st), 'patients', st.subject_id.nunique(), 'icu days', int(st.ndays.sum()), flush=True)
IN = st.intime.to_dict(); OUT = st.outtime.to_dict(); STAYS = set(st.index)


def add_day(df, tcol='charttime'):
    df = df[df.stay_id.isin(STAYS)].copy()
    df[tcol] = pd.to_datetime(df[tcol])
    df = df[df[tcol].notna()]
    df['intime'] = df.stay_id.map(IN); df['outtime'] = df.stay_id.map(OUT)
    df = df[(df[tcol] >= df.intime) & (df[tcol] < df.outtime)]
    df['day'] = ((df[tcol] - df.intime) / DAY).astype(int)
    df['hour'] = ((df[tcol] - df.intime) / HOUR).astype(int)
    if 'storetime' in df:
        stored = pd.to_datetime(df.storetime, errors='coerce')
        df = df[stored.notna() & (stored <= df.intime + (df.day + 1) * DAY)]
    return df


def to_stay(df, tcol='charttime'):
    """Attach subject-keyed events to a cohort stay by time."""
    df = df.copy(); df[tcol] = pd.to_datetime(df[tcol])
    df = df[df[tcol].notna()].sort_values(tcol)
    s = st.reset_index()[['stay_id', 'subject_id', 'intime', 'outtime']].sort_values('intime')
    m = pd.merge_asof(df, s, left_on=tcol, right_on='intime', by='subject_id', direction='backward')
    m = m[m.stay_id.notna() & (m[tcol] < m.outtime)]
    m['stay_id'] = m.stay_id.astype(int)
    return m.drop(columns=['intime', 'outtime'])


chart = pd.read_pickle(R + 'raw/chart.pkl')
chart['valuenum'] = pd.to_numeric(chart.valuenum, errors='coerce')
chart = add_day(chart)
lab = add_day(to_stay(pd.read_pickle(R + 'raw/lab.pkl')))
print('chart', len(chart), 'lab', len(lab), flush=True)
day = {}


def col(name, ser):
    day[name] = ser


# ---------- liver & hemostasis ----------
bili = lab[(lab.itemid == 50885) & (lab.valuenum > 0)].groupby(['stay_id', 'day']).valuenum.max()
col('liver_s2', pd.cut(bili, [-1, 1.2, 3, 6, 12, 1e9], labels=False).astype(float))
col('liver_s1', pd.Series(np.select([bili >= 12, bili >= 6, bili >= 2, bili >= 1.2], [4, 3, 2, 1], 0), bili.index).astype(float))
col('liver_fresh', pd.Series(True, bili.index))
plt = lab[(lab.itemid == 51265) & (lab.valuenum > 0)].groupby(['stay_id', 'day']).valuenum.min()
col('hemo_s2', pd.Series(np.select([plt <= 50, plt <= 80, plt <= 100, plt <= 150], [4, 3, 2, 1], 0), plt.index).astype(float))
col('hemo_s1', pd.Series(np.select([plt < 20, plt < 50, plt < 100, plt < 150], [4, 3, 2, 1], 0), plt.index).astype(float))
col('hemo_fresh', pd.Series(True, plt.index))

# ---------- input events ----------
inp = pd.read_pickle(R + 'raw/input.pkl')
inp = inp[inp.stay_id.isin(STAYS)].copy()
crrt_drugs = inp[inp.itemid.isin([227536, 227525]) & (pd.to_numeric(inp.amount, errors='coerce') > 0)].copy()
inp = inp[~inp.statusdescription.fillna('').str.contains('Rewritten')]
inp = inp[inp.ordercategorydescription.fillna('').str.contains('Continuous')]
for c in ('starttime', 'endtime', 'storetime'):
    inp[c] = pd.to_datetime(inp[c])
inp = inp[inp.endtime > inp.starttime]
inp['rate'] = pd.to_numeric(inp.rate, errors='coerce')
u = inp.rateuom.fillna('').str.lower()
inp['dose'] = np.where(u.str.contains('mcg/kg/min'), inp.rate,
              np.where(u.str.contains('mg/kg/min') & (inp.patientweight == 1), inp.rate,  # norepinephrine.sql
              np.where(u.str.contains('mg/kg/min'), inp.rate * 1000,
              np.where(u.str.contains('mcg/min'), inp.rate / inp.patientweight, np.nan))))
CLASS = {221906: 'ne', 221289: 'epi', 229617: 'epi', 221662: 'dopa', 221653: 'dobu'}
for i in (221749, 229630, 229631, 229632, 222315, 221986, 229709, 229764):
    CLASS[i] = 'other'
SAME_DRUG = {229617: 221289, 229630: 221749, 229631: 221749, 229632: 221749, 229764: 229709}
SED = {222168, 221668, 225150, 229420, 221712, 221385}
sed_iv = {}
for r in inp[inp.itemid.isin(SED) & (inp.rate > 0)].itertuples():
    sed_iv.setdefault(r.stay_id, []).append((r.starttime, r.endtime))
for k in sed_iv:
    x = sorted(sed_iv[k]); m = [list(x[0])]
    for a, b in x[1:]:
        if a <= m[-1][1]:
            m[-1][1] = max(m[-1][1], b)
        else:
            m.append([a, b])
    sed_iv[k] = m
V = inp[inp.itemid.isin(CLASS) & (inp.rate > 0)]
fragments = {}
for r in V.itertuples():
    c = CLASS[r.itemid]; i0 = IN[r.stay_id]; o = OUT[r.stay_id]
    a = max(r.starttime, i0); b = min(r.endtime, o)
    if b <= a or pd.isna(r.storetime):
        continue
    d = r.dose if c != 'other' else 1.0
    if not np.isfinite(d):
        d = 0.0  # active but dose unknown: counts as drug use, adds no dose
    known = max(0, int(np.ceil((r.storetime - i0) / DAY)) - 1)  # known by that day's end
    fragments.setdefault((r.stay_id, c), {}).setdefault(SAME_DRUG.get(r.itemid, r.itemid), []).append(
        ((a - i0) / MINUTE, (b - i0) / MINUTE, d, known))

mp = chart[chart.itemid.isin([220052, 220181, 225312]) & chart.valuenum.between(10, 200)].groupby(['stay_id', 'day']).valuenum.min()
mech = chart[(chart.itemid.isin([224322, 228154, 229671, 229254, 228198, 220128]) & (chart.valuenum > 0))].groupby(['stay_id', 'day']).size()
ecmo = chart[(chart.itemid.isin([229270, 229277]) & (chart.valuenum > 0))].groupby(['stay_id', 'day']).size()
va = chart[(chart.itemid == 229268) & chart.value.fillna('').str.upper().str.contains('VA')].groupby(['stay_id', 'day']).size()
rowsv = []
for stay, r in st.iterrows():
    for d in range(r.ndays):
        got = {}
        for c in ('ne', 'epi', 'dopa', 'dobu', 'other'):
            doses = [x for x in (day_doses(f, d) for f in fragments.get((stay, c), {}).values()) if x is not None]
            if doses:  # SOFA-2 note j: continuous infusion for at least 1 hour
                got[c] = np.max(doses, axis=0)
        if not got and (stay, d) not in mp.index and (stay, d) not in mech.index:
            continue
        nee = None
        for c in ('ne', 'epi'):
            if c in got:
                nee = got[c] if nee is None else nee + got[c]
        peak = float(nee.max()) if nee is not None else None
        ne_half = None
        if 'ne' in got or 'epi' in got:
            ne_half = float(((got['ne'] * .5 if 'ne' in got else 0) + (got['epi'] if 'epi' in got else 0)).max())
        oth = 'other' in got or 'dobu' in got
        dop = float(got['dopa'].max()) if 'dopa' in got else None
        mpv = mp.get((stay, d), np.nan); mc = (stay, d) in mech.index or (stay, d) in va.index

        def grade(p):
            if p is not None:
                other = oth or dop is not None
                if p > .4:
                    g = 4
                elif p > .2:
                    g = 4 if other else 3
                else:
                    g = 3 if other else 2
            elif dop is not None and not oth:
                g = 2 if dop <= 20 else (3 if dop <= 40 else 4)
            elif oth or dop is not None:
                g = 2
            else:
                g = 1 if mpv < 70 else 0
            return 4 if mc else g
        g2 = grade(peak); g2h = grade(ne_half)
        s1 = 1 if mpv < 70 else 0
        if 'dobu' in got or (dop is not None and dop <= 5):
            s1 = max(s1, 2)
        if dop is not None and dop > 5:
            s1 = max(s1, 3)
        if dop is not None and dop > 15:
            s1 = max(s1, 4)
        if peak is not None:
            s1 = max(s1, 3 if peak <= .1 else 4)
        rowsv.append((stay, d, float(g2), float(g2h), float(s1), True))
cv = pd.DataFrame(rowsv, columns=['stay_id', 'day', 'cv_s2', 'cv_s2_nehalf', 'cv_s1', 'cv_fresh']).set_index(['stay_id', 'day'])
for c in cv:
    col(c, cv[c])
print('cv done', len(cv), flush=True)

# ---------- respiratory ----------
spec = lab[(lab.itemid == 52033) & (lab.value.fillna('').str.upper().str.strip() == 'ART.')].specimen_id
abg = lab[(lab.itemid == 50821) & lab.specimen_id.isin(set(spec)) & (lab.valuenum > 0)][['stay_id', 'day', 'charttime', 'specimen_id', 'valuenum']].rename(columns={'valuenum': 'pao2'})
gf = lab[(lab.itemid == 50816) & (lab.valuenum > 0)].groupby('specimen_id').valuenum.max()


def fix_f(v):
    v = np.asarray(v, float); v = np.where(v > 1, v / 100, v)
    return np.where((v >= .21) & (v <= 1), v, np.nan)


abg['f_gas'] = fix_f(abg.specimen_id.map(gf))
cf = chart[(chart.itemid == 223835) & (chart.valuenum > 0)][['stay_id', 'charttime', 'valuenum']].copy()
cf['fio2'] = fix_f(cf.valuenum); cf = cf.dropna(subset=['fio2']).sort_values('charttime')
abg = pd.merge_asof(abg.sort_values('charttime'), cf[['stay_id', 'charttime', 'fio2']], on='charttime', by='stay_id', direction='backward', tolerance=4 * HOUR)
abg['f'] = abg.f_gas.fillna(abg.fio2)
dev = chart[chart.itemid == 226732].copy(); dev['v'] = dev.value.fillna('').str.strip().str.lower()
ADV = {'endotracheal tube', 'bipap mask', 'cpap mask', 'high flow nasal cannula'}  # ventilation.sql
MODES = ['(S) CMV', 'APRV', 'APRV/Biphasic+ApnPress', 'APRV/Biphasic+ApnVol', 'APV (cmv)', 'Ambient', 'Apnea Ventilation',
         'CMV', 'CMV/ASSIST', 'CMV/ASSIST/AutoFlow', 'CMV/AutoFlow', 'CPAP/PPS', 'CPAP/PSV', 'CPAP/PSV+Apn TCPL',
         'CPAP/PSV+ApnPres', 'CPAP/PSV+ApnVol', 'MMV', 'MMV/AutoFlow', 'MMV/PSV', 'MMV/PSV/AutoFlow', 'P-CMV', 'PCV+',
         'PCV+/PSV', 'PCV+Assist', 'PRES/AC', 'PRVC/AC', 'PRVC/SIMV', 'PSV/SBT', 'SIMV', 'SIMV/AutoFlow', 'SIMV/PRES',
         'SIMV/PSV', 'SIMV/PSV/AutoFlow', 'SIMV/VOL', 'SYNCHRON MASTER', 'SYNCHRON SLAVE', 'VOL/AC']
HAMILTON = ['APRV', 'APV (cmv)', 'Ambient', '(S) CMV', 'P-CMV', 'SIMV', 'APV (simv)', 'P-SIMV', 'VS', 'ASV', 'DuoPaP', 'NIV', 'NIV-ST']
print('device values', dev.v.value_counts().head(24).to_dict(), flush=True)
sup = pd.concat([dev[['stay_id', 'charttime']].assign(s=dev.v.isin(ADV)),
                 chart[((chart.itemid == 223849) & chart.value.isin(MODES)) | ((chart.itemid == 229314) & chart.value.isin(HAMILTON))][['stay_id', 'charttime']].assign(s=True)])
sup = sup.groupby(['stay_id', 'charttime'], as_index=False).s.max().sort_values('charttime')
ra = dev[dev.v == 'none'][['stay_id', 'charttime']].sort_values('charttime'); ra['ra'] = True


def with_support(df):
    # ventilation.sql: the last charted status holds for up to 14h. Its end rule reads the next row, which is future data here.
    return pd.merge_asof(df.sort_values('charttime'), sup, on='charttime', by='stay_id', direction='backward', tolerance=14 * HOUR).assign(s=lambda x: x.s.eq(True))


abg = with_support(abg.dropna(subset=['f']))
abg['pf'] = abg.pao2 / abg.f
abg = abg[abg.pf.between(5, 1000)]
abg['s2'] = np.select([(abg.pf <= 75) & abg.s, (abg.pf <= 150) & abg.s, abg.pf <= 225, abg.pf <= 300], [4, 3, 2, 1], 0)
abg['s1'] = np.select([(abg.pf < 100) & abg.s, (abg.pf < 200) & abg.s, abg.pf < 300, abg.pf < 400], [4, 3, 2, 1], 0)
ab2 = abg.groupby(['stay_id', 'day']).s2.max().astype(float)
ab1 = abg.groupby(['stay_id', 'day']).s1.max().astype(float)
sp = chart[(chart.itemid == 220277) & chart.valuenum.between(50, 100)][['stay_id', 'day', 'charttime', 'valuenum']].rename(columns={'valuenum': 'spo2'}).sort_values('charttime')
sp = pd.merge_asof(sp, cf[['stay_id', 'charttime', 'fio2']], on='charttime', by='stay_id', direction='backward', tolerance=4 * HOUR)
sp = pd.merge_asof(sp, ra, on='charttime', by='stay_id', direction='backward', tolerance=4 * HOUR)
sp['f'] = sp.fio2.where(sp.fio2.notna(), pd.Series(np.where(sp.ra.eq(True), .21, np.nan), index=sp.index))
sp = with_support(sp.dropna(subset=['f']))
sp = sp[sp.spo2 < 98]  # SOFA-2 note f
sp['sf'] = sp.spo2 / sp.f
sp['s2'] = np.select([(sp.sf <= 120) & sp.s, (sp.sf <= 200) & sp.s, sp.sf <= 250, sp.sf <= 300], [4, 3, 2, 1], 0)
sf2 = sp.groupby(['stay_id', 'day']).s2.max().astype(float)
sp['s1'] = np.select([(sp.sf <= 89) & sp.s, (sp.sf <= 214) & sp.s, sp.sf <= 357, sp.sf <= 512], [4, 3, 2, 1], 0)  # Pandharipande 2009
sf1 = sp.groupby(['stay_id', 'day']).s1.max().astype(float)
r2 = ab2.combine_first(sf2)
for k in ecmo.index:
    r2.loc[k] = 4.0
col('resp_s2', r2); col('resp_s2_abg', ab2); col('resp_s1', ab1.combine_first(sf1)); col('resp_fresh', pd.Series(True, r2.index))
col('resp_from_abg', pd.Series(True, ab2.index)); col('resp_from_sf', pd.Series(True, sf2.index))
print('resp done', len(ab2), len(sf2), flush=True)

# ---------- brain ----------
g = chart[chart.itemid.isin([220739, 223900, 223901])].copy()
ett = g[(g.itemid == 223900) & g.value.fillna('').str.contains('ET/Trach|ETT', case=False, regex=True)][['stay_id', 'charttime']].drop_duplicates().assign(ett=True)
p = g.pivot_table(index=['stay_id', 'charttime', 'day'], columns='itemid', values='valuenum', aggfunc='min').reset_index()
p = p.rename(columns={220739: 'e', 223900: 'v', 223901: 'm'}).merge(ett, how='left', on=['stay_id', 'charttime'])
p['ett'] = p.ett.eq(True)


def sedated(r):
    for a, b in sed_iv.get(r.stay_id, ()):
        if a <= r.charttime <= b:
            return a
    return pd.NaT


p['sed_start'] = [sedated(r) for r in p.itertuples()]
p['sed'] = p.sed_start.notna()
tot = p.e + p.v + p.m
ok = ~p.sed & ~p.ett & tot.notna()
mo = ~p.sed & ~ok & p.m.notna()
p['s2'] = np.nan
p.loc[ok, 's2'] = np.select([tot[ok] <= 5, tot[ok] <= 8, tot[ok] <= 12, tot[ok] <= 14], [4, 3, 2, 1], 0)
p.loc[mo, 's2'] = p.m[mo].map({6: 0, 5: 1, 4: 2, 3: 3, 2: 4, 1: 4})
p['s1'] = np.nan
p.loc[ok, 's1'] = np.select([tot[ok] < 6, tot[ok] <= 9, tot[ok] <= 12, tot[ok] <= 14], [4, 3, 2, 1], 0)
p.loc[mo, 's1'] = p.m[mo].map({6: 0, 5: 1, 4: 2, 3: 3, 2: 4, 1: 4})
fresh = p[p.s2.notna()]
b2 = fresh.groupby(['stay_id', 'day']).s2.max()
b1 = p[p.s1.notna()].groupby(['stay_id', 'day']).s1.max().astype(float)
pres, pres1 = {}, {}
for stay, x in p.groupby('stay_id'):
    x = x.sort_values('charttime'); val = val1 = None
    for r in x.itertuples():
        if not r.sed and np.isfinite(r.s2):
            val, val1 = r.s2, r.s1
        elif r.sed:
            k = (stay, r.day)
            pres[k] = max(pres.get(k, 0.), val if val is not None else 0.)  # note c: unknown prior GCS scores 0
            pres1[k] = max(pres1.get(k, 0.), val1 if val1 is not None else 0.)
pres = pd.Series(pres, dtype=float); pres.index = pd.MultiIndex.from_tuples(pres.index, names=['stay_id', 'day'])
pres1 = pd.Series(pres1, dtype=float); pres1.index = pres.index
hal = inp[(inp.itemid == 221824)][['stay_id', 'starttime', 'storetime']].rename(columns={'starttime': 'charttime'})
em = pd.read_pickle(R + 'raw/emar.pkl'); em = em[em.event_txt.fillna('').str.contains('Administered|Confirmed', case=False, regex=True)]
em = to_stay(em)[['stay_id', 'charttime', 'storetime']]
dl = add_day(pd.concat([hal, em])).groupby(['stay_id', 'day']).size()
dl = pd.Series(1.0, dl.index)
b_all = b2.combine_first(pres)
b_all = pd.concat([b_all, dl], axis=1).max(axis=1)
col('brain_s2', b_all)
col('brain_s1', b1.combine_first(pres1))
col('brain_fresh', pd.Series(True, b2.index.union(dl.index)))
col('brain_presed', pd.Series(True, pres.index))
col('delirium_drug', dl)
print('brain done', len(b2), len(pres), len(dl), flush=True)

# ---------- kidney ----------
ur = pd.read_pickle(R + 'raw/urine.pkl'); ur['value'] = pd.to_numeric(ur.value, errors='coerce')
ur.loc[ur.itemid == 227488, 'value'] = -ur.value[ur.itemid == 227488].abs()
ur = add_day(ur.dropna(subset=['value']))
wt = chart[chart.itemid == 226512].groupby('stay_id').valuenum.first()
wt2 = chart[chart.itemid == 224639].groupby('stay_id').valuenum.median()
wt3 = inp.groupby('stay_id').patientweight.median()
wgt = wt.combine_first(wt2).combine_first(wt3); wgt = wgt.where(wgt.between(30, 300))
print('weight known', wgt.notna().sum(), flush=True)
urows = []
for stay, x in ur.groupby('stay_id'):
    w = wgt.get(stay, np.nan)
    h = x.groupby('hour').value.sum()
    if not np.isfinite(w) or len(h) < 2:
        continue
    n = int(h.index.max()) + 1
    v = np.zeros(n); v[h.index] = h.values
    obs = np.zeros(n, bool); obs[h.index] = True
    best = {}
    for tau in h.index:
        s = 0
        for K, r_, pts in ((6, .5, 1), (12, .5, 2), (24, .3, 3)):
            a = tau - K + 1
            if a < 0 or not obs[:a + 1].any():
                continue
            o = np.flatnonzero(obs[a:tau + 1])
            if len(o) < 2 or np.diff(o).max() > 6 or o[0] > 5:
                continue
            total = v[a:tau + 1].sum()
            if total / (K * w) < r_:
                s = max(s, pts)
            if K == 12 and total <= 0:
                s = max(s, 3)
        d = tau // 24
        best[d] = max(best.get(d, 0), s)
    for d, s in best.items():
        urows.append((stay, d, float(s)))
uo = pd.DataFrame(urows, columns=['stay_id', 'day', 'u']).set_index(['stay_id', 'day']).u
uvol = ur.groupby(['stay_id', 'day']).value.agg(['sum', 'count'])
cr = lab[(lab.itemid == 50912) & (lab.valuenum > 0)].groupby(['stay_id', 'day']).valuenum.max()
c2 = pd.Series(np.select([cr > 3.5, cr > 2, cr > 1.2], [3, 2, 1], 0), cr.index).astype(float)
c1 = pd.Series(np.select([cr >= 5, cr >= 3.5, cr >= 2, cr >= 1.2], [4, 3, 2, 1], 0), cr.index).astype(float)
pr = pd.read_pickle(R + 'raw/proc.pkl'); pr = pr[pr.stay_id.isin(STAYS) & pr.value.notna()].copy()
for c_ in ('starttime', 'endtime'):
    pr[c_] = pd.to_datetime(pr[c_])
rrt, rrt_exact = set(), set()


def mark(stay, a, b):
    i0 = IN[stay]; n = st.ndays[stay]
    d0 = max(0, int((a - i0) / DAY)); d1 = min(n - 1, int((b + 72 * HOUR - i0) / DAY))  # own approximation of note q: off-RRT days stay 4 for 72h
    for d in range(d0, d1 + 1):
        rrt.add((stay, d))
    for d in range(d0, min(st.ndays[stay] - 1, int((b - i0) / DAY)) + 1):
        rrt_exact.add((stay, d))


for r in pr.dropna(subset=['starttime', 'endtime']).itertuples():
    mark(r.stay_id, r.starttime, r.endtime)
ACTIVE = [226499, 224154, 225183, 227438, 224191, 225806, 225807, 228004, 228005, 228006, 224144, 224145, 224153, 226457]  # rrt.sql
for r in chart[(chart.itemid.isin(ACTIVE) & (chart.value != '')) | ((chart.itemid == 225965) & (chart.value == 'In use'))].itertuples():
    mark(r.stay_id, r.charttime, r.charttime)
for r in crrt_drugs.dropna(subset=['starttime', 'endtime']).itertuples():
    mark(r.stay_id, pd.Timestamp(r.starttime), pd.Timestamp(r.endtime))
chron = chart[(chart.itemid == 225126) & (chart.valuenum == 1)].stay_id.unique()
print('chronic dialysis stays', len(chron), 'dialysis item values', chart[chart.itemid == 225126].value.value_counts().to_dict(), flush=True)
for stay in chron:
    if stay not in STAYS:
        continue
    for d in range(st.ndays[stay]):
        rrt.add((stay, d))
        rrt_exact.add((stay, d))
k = lab[lab.itemid.isin([50971, 50822])].groupby(['stay_id', 'day']).valuenum.max()
ph = lab[lab.itemid == 50820].groupby(['stay_id', 'day']).valuenum.min()
hc = lab[lab.itemid == 50882].groupby(['stay_id', 'day']).valuenum.min()
crit = ((k >= 6).reindex(cr.index, fill_value=False) | ((ph <= 7.2) & (hc <= 12)).reindex(cr.index, fill_value=False)) & (cr > 1.2)
col('kidney_s2_creatinine', c2)
col('kidney_s2_no_rrt', pd.concat([c2, uo], axis=1).max(axis=1))
kid = pd.concat([c2, uo], axis=1).max(axis=1)
rr_idx = pd.MultiIndex.from_tuples(sorted(rrt), names=['stay_id', 'day'])
kid = kid.reindex(kid.index.union(rr_idx))
kid.loc[rr_idx] = 4.0
kid.loc[crit[crit].index.intersection(kid.index)] = 4.0
col('kidney_s2', kid)
col('kidney_fresh', pd.Series(True, kid.index))
col('kidney_rrt', pd.Series(True, rr_idx))
col('kidney_rrt_exact', pd.Series(True, pd.MultiIndex.from_tuples(sorted(rrt_exact), names=['stay_id', 'day'])))
ks1 = c1.copy()
ok_u = uvol[uvol['count'] >= 6]['sum']
ku = pd.Series(np.select([ok_u < 200, ok_u < 500], [4, 3], 0), ok_u.index).astype(float)
col('kidney_s1', pd.concat([ks1, ku], axis=1).max(axis=1))
print('kidney done', len(kid), flush=True)

# ---------- assemble ----------
idx = pd.MultiIndex.from_tuples([(s, d) for s, n in st.ndays.items() for d in range(n)], names=['stay_id', 'day'])
T = pd.DataFrame(index=idx)
for c_, s_ in day.items():
    T[c_] = s_.reindex(idx)
for o in ORG:
    T[o + '_fresh'] = T[o + '_fresh'].eq(True)
T = T.reset_index()
T['subject_id'] = T.stay_id.map(st.subject_id)
T['start'] = T.stay_id.map(IN) + T.day * DAY
T['end'] = T.start + DAY
T['outtime'] = T.stay_id.map(OUT)
T['deathtime'] = T.stay_id.map(st.deathtime)
T['full_day'] = T.end <= T.outtime
T.to_pickle(R + 'daily.pkl')
print('saved', T.shape, flush=True)

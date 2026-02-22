import os
from itertools import combinations

import numpy as np
import pandas as pd


csv_path = '/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP/results/2026-02-11_CASF2016_Docking-Comparison_All-Results-W-Scores.csv'
out_dir = '/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP/results/figures/FINAL'

models = ['dynaformer', 'egna', 'ehign_pla', 'gign', 'onionnet-2']
source_order = ['Crystal', 'GNINA-Crystal', 'GNINA-Apo', 'GNINA-AF3', 'AlphaFold3']

seed = 42
n_boot = 20000
alpha = 0.05

pose_max_gnina_crystal = 1
pose_max_gnina_apo = 1
pose_max_gnina_af3 = 1

min_models_per_pdbid = 1

perf_csv = f'{out_dir}/performance_by_source_modelavg.csv'
pairwise_csv = f'{out_dir}/pairwise_pvalues_sources_only_modelavg.csv'


def pcc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def holm_bonferroni(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]

    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for i in range(m):
        k = m - i
        val = ranked[i] * k
        running_max = max(running_max, val)
        adj[i] = min(running_max, 1.0)

    out = np.empty(m, dtype=float)
    out[order] = adj
    return out


def paired_bootstrap_delta(y_true, pred_a, pred_b, rng, n_boot=20000, alpha=0.05):
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = y_true.size

    point_a = pcc(y_true, pred_a)
    point_b = pcc(y_true, pred_b)
    point_delta = point_a - point_b

    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a_i = pcc(y_true[idx], pred_a[idx])
        b_i = pcc(y_true[idx], pred_b[idx])
        d_i = a_i - b_i
        if np.isfinite(d_i):
            deltas.append(d_i)

    deltas = np.asarray(deltas, dtype=float)
    if deltas.size == 0:
        return point_delta, np.nan, np.nan, np.nan

    ci_lo = float(np.quantile(deltas, alpha / 2))
    ci_hi = float(np.quantile(deltas, 1 - alpha / 2))

    B = deltas.size
    count_le = int(np.sum(deltas <= 0))
    count_ge = int(np.sum(deltas >= 0))
    p_left = (count_le + 1) / (B + 1)
    p_right = (count_ge + 1) / (B + 1)
    p_two = 2 * min(p_left, p_right)
    p_two = min(p_two, 1.0)

    return point_delta, ci_lo, ci_hi, float(p_two)


os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df['pdbid'] = df['pdbid'].astype(str).str.lower().str.strip()
df['Model'] = df['Model'].astype(str).str.lower().str.strip()
df['Source'] = df['Source'].astype(str).str.lower().str.strip()
df['pose'] = df['pose'].astype(int)

rows = []

d = df[(df['Source'].str.contains('crystal', na=False) & ~df['Source'].str.contains('gnina', na=False)) & df['Model'].isin(models) & (df['pose'] == 1)].copy()
g = d.groupby(['Model', 'pdbid'])[['pK_true', 'pK_predicted']].agg({'pK_true': 'first', 'pK_predicted': 'first'}).reset_index()
g['source_group'] = 'Crystal'
rows.append(g)

d = df[(df['Source'].str.contains('alpha', na=False) & ~df['Source'].str.contains('gnina', na=False)) & df['Model'].isin(models) & (df['pose'] == 1)].copy()
g = d.groupby(['Model', 'pdbid'])[['pK_true', 'pK_predicted']].agg({'pK_true': 'first', 'pK_predicted': 'first'}).reset_index()
g['source_group'] = 'AlphaFold3'
rows.append(g)

d = df[(df['Source'].str.contains('gnina', na=False) & df['Source'].str.contains('af3', na=False)) & df['Model'].isin(models) & (df['pose'] <= pose_max_gnina_af3)].copy()
g = d.groupby(['Model', 'pdbid'])[['pK_true', 'pK_predicted']].agg({'pK_true': 'first', 'pK_predicted': 'mean'}).reset_index()
g['source_group'] = 'GNINA-AF3'
rows.append(g)

d = df[(df['Source'].str.contains('gnina', na=False) & df['Source'].str.contains('apo', na=False) & ~df['Source'].str.contains('af3', na=False)) & df['Model'].isin(models) & (df['pose'] <= pose_max_gnina_apo)].copy()
g = d.groupby(['Model', 'pdbid'])[['pK_true', 'pK_predicted']].agg({'pK_true': 'first', 'pK_predicted': 'mean'}).reset_index()
g['source_group'] = 'GNINA-Apo'
rows.append(g)

d = df[(df['Source'].str.contains('gnina', na=False) & ~df['Source'].str.contains('af3', na=False) & ~df['Source'].str.contains('apo', na=False)) & df['Model'].isin(models) & (df['pose'] <= pose_max_gnina_crystal)].copy()
g = d.groupby(['Model', 'pdbid'])[['pK_true', 'pK_predicted']].agg({'pK_true': 'first', 'pK_predicted': 'mean'}).reset_index()
g['source_group'] = 'GNINA-Crystal'
rows.append(g)

agg = pd.concat(rows, ignore_index=True)

ens = agg.groupby(['pdbid', 'source_group']).agg(
    pK_true=('pK_true', 'first'),
    pK_predicted=('pK_predicted', 'mean'),
    n_models=('Model', 'nunique')
).reset_index()

ens = ens[ens['n_models'] >= min_models_per_pdbid].copy()

wide = ens.pivot_table(index='pdbid', columns='source_group', values='pK_predicted', aggfunc='first')
y_true = ens.groupby('pdbid')['pK_true'].first()

perf_rows = []
for src in source_order:
    pccs = []
    ns = []

    for m in models:
        tmp = agg[(agg['source_group'] == src) & (agg['Model'] == m)][['pdbid', 'pK_true', 'pK_predicted']].dropna()
        if tmp.empty:
            continue

        tmp = tmp.groupby('pdbid', as_index=False).agg(
            pK_true=('pK_true', 'first'),
            pred=('pK_predicted', 'mean')
        ).dropna()

        if len(tmp) < 10:
            continue

        pccs.append(pcc(tmp['pK_true'].to_numpy(), tmp['pred'].to_numpy()))
        ns.append(len(tmp))

    if len(pccs) == 0:
        continue

    perf_rows.append({
        'source_group': src,
        'n': int(np.min(ns)),
        'pcc': float(np.mean(pccs)),
    })

perf = pd.DataFrame(perf_rows)
perf.to_csv(perf_csv, index=False)

rng = np.random.default_rng(seed)

pair_rows = []
sources_present = [s for s in source_order if s in wide.columns]

for a, b in combinations(sources_present, 2):
    tmp = pd.DataFrame({
        'pK_true': y_true,
        'pred_a': wide[a],
        'pred_b': wide[b],
    }).dropna()

    if len(tmp) < 10:
        continue

    d, lo, hi, p = paired_bootstrap_delta(
        tmp['pK_true'].to_numpy(),
        tmp['pred_a'].to_numpy(),
        tmp['pred_b'].to_numpy(),
        rng,
        n_boot=n_boot,
        alpha=alpha
    )

    pair_rows.append({
        'metric': 'pcc',
        'A_source': a,
        'B_source': b,
        'delta_A_better': d,
        'ci_lo': lo,
        'ci_hi': hi,
        'p_uncorrected': p,
        'n': int(len(tmp)),
    })

pairwise = pd.DataFrame(pair_rows)

pairwise['p_holm'] = holm_bonferroni(pairwise['p_uncorrected'].to_numpy())
pairwise.to_csv(pairwise_csv, index=False)

pd.set_option('display.float_format', lambda x: f'{x:.3e}')
print('Wrote:', perf_csv)
print('Wrote:', pairwise_csv)
print(pairwise.sort_values('p_holm').head(20))

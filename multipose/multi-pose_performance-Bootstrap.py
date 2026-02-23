import os
from itertools import combinations

import numpy as np
import pandas as pd


# -----------------------
# Config
# -----------------------
csv_path = '/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP/results/2026-02-11_CASF2016_Docking-Comparison_All-Results-W-Scores.csv'
out_dir = '/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP/results/figures/FINAL/multipose'

k_list = [1, 2, 3, 5, 10, 20, 30, 50, 100]
models = ['dynaformer', 'egna', 'ehign_pla', 'gign', 'onionnet-2']
sources = ['gnina-crystal', 'gnina-apo', 'gnina-af3', 'alphafold3']
methods = ['mean', 'median']

n_boot = 20000
seed = 42
ci_alpha = 0.05

compare_k_list = k_list  # set to [1, 100] if you only want endpoints

os.makedirs(out_dir, exist_ok=True)


# -----------------------
# Load + cleanup
# -----------------------
df = pd.read_csv(csv_path)

df['pdbid'] = df['pdbid'].astype(str).str.lower().str.strip()
df['Model'] = df['Model'].astype(str).str.lower().str.strip()
df['Source'] = df['Source'].astype(str).str.lower().str.strip()
df['pose'] = pd.to_numeric(df['pose'], errors='coerce')


# -----------------------
# Metrics
# -----------------------
def safe_pcc_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        return np.nan, np.nan, 0

    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        pcc_val = np.nan
    else:
        pcc_val = float(np.corrcoef(y_true, y_pred)[0, 1])

    return pcc_val, rmse_val, int(y_true.size)


def bootstrap_ci_both(y_true, y_pred, n_boot=20000, seed=42, ci_alpha=0.05):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    n = y_true.size
    if n == 0:
        return {
            'n_targets': 0,
            'pcc': np.nan,
            'pcc_ci_low': np.nan,
            'pcc_ci_high': np.nan,
            'rmse': np.nan,
            'rmse_ci_low': np.nan,
            'rmse_ci_high': np.nan,
            'n_boot_pcc': 0,
            'n_boot_rmse': 0,
        }

    rng = np.random.default_rng(seed)
    pcc_vals = []
    rmse_vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]

        rmse_vals.append(float(np.sqrt(np.mean((yt - yp) ** 2))))

        if yt.size >= 2 and np.std(yt) != 0.0 and np.std(yp) != 0.0:
            v = float(np.corrcoef(yt, yp)[0, 1])
            if np.isfinite(v):
                pcc_vals.append(v)

    if len(rmse_vals) == 0:
        rmse_mid = np.nan
        rmse_lo = np.nan
        rmse_hi = np.nan
    else:
        rmse_arr = np.asarray(rmse_vals, dtype=float)
        rmse_mid = float(np.median(rmse_arr))
        rmse_lo = float(np.quantile(rmse_arr, ci_alpha / 2.0))
        rmse_hi = float(np.quantile(rmse_arr, 1.0 - ci_alpha / 2.0))

    if len(pcc_vals) == 0:
        pcc_mid = np.nan
        pcc_lo = np.nan
        pcc_hi = np.nan
    else:
        pcc_arr = np.asarray(pcc_vals, dtype=float)
        pcc_mid = float(np.median(pcc_arr))
        pcc_lo = float(np.quantile(pcc_arr, ci_alpha / 2.0))
        pcc_hi = float(np.quantile(pcc_arr, 1.0 - ci_alpha / 2.0))

    pcc_pt, rmse_pt, _ = safe_pcc_rmse(y_true, y_pred)

    return {
        'n_targets': int(n),
        'pcc': pcc_pt,
        'pcc_ci_low': pcc_lo,
        'pcc_ci_high': pcc_hi,
        'rmse': rmse_pt,
        'rmse_ci_low': rmse_lo,
        'rmse_ci_high': rmse_hi,
        'n_boot_pcc': int(len(pcc_vals)),
        'n_boot_rmse': int(len(rmse_vals)),
    }


def paired_bootstrap_delta(y_true, y_pred_a, y_pred_b, metric, n_boot=20000, seed=42, ci_alpha=0.05):
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_a, dtype=float)
    y_pred_b = np.asarray(y_pred_b, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred_a) & np.isfinite(y_pred_b)
    y_true = y_true[mask]
    y_pred_a = y_pred_a[mask]
    y_pred_b = y_pred_b[mask]

    n = y_true.size
    if n == 0:
        return {
            'n_targets': 0,
            'delta': np.nan,
            'ci_low': np.nan,
            'ci_high': np.nan,
            'p_two_sided': np.nan,
            'n_boot': 0,
        }

    rng = np.random.default_rng(seed)
    deltas = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ya = y_pred_a[idx]
        yb = y_pred_b[idx]

        if metric == 'rmse':
            ra = float(np.sqrt(np.mean((yt - ya) ** 2)))
            rb = float(np.sqrt(np.mean((yt - yb) ** 2)))
            deltas.append(ra - rb)
        elif metric == 'pcc':
            if yt.size < 2:
                continue
            if np.std(yt) == 0.0 or np.std(ya) == 0.0 or np.std(yb) == 0.0:
                continue
            ca = float(np.corrcoef(yt, ya)[0, 1])
            cb = float(np.corrcoef(yt, yb)[0, 1])
            if np.isfinite(ca) and np.isfinite(cb):
                deltas.append(ca - cb)
        else:
            raise ValueError(f'Unknown metric: {metric}')

    if len(deltas) == 0:
        return {
            'n_targets': int(n),
            'delta': np.nan,
            'ci_low': np.nan,
            'ci_high': np.nan,
            'p_two_sided': np.nan,
            'n_boot': 0,
        }

    deltas = np.asarray(deltas, dtype=float)
    mid = float(np.median(deltas))
    lo = float(np.quantile(deltas, ci_alpha / 2.0))
    hi = float(np.quantile(deltas, 1.0 - ci_alpha / 2.0))
    p_two_sided = float(2.0 * min(np.mean(deltas <= 0.0), np.mean(deltas >= 0.0)))

    return {
        'n_targets': int(n),
        'delta': mid,
        'ci_low': lo,
        'ci_high': hi,
        'p_two_sided': p_two_sided,
        'n_boot': int(len(deltas)),
    }


# -----------------------
# Aggregation helper
# -----------------------
def make_agg(df, source, model, n_poses, method):
    d = df[(df['Source'].str.contains(source, na=False)) & (df['Model'] == model)].copy()

    d = d.dropna(subset=['pose'])
    if d.empty:
        return pd.DataFrame(columns=['pdbid', 'pK_true', 'pK_predicted'])

    d['pose'] = d['pose'].astype(int)
    d = d.sort_values(['pdbid', 'pose'])

    dk = d[d['pose'] <= int(n_poses)].copy()
    dk = dk.dropna(subset=['pK_true', 'pK_predicted'])
    if dk.empty:
        return pd.DataFrame(columns=['pdbid', 'pK_true', 'pK_predicted'])

    pred_agg = 'mean' if method == 'mean' else 'median'
    agg = dk.groupby('pdbid', as_index=False).agg(
        pK_true=('pK_true', 'first'),
        pK_predicted=('pK_predicted', pred_agg),
    )

    agg = agg.dropna(subset=['pK_true', 'pK_predicted'])
    return agg


# -----------------------
# (b) Keep models separate: per-point metrics + bootstrap CIs
# -----------------------
rows_model = []
for method in methods:
    for source in sources:
        for model in models:
            for n_poses in k_list:
                agg = make_agg(df, source, model, n_poses, method)

                stats = bootstrap_ci_both(
                    agg['pK_true'].to_numpy(),
                    agg['pK_predicted'].to_numpy(),
                    n_boot=n_boot,
                    seed=seed,
                    ci_alpha=ci_alpha,
                )

                rows_model.append({
                    'method': method,
                    'source': source,
                    'model': model,
                    'n_poses': int(n_poses),
                    **stats,
                })

res_model = pd.DataFrame(rows_model)
res_model.to_csv(f'{out_dir}/multipose_summary_by_model_bootstrap.csv', index=False)


# -----------------------
# (a) Average across models: point metrics + bootstrap CIs
# The average is computed as the mean of per-model metrics on a common pdbid set.
# -----------------------
rows_avg = []
for method in methods:
    for source in sources:
        for n_poses in k_list:
            aggs = {}
            pdb_sets = []

            for model in models:
                agg = make_agg(df, source, model, n_poses, method)
                aggs[model] = agg
                pdb_sets.append(set(agg['pdbid'].tolist()))

            if len(pdb_sets) == 0:
                continue

            common = set.intersection(*pdb_sets) if all(len(s) > 0 for s in pdb_sets) else set()
            if len(common) == 0:
                rows_avg.append({
                    'method': method,
                    'source': source,
                    'n_poses': int(n_poses),
                    'n_targets': 0,
                    'pcc_mean_over_models': np.nan,
                    'pcc_ci_low': np.nan,
                    'pcc_ci_high': np.nan,
                    'rmse_mean_over_models': np.nan,
                    'rmse_ci_low': np.nan,
                    'rmse_ci_high': np.nan,
                    'n_boot_pcc': 0,
                    'n_boot_rmse': 0,
                })
                continue

            common = sorted(list(common))
            per_model_true = {}
            per_model_pred = {}

            for model in models:
                m = aggs[model].set_index('pdbid').loc[common]
                per_model_true[model] = m['pK_true'].to_numpy(dtype=float)
                per_model_pred[model] = m['pK_predicted'].to_numpy(dtype=float)

            pcc_list = []
            rmse_list = []
            for model in models:
                pcc_val, rmse_val, _ = safe_pcc_rmse(per_model_true[model], per_model_pred[model])
                pcc_list.append(pcc_val)
                rmse_list.append(rmse_val)

            pcc_point = float(np.nanmean(pcc_list)) if np.any(np.isfinite(pcc_list)) else np.nan
            rmse_point = float(np.nanmean(rmse_list)) if np.any(np.isfinite(rmse_list)) else np.nan

            rng = np.random.default_rng(seed)
            pcc_boot = []
            rmse_boot = []

            n = len(common)
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)

                pcc_b = []
                rmse_b = []
                for model in models:
                    yt = per_model_true[model][idx]
                    yp = per_model_pred[model][idx]

                    rmse_b.append(float(np.sqrt(np.mean((yt - yp) ** 2))))

                    if yt.size >= 2 and np.std(yt) != 0.0 and np.std(yp) != 0.0:
                        v = float(np.corrcoef(yt, yp)[0, 1])
                        if np.isfinite(v):
                            pcc_b.append(v)

                rmse_boot.append(float(np.nanmean(rmse_b)) if np.any(np.isfinite(rmse_b)) else np.nan)
                pcc_boot.append(float(np.nanmean(pcc_b)) if np.any(np.isfinite(pcc_b)) else np.nan)

            rmse_boot = np.asarray([v for v in rmse_boot if np.isfinite(v)], dtype=float)
            pcc_boot = np.asarray([v for v in pcc_boot if np.isfinite(v)], dtype=float)

            if rmse_boot.size == 0:
                rmse_lo = np.nan
                rmse_hi = np.nan
                n_boot_rmse = 0
            else:
                rmse_lo = float(np.quantile(rmse_boot, ci_alpha / 2.0))
                rmse_hi = float(np.quantile(rmse_boot, 1.0 - ci_alpha / 2.0))
                n_boot_rmse = int(rmse_boot.size)

            if pcc_boot.size == 0:
                pcc_lo = np.nan
                pcc_hi = np.nan
                n_boot_pcc = 0
            else:
                pcc_lo = float(np.quantile(pcc_boot, ci_alpha / 2.0))
                pcc_hi = float(np.quantile(pcc_boot, 1.0 - ci_alpha / 2.0))
                n_boot_pcc = int(pcc_boot.size)

            rows_avg.append({
                'method': method,
                'source': source,
                'n_poses': int(n_poses),
                'n_targets': int(len(common)),
                'pcc_mean_over_models': pcc_point,
                'pcc_ci_low': pcc_lo,
                'pcc_ci_high': pcc_hi,
                'rmse_mean_over_models': rmse_point,
                'rmse_ci_low': rmse_lo,
                'rmse_ci_high': rmse_hi,
                'n_boot_pcc': n_boot_pcc,
                'n_boot_rmse': n_boot_rmse,
            })

res_avg = pd.DataFrame(rows_avg)
res_avg.to_csv(f'{out_dir}/multipose_summary_modelavg_bootstrap.csv', index=False)


# -----------------------
# Pairwise comparisons (b): sources compared within each model
# Delta is A minus B. For PCC positive favors A. For RMSE negative favors A.
# -----------------------
pair_rows_model = []
for method in methods:
    for model in models:
        for n_poses in compare_k_list:
            aggs = {}
            for source in sources:
                aggs[source] = make_agg(df, source, model, n_poses, method)

            for a, b in combinations(sources, 2):
                da = aggs[a]
                db = aggs[b]

                if da.empty or db.empty:
                    continue

                m = da.merge(db, on='pdbid', suffixes=('_a', '_b'))
                if m.empty:
                    continue

                dpcc = paired_bootstrap_delta(
                    m['pK_true_a'].to_numpy(),
                    m['pK_predicted_a'].to_numpy(),
                    m['pK_predicted_b'].to_numpy(),
                    metric='pcc',
                    n_boot=n_boot,
                    seed=seed,
                    ci_alpha=ci_alpha,
                )

                drmse = paired_bootstrap_delta(
                    m['pK_true_a'].to_numpy(),
                    m['pK_predicted_a'].to_numpy(),
                    m['pK_predicted_b'].to_numpy(),
                    metric='rmse',
                    n_boot=n_boot,
                    seed=seed,
                    ci_alpha=ci_alpha,
                )

                pair_rows_model.append({
                    'method': method,
                    'model': model,
                    'n_poses': int(n_poses),
                    'source_a': a,
                    'source_b': b,
                    'n_targets': int(len(m)),
                    'delta_pcc_a_minus_b': dpcc['delta'],
                    'delta_pcc_ci_low': dpcc['ci_low'],
                    'delta_pcc_ci_high': dpcc['ci_high'],
                    'delta_pcc_p_two_sided': dpcc['p_two_sided'],
                    'n_boot_pcc': dpcc['n_boot'],
                    'delta_rmse_a_minus_b': drmse['delta'],
                    'delta_rmse_ci_low': drmse['ci_low'],
                    'delta_rmse_ci_high': drmse['ci_high'],
                    'delta_rmse_p_two_sided': drmse['p_two_sided'],
                    'n_boot_rmse': drmse['n_boot'],
                })

pairwise_model = pd.DataFrame(pair_rows_model)
pairwise_model.to_csv(f'{out_dir}/multipose_pairwise_sources_by_model.csv', index=False)


# -----------------------
# Pairwise comparisons (a): sources compared after averaging over models
# The comparison is done on a common pdbid set shared across all models and both sources.
# Delta is A minus B on the averaged metric.
# -----------------------
pair_rows_avg = []
for method in methods:
    for n_poses in compare_k_list:
        for a, b in combinations(sources, 2):
            aggs_a = {}
            aggs_b = {}
            pdb_sets = []

            for model in models:
                aa = make_agg(df, a, model, n_poses, method)
                bb = make_agg(df, b, model, n_poses, method)

                aggs_a[model] = aa
                aggs_b[model] = bb

                pdb_a = set(aa['pdbid'].tolist())
                pdb_b = set(bb['pdbid'].tolist())
                pdb_sets.append(pdb_a.intersection(pdb_b))

            common = set.intersection(*pdb_sets) if all(len(s) > 0 for s in pdb_sets) else set()
            if len(common) == 0:
                pair_rows_avg.append({
                    'method': method,
                    'n_poses': int(n_poses),
                    'source_a': a,
                    'source_b': b,
                    'n_targets': 0,
                    'delta_pcc_a_minus_b': np.nan,
                    'delta_pcc_ci_low': np.nan,
                    'delta_pcc_ci_high': np.nan,
                    'delta_pcc_p_two_sided': np.nan,
                    'n_boot_pcc': 0,
                    'delta_rmse_a_minus_b': np.nan,
                    'delta_rmse_ci_low': np.nan,
                    'delta_rmse_ci_high': np.nan,
                    'delta_rmse_p_two_sided': np.nan,
                    'n_boot_rmse': 0,
                })
                continue

            common = sorted(list(common))
            per_model_true = {}
            per_model_pred_a = {}
            per_model_pred_b = {}

            for model in models:
                ma = aggs_a[model].set_index('pdbid').loc[common]
                mb = aggs_b[model].set_index('pdbid').loc[common]

                per_model_true[model] = ma['pK_true'].to_numpy(dtype=float)
                per_model_pred_a[model] = ma['pK_predicted'].to_numpy(dtype=float)
                per_model_pred_b[model] = mb['pK_predicted'].to_numpy(dtype=float)

            pcc_a = []
            pcc_b = []
            rmse_a = []
            rmse_b = []

            for model in models:
                pa, ra, _ = safe_pcc_rmse(per_model_true[model], per_model_pred_a[model])
                pb, rb, _ = safe_pcc_rmse(per_model_true[model], per_model_pred_b[model])
                pcc_a.append(pa)
                pcc_b.append(pb)
                rmse_a.append(ra)
                rmse_b.append(rb)

            pcc_point = float(np.nanmean(pcc_a) - np.nanmean(pcc_b)) if np.any(np.isfinite(pcc_a)) and np.any(np.isfinite(pcc_b)) else np.nan
            rmse_point = float(np.nanmean(rmse_a) - np.nanmean(rmse_b)) if np.any(np.isfinite(rmse_a)) and np.any(np.isfinite(rmse_b)) else np.nan

            rng = np.random.default_rng(seed)
            pcc_deltas = []
            rmse_deltas = []

            n = len(common)
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)

                pcc_a_b = []
                pcc_b_b = []
                rmse_a_b = []
                rmse_b_b = []

                for model in models:
                    yt = per_model_true[model][idx]
                    ya = per_model_pred_a[model][idx]
                    yb = per_model_pred_b[model][idx]

                    rmse_a_b.append(float(np.sqrt(np.mean((yt - ya) ** 2))))
                    rmse_b_b.append(float(np.sqrt(np.mean((yt - yb) ** 2))))

                    if yt.size >= 2 and np.std(yt) != 0.0 and np.std(ya) != 0.0:
                        v = float(np.corrcoef(yt, ya)[0, 1])
                        if np.isfinite(v):
                            pcc_a_b.append(v)

                    if yt.size >= 2 and np.std(yt) != 0.0 and np.std(yb) != 0.0:
                        v = float(np.corrcoef(yt, yb)[0, 1])
                        if np.isfinite(v):
                            pcc_b_b.append(v)

                da_rmse = float(np.nanmean(rmse_a_b)) if np.any(np.isfinite(rmse_a_b)) else np.nan
                db_rmse = float(np.nanmean(rmse_b_b)) if np.any(np.isfinite(rmse_b_b)) else np.nan
                if np.isfinite(da_rmse) and np.isfinite(db_rmse):
                    rmse_deltas.append(da_rmse - db_rmse)

                da_pcc = float(np.nanmean(pcc_a_b)) if np.any(np.isfinite(pcc_a_b)) else np.nan
                db_pcc = float(np.nanmean(pcc_b_b)) if np.any(np.isfinite(pcc_b_b)) else np.nan
                if np.isfinite(da_pcc) and np.isfinite(db_pcc):
                    pcc_deltas.append(da_pcc - db_pcc)

            pcc_deltas = np.asarray(pcc_deltas, dtype=float)
            rmse_deltas = np.asarray(rmse_deltas, dtype=float)

            if pcc_deltas.size == 0:
                pcc_lo = np.nan
                pcc_hi = np.nan
                pcc_p = np.nan
                n_boot_pcc = 0
            else:
                pcc_lo = float(np.quantile(pcc_deltas, ci_alpha / 2.0))
                pcc_hi = float(np.quantile(pcc_deltas, 1.0 - ci_alpha / 2.0))
                pcc_p = float(2.0 * min(np.mean(pcc_deltas <= 0.0), np.mean(pcc_deltas >= 0.0)))
                n_boot_pcc = int(pcc_deltas.size)

            if rmse_deltas.size == 0:
                rmse_lo = np.nan
                rmse_hi = np.nan
                rmse_p = np.nan
                n_boot_rmse = 0
            else:
                rmse_lo = float(np.quantile(rmse_deltas, ci_alpha / 2.0))
                rmse_hi = float(np.quantile(rmse_deltas, 1.0 - ci_alpha / 2.0))
                rmse_p = float(2.0 * min(np.mean(rmse_deltas <= 0.0), np.mean(rmse_deltas >= 0.0)))
                n_boot_rmse = int(rmse_deltas.size)

            pair_rows_avg.append({
                'method': method,
                'n_poses': int(n_poses),
                'source_a': a,
                'source_b': b,
                'n_targets': int(len(common)),
                'delta_pcc_a_minus_b': pcc_point,
                'delta_pcc_ci_low': pcc_lo,
                'delta_pcc_ci_high': pcc_hi,
                'delta_pcc_p_two_sided': pcc_p,
                'n_boot_pcc': n_boot_pcc,
                'delta_rmse_a_minus_b': rmse_point,
                'delta_rmse_ci_low': rmse_lo,
                'delta_rmse_ci_high': rmse_hi,
                'delta_rmse_p_two_sided': rmse_p,
                'n_boot_rmse': n_boot_rmse,
            })

pairwise_avg = pd.DataFrame(pair_rows_avg)
pairwise_avg.to_csv(f'{out_dir}/multipose_pairwise_sources_modelavg.csv', index=False)


print('[OK] Wrote:')
print(f'  - {out_dir}/multipose_summary_by_model_bootstrap.csv')
print(f'  - {out_dir}/multipose_summary_modelavg_bootstrap.csv')
print(f'  - {out_dir}/multipose_pairwise_sources_by_model.csv')
print(f'  - {out_dir}/multipose_pairwise_sources_modelavg.csv')
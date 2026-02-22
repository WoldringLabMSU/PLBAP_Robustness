import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy import stats

n_boot = 5000
seed = 42
ci_alpha = 0.05

matplotlib.rcParams.update({'font.size': 12})

csv_path = '/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP/results/2026-02-11_CASF2016_Docking-Comparison_All-Results-W-Scores.csv'
out_dir = '/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP/results/figures/2026-02-11'
models = ['dynaformer', 'egna', 'ehign_pla', 'gign', 'onionnet-2']
source_order = ['Crystal', 'GNINA-Crystal', 'Rosetta', 'GNINA-Apo', 'GNINA-AF3', 'AlphaFold3-Cofold', 'Boltz-2', ]

df = pd.read_csv(csv_path)
df['pdbid'] = df['pdbid'].astype(str).str.lower().str.strip()
df['Model'] = df['Model'].astype(str).str.lower().str.strip()
df['Source'] = df['Source'].astype(str).str.lower().str.strip()
df['pose'] = df['pose'].astype(int)

df['pK_true'] = df['pK_true'].fillna(df['-logKd/Ki'])

def pcc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # to handle if bootstrapping resampling is ever wonky
    if y_true.size < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def bootstrap_ci(y_true, y_pred, metric_funct, rng, n_boot=5000, alpha=0.05):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.shape[0]

    point = metric_funct(y_true, y_pred) # metric_funct will be pcc or rmse

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = metric_funct(y_true[idx], y_pred[idx])
        if np.isfinite(v):
            boots.append(v)

    boots = np.asarray(boots, dtype=float)
    if boots.size == 0:
        return point, np.nan, np.nan

    lo = float(np.quantile(boots, alpha/2))
    hi = float(np.quantile(boots, 1-alpha/2))
    return point, lo, hi

def add_ci_errorbars(ax, plot_df, models, source_order):
    lut = {}
    for _, r in plot_df.iterrows():
        lut[(r['Model'], r['source_group'])] = (r['value'], r['ci_lo'], r['ci_hi'])

    containers = [c for c in ax.containers if hasattr(c, 'patches')]
    cont_by_label = {c.get_label(): c for c in containers}

    for src in source_order:
        cont = cont_by_label.get(src)
        if cont is None:
            continue

        for model, patch in zip(models, cont.patches):
            key = (model, src)
            if key not in lut:
                continue

            y, lo, hi = lut[key]
            if not (np.isfinite(y) and np.isfinite(lo) and np.isfinite(hi)):
                continue

            x = patch.get_x() + patch.get_width() / 2
            ax.errorbar(
                x, y,
                yerr=[[y - lo], [hi - y]],
                fmt='none',
                ecolor='black',
                capsize=3,
                lw=1.2,
                zorder=10
            )


rows = []

# crystal (top pose)
d = df[(df['Source'].str.contains('crystal', na=False) & ~df['Source'].str.contains('gnina', na=False)) & df['Model'].isin(models) & (df['pose'] == 1)].copy()
g = d.groupby(['Model', 'pdbid'])[['pK_true', 'pK_predicted']].agg({'pK_true': 'first', 'pK_predicted': 'first'}).reset_index()
g['source_group'] = 'Crystal'
rows.append(g)

# af3 top pose
d = df[(df['Source'].str.contains('alpha', na=False) & ~df['Source'].str.contains('gnina', na=False)) & df['Model'].isin(models) & (df['pose'] == 1)].copy()
g = d.groupby(['Model', 'pdbid'])[['pK_true', 'pK_predicted']].agg({'pK_true': 'first', 'pK_predicted': 'first'}).reset_index()
g['source_group'] = 'AlphaFold3-Cofold'
rows.append(g)

# boltz2 top pose
d = df[df['Source'].str.contains('boltz', na=False) & df['Model'].isin(models) & (df['pose'] == 1)].copy()
g = d.groupby(['Model', 'pdbid'])[['pK_true', 'pK_predicted']].agg({'pK_true': 'first', 'pK_predicted': 'first'}).reset_index()
g['source_group'] = 'Boltz-2'
rows.append(g)

# rosetta top pose
d = df[df['Source'].str.contains('rosetta', na=False) & df['Model'].isin(models) & (df['pose'] == 1)].copy()
g = d.groupby(['Model', 'pdbid'])[['pK_true', 'pK_predicted']].agg({'pK_true': 'first', 'pK_predicted': 'mean'}).reset_index()
g['source_group'] = 'Rosetta'
rows.append(g)

# gnina top pose
d = df[(df['Source'].str.contains('gnina') & ~df['Source'].str.contains('af3', na=False) & ~df['Source'].str.contains('apo', na=False)) & df['Model'].isin(models) & (df['pose'] == 1)].copy()
g = d.groupby(['Model', 'pdbid'])[['pK_true', 'pK_predicted']].agg({'pK_true': 'first', 'pK_predicted': 'mean'}).reset_index()
g['source_group'] = 'GNINA-Crystal'
rows.append(g)

# gnina-af3 top pose
d = df[df['Source'].str.contains('gnina-af3', na=False) & df['Model'].isin(models) & (df['pose'] == 1)].copy()
g = d.groupby(['Model', 'pdbid'])[['pK_true', 'pK_predicted']].agg({'pK_true': 'first', 'pK_predicted': 'mean'}).reset_index()
g['source_group'] = 'GNINA-AF3'
rows.append(g)

# gnina-apo top pose
d = df[df['Source'].str.contains('gnina-apo', na=False) & df['Model'].isin(models) & (df['pose'] == 1)].copy()
g = d.groupby(['Model', 'pdbid'])[['pK_true', 'pK_predicted']].agg({'pK_true': 'first', 'pK_predicted': 'mean'}).reset_index()
g['source_group'] = 'GNINA-Apo'
rows.append(g)

agg = pd.concat(rows, ignore_index=True)

rng = np.random.default_rng(seed)

metric_rows = []
for model in models:
    for src in source_order:
        a = agg[(agg['Model'] == model) & (agg['source_group'] == src)].copy()
        if a.empty:
            continue
        
        y_true = a['pK_true'].to_numpy()
        y_pred = a['pK_predicted'].to_numpy()
        
        val, lo, hi = bootstrap_ci(y_true, y_pred, pcc, rng, n_boot=n_boot, alpha=ci_alpha)
        metric_rows.append({'Model': model, 'source_group': src, 'metric': 'pcc', 'value': val, 'ci_lo': lo, 'ci_hi': hi, 'n': len(a)})
        
        val, lo, hi = bootstrap_ci(y_true, y_pred, rmse, rng, n_boot=n_boot, alpha=ci_alpha)
        metric_rows.append({'Model': model, 'source_group': src, 'metric': 'rmse', 'value': val, 'ci_lo': lo, 'ci_hi': hi, 'n': len(a)})

met = pd.DataFrame(metric_rows)
met.to_csv(f'{out_dir}/performance_bootstrap_summary.csv', index=False)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

sns.barplot(
    data=met[met['metric'] == 'pcc'],
    x='Model',
    y='value',
    hue='source_group',
    order=models,
    hue_order=source_order,
    ax=axes[0],
    palette='colorblind'
)
axes[0].set_ylabel('PCC')
axes[0].set_xlabel('')
# axes[0].set_ylim(0.62,0.86)
axes[0].legend(title='', frameon=False, ncol=6, loc='lower center', bbox_to_anchor=[0.5,1.01])
add_ci_errorbars(axes[0], met[met['metric'] == 'pcc'], models, source_order)

sns.barplot(
    data=met[met['metric'] == 'rmse'],
    x='Model',
    y='value',
    hue='source_group',
    order=models,
    hue_order=source_order,
    ax=axes[1],
    palette='colorblind'
)
axes[1].set_ylabel('RMSE')
axes[1].set_xlabel('')
axes[1].legend_.remove()
add_ci_errorbars(axes[1], met[met['metric'] == 'rmse'], models, source_order)

plt.tight_layout()
plt.savefig(f'{out_dir}/barplot_performance_comparison-bootstrap.png', dpi=1200)
plt.close()

print(met[['metric', 'value', 'ci_lo', 'ci_hi']].head(10))
print(met[['ci_lo', 'ci_hi']].isna().mean())


### For Supplement
ttest_csv = f'{out_dir}/pairwise_paired_ttest_all_comparisons.csv'

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


def paired_ttest_on_errors(y_true, pred_a, pred_b, error_metric='abs', alpha=0.05):
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)

    err_a = pred_a - y_true
    err_b = pred_b - y_true

    if error_metric == 'abs':
        e_a = np.abs(err_a)
        e_b = np.abs(err_b)
    elif error_metric == 'sq':
        e_a = err_a ** 2
        e_b = err_b ** 2
    else:
        raise ValueError(error_metric)

    d = e_b - e_a  # positive means A has smaller error than B

    n = d.size
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1))
    se_d = sd_d / np.sqrt(n)

    t_stat, p_two = stats.ttest_1samp(d, 0.0, alternative='two-sided')

    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lo = mean_d - tcrit * se_d
    ci_hi = mean_d + tcrit * se_d

    return {
        'n': int(n),
        'mean_delta_A_better': mean_d,
        't_stat': float(t_stat),
        'p_uncorrected': float(p_two),
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
    }


agg = agg.copy()
agg['method'] = agg['Model'] + '|' + agg['source_group']

wide_pred = agg.pivot_table(index='pdbid', columns='method', values='pK_predicted', aggfunc='first')
y_true = agg.groupby('pdbid')['pK_true'].first()

methods_in_order = [m + '|' + s for s in source_order for m in models]
methods = [x for x in methods_in_order if x in wide_pred.columns]

rows = []
for a, b in combinations(methods, 2):
    tmp = pd.DataFrame({
        'pK_true': y_true,
        'pred_a': wide_pred[a],
        'pred_b': wide_pred[b],
    }).dropna()

    if len(tmp) < 10:
        continue

    for em in ['abs', 'sq']:
        res = paired_ttest_on_errors(
            tmp['pK_true'].to_numpy(),
            tmp['pred_a'].to_numpy(),
            tmp['pred_b'].to_numpy(),
            error_metric=em,
            alpha=0.05
        )
        rows.append({
            'A': a,
            'B': b,
            'error_metric': 'abs_error' if em == 'abs' else 'squared_error',
            **res
        })

pairwise_t = pd.DataFrame(rows)

pairwise_t['p_holm'] = np.nan
for em in pairwise_t['error_metric'].unique():
    mask = pairwise_t['error_metric'] == em
    pairwise_t.loc[mask, 'p_holm'] = holm_bonferroni(pairwise_t.loc[mask, 'p_uncorrected'].to_numpy())

pairwise_t.to_csv(ttest_csv, index=False)

print(pairwise_t[['error_metric', 'p_uncorrected', 'p_holm']].describe())


def nice_label(method):
    model, src = method.split('|', 1)
    return f'{model}\n{src}'

def build_p_matrix(df, error_metric, methods_order, p_col='p_holm'):
    dfm = df[df['error_metric'] == error_metric].copy()

    p_mat = pd.DataFrame(np.nan, index=methods_order, columns=methods_order)

    for _, r in dfm.iterrows():
        a = r['A']
        b = r['B']
        if a not in p_mat.index or b not in p_mat.columns:
            continue

        p = float(r[p_col])
        p_mat.loc[a, b] = p
        p_mat.loc[b, a] = p

    for m in methods_order:
        p_mat.loc[m, m] = np.nan

    return p_mat

def plot_sig_pvals(p_mat, out_png, title, alpha=0.05):
    labels = [nice_label(m) for m in p_mat.index.tolist()]

    vals = p_mat.values.copy()
    upper = np.triu(np.ones_like(vals, dtype=bool), k=1)
    nonsig = ~(vals < alpha)
    mask = upper | nonsig | ~np.isfinite(vals)

    fig_w = max(12, 0.45 * len(labels))
    fig_h = max(10, 0.40 * len(labels))
    plt.figure(figsize=(fig_w, fig_h))

    ax = sns.heatmap(
        p_mat,
        mask=mask,
        vmin=0.0,
        vmax=alpha,
        square=True,
        cbar=True,
        norm=matplotlib.colors.LogNorm(),
        xticklabels=labels,
        yticklabels=labels
    )

    ax.set_title(title)
    ax.tick_params(axis='x', rotation=90)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=1200)
    plt.close()

methods_in_order = [m + '|' + s for s in source_order for m in models]
methods_present = sorted(set(pairwise_t['A']).union(set(pairwise_t['B'])))
methods_order = [m for m in methods_in_order if m in methods_present]

for em in ['abs_error', 'squared_error']:
    if em not in set(pairwise_t['error_metric']):
        continue

    p_mat = build_p_matrix(pairwise_t, em, methods_order, p_col='p_holm')

    plot_sig_pvals(
        p_mat,
        out_png=f'{out_dir}/heatmap_pvals_sigonly_ttest_{em}.png',
        title=f'Holm-adjusted p-values < {ci_alpha} only (blank otherwise) for {em}',
        alpha=ci_alpha
    )


pvals_bar_csv = f'{out_dir}/barplot_pvalues_vs_crystal_pcc_rmse.csv'

def paired_bootstrap_delta_pvalue(y_true, pred_a, pred_b, metric, rng, n_boot=20000, alpha=0.05):
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = y_true.size

    if metric == 'pcc':
        metric_fn = pcc
        def delta(a, b):
            return a - b
    elif metric == 'rmse':
        metric_fn = rmse
        def delta(a, b):
            return b - a
    else:
        raise ValueError(metric)

    point_a = metric_fn(y_true, pred_a)
    point_b = metric_fn(y_true, pred_b)
    point_delta = delta(point_a, point_b)

    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a_i = metric_fn(y_true[idx], pred_a[idx])
        b_i = metric_fn(y_true[idx], pred_b[idx])
        d_i = delta(a_i, b_i)
        if np.isfinite(d_i):
            deltas.append(d_i)

    deltas = np.asarray(deltas, dtype=float)
    if deltas.size == 0:
        return point_delta, np.nan, np.nan, np.nan

    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))

    B = deltas.size
    count_le = int(np.sum(deltas <= 0))
    count_ge = int(np.sum(deltas >= 0))
    p_left = (count_le + 1) / (B + 1)
    p_right = (count_ge + 1) / (B + 1)
    p_two = 2 * min(p_left, p_right)
    p_two = min(p_two, 1.0)

    return point_delta, lo, hi, float(p_two)

rows_bar = []
for source in source_order:
    baseline_src = source
    n_boot_p = 20000
    rng_bar = np.random.default_rng(seed)

    agg_bar = agg.copy()
    agg_bar['method'] = agg_bar['Model'] + '|' + agg_bar['source_group']

    wide_pred_bar = agg_bar.pivot_table(index='pdbid', columns='method', values='pK_predicted', aggfunc='first')
    y_true_bar = agg_bar.groupby('pdbid')['pK_true'].first()

    for model in models:
        base_method = model + '|' + baseline_src
        if base_method not in wide_pred_bar.columns:
            continue

        for src in source_order:
            if src == baseline_src:
                continue

            m_src = model + '|' + src
            if m_src not in wide_pred_bar.columns:
                continue

            tmp = pd.DataFrame({
                'pK_true': y_true_bar,
                'pred_src': wide_pred_bar[m_src],
                'pred_base': wide_pred_bar[base_method],
            }).dropna()

            if len(tmp) < 10:
                continue

            y_true = tmp['pK_true'].to_numpy()
            pred_src = tmp['pred_src'].to_numpy()
            pred_base = tmp['pred_base'].to_numpy()

            d, lo, hi, p = paired_bootstrap_delta_pvalue(
                y_true, pred_src, pred_base,
                metric='pcc',
                rng=rng_bar,
                n_boot=n_boot_p,
                alpha=ci_alpha
            )
            rows_bar.append({
                'Model': model,
                'A_source': src,
                'B_source': baseline_src,
                'metric': 'pcc',
                'delta_A_better': d,
                'ci_lo': lo,
                'ci_hi': hi,
                'p_uncorrected': p,
                'n': int(len(tmp)),
            })

            d, lo, hi, p = paired_bootstrap_delta_pvalue(
                y_true, pred_src, pred_base,
                metric='rmse',
                rng=rng_bar,
                n_boot=n_boot_p,
                alpha=ci_alpha
            )
            rows_bar.append({
                'Model': model,
                'A_source': src,
                'B_source': baseline_src,
                'metric': 'rmse',
                'delta_A_better': d,
                'ci_lo': lo,
                'ci_hi': hi,
                'p_uncorrected': p,
                'n': int(len(tmp)),
            })

pvals_bar = pd.DataFrame(rows_bar)

pvals_bar['p_holm'] = np.nan
for metric in ['pcc', 'rmse']:
    mask = pvals_bar['metric'] == metric
    pvals_bar.loc[mask, 'p_holm'] = holm_bonferroni(pvals_bar.loc[mask, 'p_uncorrected'].to_numpy())

pvals_bar.to_csv(pvals_bar_csv, index=False)

pd.set_option('display.float_format', lambda x: f'{x:.3e}')
print('Saved:', pvals_bar_csv)
print(pvals_bar.sort_values(['metric', 'p_holm']).head(30)[['metric', 'Model', 'A_source', 'delta_A_better', 'ci_lo', 'ci_hi', 'p_uncorrected', 'p_holm', 'n']])

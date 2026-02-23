import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

matplotlib.rcParams.update({'font.size': 12})

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
plip_csv = '/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP/results/plip_counts_FINAL.csv'
out_dir  = '/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP/results/figures/FINAL/plip'

# ─────────────────────────────────────────────
# Load & clean
# ─────────────────────────────────────────────
df = pd.read_csv(plip_csv)
df['count'] = df['count'].astype(int)

rename = {
    'crystal':    'Crystal',
    'gnina':      'GNINA-Crystal',
    'gnina-apo':  'GNINA-Apo',
    'gnina-af3':  'GNINA-AF3',
    'af3':        'AlphaFold3',
}
df['source'] = df['source'].map(rename).fillna(df['source'])

sources = ['Crystal', 'GNINA-Crystal', 'GNINA-Apo', 'GNINA-AF3', 'AlphaFold3']
df = df[df['source'].isin(sources)]
df = df[df['int_type'] != 'water_bridge']

int_types = df['int_type'].unique().tolist()


# ─────────────────────────────────────────────
# Helper: 3×2 violin grid
# ─────────────────────────────────────────────
def make_violin_grid(data, title_suffix, filename, hue=None, split=False,
                     hue_order=None, legend_handles=None, density_norm='width'):
    fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex=True, sharey='row')
    for i, int_type in enumerate(int_types):
        df_int = data[data['int_type'] == int_type]
        n  = i // 2
        j  = i  % 2
        kw = dict(data=df_int, x='source', y='count', order=sources,
                  ax=axes[n, j], inner='quart', density_norm=density_norm)
        if hue:
            kw.update(hue=hue, hue_order=hue_order, split=split,
                      fill=False, legend=False, palette='colorblind', gap=0.1)
        sns.violinplot(**kw)
        axes[n, j].set_title(int_type, y=0.87)
        axes[n, j].set_ylabel('')
        axes[n, j].set_xlabel('')
        axes[n, j].set_xticklabels(
            ['Crystal', 'GNINA-\nCrystal', 'GNINA-\nApo', 'GNINA-\nAF3', 'Alpha-\nFold3'])
    if legend_handles:
        fig.legend(handles=legend_handles, title='Top N', frameon=False,
                   loc='lower right', bbox_to_anchor=(0.98, 0.02))
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{filename}', dpi=1200)
    plt.close()


# ─────────────────────────────────────────────
# Figure 1 – all 100 poses
# ─────────────────────────────────────────────
make_violin_grid(df, '100 poses', 'plip_distribs_100poses.png')

# ─────────────────────────────────────────────
# Figure 2 – top 1 pose only
# ─────────────────────────────────────────────
df1 = df[df['pose'] == 1].copy()
make_violin_grid(df1, 'pose 1', 'plip_distribs_1pose.png')

# ─────────────────────────────────────────────
# Figure 3 – 1 vs 100 poses (split violins)
# ─────────────────────────────────────────────
df['Top N']  = 100
df1['Top N'] = 1
df_all = pd.concat([df, df1])

legend_handles = [
    Patch(facecolor='none',  edgecolor='black', label='Top 1'),
    Patch(facecolor='0.85',  edgecolor='black', label='Top 100'),
]
make_violin_grid(df_all, '1 vs 100', 'plip_distribs_1vs100poses.png',
                 hue='Top N', split=True, hue_order=[1, 100],
                 legend_handles=legend_handles)

# ─────────────────────────────────────────────
# Figure 4 – total interactions, split violins
# ─────────────────────────────────────────────
group_cols = [c for c in ['pdbid', 'source', 'pose'] if c in df.columns]

tot_100 = (df.groupby(group_cols, as_index=False)['count']
             .sum().rename(columns={'count': 'total_interactions'}))
tot_100['Top N'] = 100

tot_1 = (df1.groupby(group_cols, as_index=False)['count']
              .sum().rename(columns={'count': 'total_interactions'}))
tot_1['Top N'] = 1

tot_split = pd.concat([tot_1[tot_1['source'] != 'Crystal'],
                        tot_100[tot_100['source'] != 'Crystal']], ignore_index=True)
tot_crys  = tot_1[tot_1['source'] == 'Crystal'].copy()

fig, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(data=tot_split, x='source', y='total_interactions',
               hue='Top N', hue_order=[1, 100], split=True, inner='quart',
               order=sources, ax=ax, legend=False, density_norm='width')
sns.violinplot(data=tot_crys,  x='source', y='total_interactions',
               order=sources, ax=ax, inner='quart', density_norm='width')

light_grey  = '0.85'
polys       = [c for c in ax.collections if isinstance(c, PolyCollection)]
n_noncrystal = len(sources) - 1
n_split_polys = 2 * n_noncrystal

for i, pc in enumerate(polys[:n_split_polys]):
    pc.set_edgecolor('black'); pc.set_linewidth(1.0)
    pc.set_facecolor('none' if i % 2 == 0 else light_grey)
for pc in polys[n_split_polys:]:
    pc.set_edgecolor('black'); pc.set_linewidth(1.0); pc.set_facecolor('none')
for ln in ax.lines:
    ln.set_color('black'); ln.set_linewidth(1.0)

ax.legend(handles=legend_handles, title='Top N', frameon=False)
ax.set_xlabel('')
ax.set_ylabel('Total Interactions Per Pose')
ax.set_xticklabels(['Crystal', 'GNINA-\nCrystal', 'GNINA-\nApo', 'GNINA-\nAF3', 'Alpha-\nFold3'])
plt.tight_layout()
plt.savefig(f'{out_dir}/plip_total_interactions_1vs100poses_crystal_full_greywhite.png', dpi=1200)
plt.close()


# ═════════════════════════════════════════════
# STATISTICS
# ═════════════════════════════════════════════

stat_rows = []   # collect every test result here

# ─────────────────────────────────────────────
# 1. Kruskal-Wallis across sources (per int_type, per Top-N condition)
# ─────────────────────────────────────────────
print('\n' + '='*70)
print('KRUSKAL-WALLIS: do any sources differ? (per interaction type)')
print('='*70)

kw_rows = []
for topn_label, data in [('Top100', df), ('Top1', df1)]:
    for int_type in int_types:
        df_int = data[data['int_type'] == int_type]
        groups = [grp['count'].values
                  for _, grp in df_int.groupby('source')
                  if len(grp) > 0]
        if len(groups) < 2:
            continue
        H, p = stats.kruskal(*groups)
        kw_rows.append({'TopN': topn_label, 'int_type': int_type,
                        'H': H, 'p': p,
                        'significant (p<0.05)': p < 0.05})

kw_df = pd.DataFrame(kw_rows)
print(kw_df.to_string(index=False))


# ─────────────────────────────────────────────
# 2. Pairwise Mann-Whitney U with BH correction (per int_type, per Top-N)
# ─────────────────────────────────────────────
print('\n' + '='*70)
print('PAIRWISE MANN-WHITNEY U (BH-corrected) per interaction type')
print('='*70)

mw_rows = []
pairs = list(combinations(sources, 2))

for topn_label, data in [('Top100', df), ('Top1', df1)]:
    for int_type in int_types:
        df_int = data[data['int_type'] == int_type]
        pvals  = []
        valid_pairs = []
        for s1, s2 in pairs:
            g1 = df_int[df_int['source'] == s1]['count'].values
            g2 = df_int[df_int['source'] == s2]['count'].values
            if len(g1) < 2 or len(g2) < 2:
                continue
            _, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            pvals.append(p)
            valid_pairs.append((s1, s2))

        if not pvals:
            continue

        _, pvals_corr, _, _ = multipletests(pvals, method='fdr_bh')
        for (s1, s2), p_raw, p_adj in zip(valid_pairs, pvals, pvals_corr):
            mw_rows.append({
                'TopN': topn_label, 'int_type': int_type,
                'source_1': s1, 'source_2': s2,
                'p_raw': p_raw,
                'p_adj': p_adj,
                'significant (p_adj<0.05)': p_adj < 0.05,
            })

mw_df = pd.DataFrame(mw_rows)
sig_mw = mw_df[mw_df['significant (p_adj<0.05)']]
print(f'\nSignificant pairs (showing {len(sig_mw)} of {len(mw_df)}):')
print(sig_mw.to_string(index=False))


# ─────────────────────────────────────────────
# 3. Kruskal-Wallis on TOTAL interactions across sources
# ─────────────────────────────────────────────
print('\n' + '='*70)
print('KRUSKAL-WALLIS: total interactions across sources')
print('='*70)

tot_kw_rows = []
for topn_label, data in [('Top100', tot_100), ('Top1', tot_1)]:
    groups = [grp['total_interactions'].values
              for _, grp in data.groupby('source')
              if len(grp) > 0]
    H, p = stats.kruskal(*groups)
    tot_kw_rows.append({'TopN': topn_label, 'H': H,
                         'p': p, 'significant': p < 0.05})
    print(f"  {topn_label}: H={H:.3f}, p={p:.5f}")


# ─────────────────────────────────────────────
# 4. Pairwise Mann-Whitney on total interactions
# ─────────────────────────────────────────────
print('\n' + '='*70)
print('PAIRWISE MANN-WHITNEY U on total interactions (BH-corrected)')
print('='*70)

tot_mw_rows = []
for topn_label, data in [('Top100', tot_100), ('Top1', tot_1)]:
    pvals, valid_pairs = [], []
    for s1, s2 in pairs:
        g1 = data[data['source'] == s1]['total_interactions'].values
        g2 = data[data['source'] == s2]['total_interactions'].values
        if len(g1) < 2 or len(g2) < 2:
            continue
        _, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        pvals.append(p)
        valid_pairs.append((s1, s2))

    if not pvals:
        continue
    _, pvals_corr, _, _ = multipletests(pvals, method='fdr_bh')
    for (s1, s2), p_raw, p_adj in zip(valid_pairs, pvals, pvals_corr):
        tot_mw_rows.append({
            'TopN': topn_label, 'source_1': s1, 'source_2': s2,
            'p_raw': p_raw, 'p_adj': p_adj,
            'significant (p_adj<0.05)': p_adj < 0.05,
        })

tot_mw_df = pd.DataFrame(tot_mw_rows)
print(tot_mw_df.to_string(index=False))


# ─────────────────────────────────────────────
# 5. Top-1 vs Top-100 comparison (Wilcoxon if paired, else Mann-Whitney)
# ─────────────────────────────────────────────
print('\n' + '='*70)
print('TOP-1 vs TOP-100: Wilcoxon signed-rank (paired by pdbid, median of Top-100)')
print('='*70)

paired_rows = []
use_pdbid = 'pdbid' in df.columns

for src in sources:
    if src == 'Crystal':
        # Crystal has only one pose; skip Top-100 comparison
        continue

    top1_vals = df1[df1['source'] == src].copy()
    top100_src = df[df['source'] == src].copy()

    if use_pdbid:
        # Aggregate Top-100 to median per pdbid (avoids inflating n)
        top100_agg = (top100_src.groupby(['pdbid', 'int_type'])['count']
                                 .median()
                                 .reset_index()
                                 .rename(columns={'count': 'count_100'}))
        merged = top1_vals.merge(top100_agg, on=['pdbid', 'int_type'], how='inner')

        for int_type in int_types:
            sub = merged[merged['int_type'] == int_type]
            if len(sub) < 5:
                continue
            try:
                stat_w, p_w = stats.wilcoxon(sub['count'], sub['count_100'],
                                              zero_method='wilcox', alternative='two-sided')
            except ValueError:
                # All differences are zero
                stat_w, p_w = np.nan, 1.0
            paired_rows.append({
                'source': src, 'int_type': int_type,
                'median_top1':  sub['count'].median(),
                'median_top100': sub['count_100'].median(),
                'W': stat_w if not np.isnan(stat_w) else 'NA',
                'p': p_w,
                'significant (p<0.05)': p_w < 0.05,
            })
    else:
        # Not paired — use Mann-Whitney
        for int_type in int_types:
            g1 = top1_vals[top1_vals['int_type'] == int_type]['count'].values
            g2 = top100_src[top100_src['int_type'] == int_type]['count'].values
            if len(g1) < 2 or len(g2) < 2:
                continue
            _, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            paired_rows.append({
                'source': src, 'int_type': int_type,
                'median_top1':  np.median(g1),
                'median_top100': np.median(g2),
                'p': p,
                'significant (p<0.05)': p < 0.05,
            })

paired_df = pd.DataFrame(paired_rows)
print(paired_df.to_string(index=False))


# ─────────────────────────────────────────────
# 6. Save all results to CSV
# ─────────────────────────────────────────────
kw_df.to_csv(f'{out_dir}/stats_kruskalwallis_per_inttype.csv', index=False)
mw_df.to_csv(f'{out_dir}/stats_mannwhitney_pairwise_per_inttype.csv', index=False)
tot_mw_df.to_csv(f'{out_dir}/stats_mannwhitney_total_interactions.csv', index=False)
paired_df.to_csv(f'{out_dir}/stats_wilcoxon_top1_vs_top100.csv', index=False)

print('\n' + '='*70)
print('All stats saved to CSV files in out_dir.')
print('='*70)
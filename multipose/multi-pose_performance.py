import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = '/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP/results/2026-02-11_CASF2016_Docking-Comparison_All-Results-W-Scores.csv'
out_dir = '/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP/results/figures/FINAL'
k_list = [1, 2, 3, 5, 10, 20, 30, 50, 100]
models = ['dynaformer', 'egna', 'ehign_pla', 'gign', 'onionnet-2']
# sources = ['gnina-crystal', 'rosetta', 'gnina-apo', 'gnina-af3', 'alphafold3']


sources = ['gnina-crystal', 'gnina-apo', 'gnina-af3', 'alphafold3']

os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df['pdbid'] = df['pdbid'].astype(str).str.lower().str.strip()
df['Model'] = df['Model'].astype(str).str.lower().str.strip()
df['Source'] = df['Source'].astype(str).str.lower().str.strip()
df['pose'] = pd.to_numeric(df['pose'], errors='coerce')

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

rows = []
for method in ['mean', 'median']:
    for source in sources:
        for model in models:
            d = df[(df['Source'].str.contains(source, na=False)) & (df['Model'] == model)].copy()
            d = d.dropna(subset=['pose'])
            d['pose'] = d['pose'].astype(int)
            d = d.sort_values(['pdbid', 'pose'])

            for n_poses in k_list:
                dk = d[d['pose'] <= n_poses].copy()

                dk = dk.dropna(subset=['pK_true', 'pK_predicted'])
                if dk.empty:
                    rows.append({
                        'method': method,
                        'source': source,
                        'model': model,
                        'n_poses': n_poses,
                        'pcc': np.nan,
                        'rmse': np.nan,
                        'n_targets': 0,
                    })
                    continue

                pred_agg = 'mean' if method == 'mean' else 'median'
                agg = dk.groupby('pdbid', as_index=False).agg(
                    pK_true=('pK_true', 'first'),
                    pK_predicted=('pK_predicted', pred_agg),
                )

                pcc_val, rmse_val, n_targets = safe_pcc_rmse(
                    agg['pK_true'].to_numpy(),
                    agg['pK_predicted'].to_numpy(),
                )

                rows.append({
                    'method': method,
                    'source': source,
                    'model': model,
                    'n_poses': n_poses,
                    'pcc': pcc_val,
                    'rmse': rmse_val,
                    'n_targets': n_targets,
                })

res = pd.DataFrame(rows)
res.to_csv(f'{out_dir}/multipose_mean_median_summary.csv', index=False)

for method in ['mean', 'median']:
    sub = res[res['method'] == method].copy()
    sub = sub.sort_values(['source', 'model', 'n_poses'])

    fig, axes = plt.subplots(1, 4, sharex=True, sharey='row', figsize=(8.5, 4))
    for i, source in enumerate(sources):
        sd = sub[sub['source'] == source].copy()

        sns.lineplot(
            data=sd,
            x='n_poses',
            y='pcc',
            hue='model',
            hue_order=models,
            ax=axes[i],
            palette='colorblind',
            errorbar=None,
            estimator=None,
        )

        if i != 0 and axes[i].get_legend() is not None:
            axes[i].get_legend().remove()

        axes[i].set_title(source)
        axes[i].set_xlabel('Number of Poses')

    axes[0].legend(title='')

    for ax in axes.ravel():
        ax.set_xscale('log')
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(['1', '10', '100'])
        ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/multipose_pcc_{method}.png', dpi=1200)
    plt.close()

import pandas as pd

research_dir = '/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP'
sources = ['gnina', 'gnina-apo', 'gnina-af3', 'rosetta', 'af3', 'boltz2']

idx_df = pd.read_csv(f'{research_dir}/preprocessing/casf2016_smiles_seqs.csv')

df_list = []
for source in sources:
    if source in ['gnina', 'gnina-apo', 'gnina-af3', 'rosetta', 'af3']:
        for n in range(100):
            pose_id = str(n+1).zfill(4)
            rmsd_df = pd.read_csv(f'{research_dir}/results/rmsd_by_archive/{source}_best{pose_id}_rmsd.csv')
            df_list.append(rmsd_df)
    else:
        rmsd_df = pd.read_csv(f'{research_dir}/results/rmsd_by_archive/{source}_rmsd.csv')
        df_list.append(rmsd_df)

df = pd.concat(df_list)

df.to_csv(f'{research_dir}/results/CASF2016_rmsd_all-sources_per-pose.csv', index=False)
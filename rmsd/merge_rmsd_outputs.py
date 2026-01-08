import pandas as pd

research_dir = '/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP'

idx_df = pd.read_csv(f'{research_dir}/preprocessing/casf2016_smiles_seqs.csv')

df_list = []
for pdbid in idx_df['pdb_id'].unique():
    rmsd_df = pd.read_csv(f'{research_dir}/results/rmsd/{pdbid}_rmsd.csv')
    rmsd_df.drop(columns=['lig_rmsd', 'n_lig_atoms_matched'], inplace=True)
    lig_df = pd.read_csv(f'{research_dir}/results/rmsd_lig/{pdbid}_ligand_rmsd.csv')
    temp_df = rmsd_df.merge(lig_df, how='left', on=['pdbid', 'source', 'pose'])
    df_list.append(temp_df)

df = pd.concat(df_list)

df.to_csv(f'{research_dir}/results/CASF2016_rmsd_all-sources_per-pose.csv', index=False)
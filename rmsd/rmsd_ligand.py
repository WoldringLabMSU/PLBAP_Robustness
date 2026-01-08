import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms


def load_u(p):
    u = mda.Universe(p)
    try:
        u.guess_TopologyAttrs(context='default', to_guess=['elements'])
    except Exception:
        pass
    return u


def pick_ligand_atoms(u):
    ag = u.select_atoms('not protein and not nucleic and not resname HOH')
    if len(ag) == 0:
        return u.atoms
    if len(ag.residues) <= 1:
        return ag
    res = max(ag.residues, key=lambda r: r.atoms.n_atoms)
    return res.atoms


def matched_positions_by_atom_name(ref_atoms, mob_atoms):
    ref_names = list(ref_atoms.names)
    mob_names = list(mob_atoms.names)

    mob_set = set(mob_names)
    names = [n for n in ref_names if n in mob_set]
    if len(names) < 3:
        return None, None, 0

    ref_map = {}
    for n, p in zip(ref_names, ref_atoms.positions):
        if n not in ref_map:
            ref_map[n] = p

    mob_map = {}
    for n, p in zip(mob_names, mob_atoms.positions):
        if n not in mob_map:
            mob_map[n] = p

    names = [n for n in names if n in ref_map and n in mob_map]
    if len(names) < 3:
        return None, None, 0

    ref_pos = np.asarray([ref_map[n] for n in names])
    mob_pos = np.asarray([mob_map[n] for n in names])
    return ref_pos, mob_pos, ref_pos.shape[0]


def ligand_rmsd_simple(ref_lig_u, mob_lig_u):
    ref = pick_ligand_atoms(ref_lig_u)
    mob = pick_ligand_atoms(mob_lig_u)

    try:
        ref = ref.select_atoms('not element H')
        mob = mob.select_atoms('not element H')
    except Exception:
        ref = ref.select_atoms('not name H*')
        mob = mob.select_atoms('not name H*')

    n_ref = int(len(ref))
    n_mob = int(len(mob))

    if n_ref < 3 or n_mob < 3:
        return np.nan, 0, n_ref, n_mob, 'too_few_atoms'

    if n_ref == n_mob:
        val = rms.rmsd(mob.positions, ref.positions, center=True, superposition=True)
        return float(val), n_ref, n_ref, n_mob, 'order'

    ref_pos, mob_pos, n = matched_positions_by_atom_name(ref, mob)
    if n >= 3:
        val = rms.rmsd(mob_pos, ref_pos, center=True, superposition=True)
        return float(val), int(n), n_ref, n_mob, 'name_intersection'

    return np.nan, 0, n_ref, n_mob, 'no_mapping'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdbid', required=True)
    ap.add_argument('--out_csv', required=True)
    args = ap.parse_args()

    pdbid = str(args.pdbid).strip().lower()
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    base_dir = '/mnt/scratch/jeaves/CASF-2016'
    cryst_dir = f'{base_dir}/coreset'

    sources = {
        'Rosetta': 'rosetta',
        'GNINA-Crystal': 'gnina',
        'GNINA-AF3': 'gnina-af3',
        'AF3': 'af3',
        'Boltz': 'boltz2',
    }

    npose_dict = {
        'Rosetta': 100,
        'GNINA-Crystal': 100,
        'GNINA-AF3': 100,
        'AF3': 5,
    }

    cryst_lig_file = f'{cryst_dir}/{pdbid}/{pdbid}_ligand.pdb'
    if not os.path.exists(cryst_lig_file):
        raise FileNotFoundError(f'Missing crystal ligand: {cryst_lig_file}')

    cryst_lig = load_u(cryst_lig_file)

    rows = []

    for source, name in sources.items():
        if source == 'Boltz':
            src_dir = f'{base_dir}/{name}/{pdbid}'
            lig_file = f'{src_dir}/{pdbid}_boltz2_model_0_ligand.pdb'

            if not os.path.exists(lig_file):
                print(f'[WARN] Ligand file does not exist: {lig_file}. Skipping.')
                continue

            try:
                comp_lig = load_u(lig_file)
                lig_rmsd, n_lig, n_ref_lig, n_mob_lig, lig_method = ligand_rmsd_simple(cryst_lig, comp_lig)
            except Exception:
                lig_rmsd = np.nan
                n_lig = 0
                n_ref_lig = 0
                n_mob_lig = 0
                lig_method = 'error'

            rows.append({
                'pdbid': pdbid,
                'source': source,
                'pose': '0001',
                'lig_rmsd': lig_rmsd,
                'n_lig_atoms_matched': n_lig,
                'lig_n_heavy_ref': n_ref_lig,
                'lig_n_heavy_mob': n_mob_lig,
                'lig_rmsd_method': lig_method,
            })
            continue

        for n in range(npose_dict[source]):
            pose_id = str(n + 1).zfill(4)
            src_dir = f'{base_dir}/{name}_best{pose_id}/{pdbid}'
            lig_file = f'{src_dir}/{pdbid}_ligand.pdb'

            if not os.path.exists(lig_file):
                print(f'[WARN] Ligand file does not exist: {lig_file}. Skipping.')
                continue

            try:
                comp_lig = load_u(lig_file)
                lig_rmsd, n_lig, n_ref_lig, n_mob_lig, lig_method = ligand_rmsd_simple(cryst_lig, comp_lig)
            except Exception:
                lig_rmsd = np.nan
                n_lig = 0
                n_ref_lig = 0
                n_mob_lig = 0
                lig_method = 'error'

            rows.append({
                'pdbid': pdbid,
                'source': source,
                'pose': pose_id,
                'lig_rmsd': lig_rmsd,
                'n_lig_atoms_matched': n_lig,
                'lig_n_heavy_ref': n_ref_lig,
                'lig_n_heavy_mob': n_mob_lig,
                'lig_rmsd_method': lig_method,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)


if __name__ == '__main__':
    main()

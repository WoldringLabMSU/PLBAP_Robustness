import os
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from Bio.Align import PairwiseAligner
import argparse
from pathlib import Path


def renumber_protein_resids_from_1_inplace(u, start=1):
    prot = u.select_atoms('protein')
    chain_ids = [c for c in np.unique(prot.atoms.chainIDs) if str(c).strip() != '']

    if len(chain_ids) == 0:
        residues = prot.residues
        residues.resids = np.arange(start, start + len(residues), dtype=int)
        return

    for c in chain_ids:
        residues = u.select_atoms(f'protein and chainID {c}').residues
        residues.resids = np.arange(start, start + len(residues), dtype=int)


def protein_chain_ids(u):
    prot = u.select_atoms('protein')
    ids = [str(x).strip() for x in np.unique(prot.atoms.chainIDs)]
    ids = [x for x in ids if x != '']
    if len(ids) == 0:
        return [None]
    return ids


def chain_residues(u, chain_id):
    if chain_id is None:
        return u.select_atoms('protein').residues
    return u.select_atoms(f'protein and chainID {chain_id}').residues


def residue_sequence(residues):
    try:
        return str(residues.sequence())
    except Exception:
        return ''


def build_residue_pairs_from_alignment(ref_res, mob_res, aln):
    pairs = []
    for (rs, re), (ms, me) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(re - rs):
            i = rs + k
            j = ms + k
            if i < 0 or j < 0 or i >= len(ref_res) or j >= len(mob_res):
                continue
            pairs.append((ref_res[i], mob_res[j]))
    return pairs


def matched_positions_by_atom_name(ref_atoms, mob_atoms):
    if len(ref_atoms) == 0 or len(mob_atoms) == 0:
        return None, None, 0

    ref_names = list(ref_atoms.names)
    mob_names = list(mob_atoms.names)

    if len(set(ref_names)) != len(ref_names):
        return None, None, 0
    if len(set(mob_names)) != len(mob_names):
        return None, None, 0

    mob_set = set(mob_names)
    names = [n for n in ref_names if n in mob_set]
    if len(names) < 3:
        return None, None, 0

    ref_map = dict(zip(ref_names, ref_atoms.positions))
    mob_map = dict(zip(mob_names, mob_atoms.positions))

    ref_pos = np.asarray([ref_map[n] for n in names])
    mob_pos = np.asarray([mob_map[n] for n in names])
    return ref_pos, mob_pos, ref_pos.shape[0]


def matched_protein_positions_from_pairs(pairs, atom_sel):
    ref_pos = []
    mob_pos = []
    n_atoms = 0

    for ref_res, mob_res in pairs:
        ref_atoms = ref_res.atoms.select_atoms(atom_sel)
        mob_atoms = mob_res.atoms.select_atoms(atom_sel)

        rpos, mpos, n = matched_positions_by_atom_name(ref_atoms, mob_atoms)
        if n == 0:
            continue

        ref_pos.append(rpos)
        mob_pos.append(mpos)
        n_atoms += n

    if n_atoms < 3:
        return None, None, 0

    return np.vstack(ref_pos), np.vstack(mob_pos), n_atoms


def matched_ca_positions_from_pairs(pairs):
    ref_pos = []
    mob_pos = []

    for ref_res, mob_res in pairs:
        ref_ca = ref_res.atoms.select_atoms('name CA')
        mob_ca = mob_res.atoms.select_atoms('name CA')
        if len(ref_ca) != 1 or len(mob_ca) != 1:
            continue
        ref_pos.append(ref_ca.positions[0])
        mob_pos.append(mob_ca.positions[0])

    if len(ref_pos) < 3:
        return None, None, 0

    ref_pos = np.asarray(ref_pos)
    mob_pos = np.asarray(mob_pos)
    return ref_pos, mob_pos, ref_pos.shape[0]


def seq_assisted_protein_rmsds(ref_u, mob_u, aligner, ref_chain_prefer='A'):
    ref_chains = protein_chain_ids(ref_u)
    if ref_chain_prefer in ref_chains:
        ref_chain = ref_chain_prefer
    else:
        ref_chain = ref_chains[0]

    ref_res = chain_residues(ref_u, ref_chain)
    ref_seq = residue_sequence(ref_res)
    if len(ref_seq) == 0:
        return np.nan, 0, np.nan, 0, np.nan, 0, ref_chain, None

    best_pairs = None
    best_mob_chain = None
    best_nres = 0
    best_score = None

    for mob_chain in protein_chain_ids(mob_u):
        mob_res = chain_residues(mob_u, mob_chain)
        mob_seq = residue_sequence(mob_res)
        if len(mob_seq) == 0:
            continue

        try:
            aln = aligner.align(ref_seq, mob_seq)[0]
        except Exception:
            continue

        nres = int(sum(re - rs for rs, re in aln.aligned[0]))
        score = float(aln.score)

        if nres > best_nres:
            best_nres = nres
            best_score = score
            best_mob_chain = mob_chain
            best_pairs = build_residue_pairs_from_alignment(ref_res, mob_res, aln)
        elif nres == best_nres and best_score is not None and score > best_score:
            best_score = score
            best_mob_chain = mob_chain
            best_pairs = build_residue_pairs_from_alignment(ref_res, mob_res, aln)

    if best_pairs is None or len(best_pairs) < 3:
        return np.nan, 0, np.nan, 0, np.nan, 0, ref_chain, best_mob_chain

    ca_ref, ca_mob, n_ca = matched_ca_positions_from_pairs(best_pairs)
    if n_ca >= 3:
        ca_rmsd = float(rms.rmsd(ca_mob, ca_ref, center=True, superposition=True))
    else:
        ca_rmsd = np.nan

    sc_ref, sc_mob, n_sc = matched_protein_positions_from_pairs(best_pairs, 'not backbone and not name H*')
    if n_sc >= 3:
        sc_rmsd = float(rms.rmsd(sc_mob, sc_ref, center=True, superposition=True))
    else:
        sc_rmsd = np.nan

    all_ref, all_mob, n_all = matched_protein_positions_from_pairs(best_pairs, 'not name H*')
    if n_all >= 3:
        all_rmsd = float(rms.rmsd(all_mob, all_ref, center=True, superposition=True))
    else:
        all_rmsd = np.nan

    return ca_rmsd, n_ca, sc_rmsd, n_sc, all_rmsd, n_all, ref_chain, best_mob_chain


def ligand_rmsd_by_atom_name(ref_lig_u, mob_lig_u):
    ref = ref_lig_u.atoms.select_atoms('not name H*')
    mob = mob_lig_u.atoms.select_atoms('not name H*')

    if len(ref) < 3 or len(mob) < 3:
        return np.nan, 0

    if len(ref) == len(mob) and np.all(ref.names == mob.names):
        val = rms.rmsd(mob.positions, ref.positions, center=True, superposition=True)
        return float(val), int(len(ref))

    rpos, mpos, n = matched_positions_by_atom_name(ref, mob)
    if n < 3:
        return np.nan, 0

    val = rms.rmsd(mpos, rpos, center=True, superposition=True)
    return float(val), int(n)


ap = argparse.ArgumentParser()
ap.add_argument('--pdbid', required=True)
ap.add_argument('--out_csv', required=True)
args = ap.parse_args()

pdbid = str(args.pdbid).strip().lower()
out_csv = Path(args.out_csv)
out_csv.parent.mkdir(parents=True, exist_ok=True)

base_dir = '/mnt/scratch/jeaves/CASF-2016'
cryst_dir = f'{base_dir}/coreset'

sources = {'Rosetta': 'rosetta',
           'GNINA-Crystal': 'gnina',
           'GNINA-AF3': 'gnina-af3',
           'AF3': 'af3',
           'Boltz': 'boltz2'}

npose_dict = {'Rosetta': 100, 'GNINA-Crystal': 100, 'GNINA-AF3': 100, 'AF3': 5}

aligner = PairwiseAligner()
aligner.match_score = 2
aligner.mismatch_score = -1
aligner.open_gap_score = -2
aligner.extend_gap_score = -0.1

results_df_list = []

cryst_prot_file = f'{cryst_dir}/{pdbid}/{pdbid}_protein.pdb'
cryst_lig_file = f'{cryst_dir}/{pdbid}/{pdbid}_ligand.pdb'

if not os.path.exists(cryst_prot_file):
    raise FileNotFoundError(f'Missing crystal protein: {cryst_prot_file}')

cryst_prot = mda.Universe(cryst_prot_file)
cryst_prot_from1 = mda.Universe(cryst_prot_file)
renumber_protein_resids_from_1_inplace(cryst_prot_from1)

cryst_lig = None
if os.path.exists(cryst_lig_file):
    cryst_lig = mda.Universe(cryst_lig_file)

for source, name in sources.items():
    if source == 'Boltz':
        src_dir = f'{base_dir}/{name}/{pdbid}'
        prot_file = f'{src_dir}/{pdbid}_boltz2_model_0_protein.pdb'
        lig_file = f'{src_dir}/{pdbid}_boltz2_model_0_ligand.pdb'

        if not os.path.exists(prot_file):
            print(f'[WARN] Protein file does not exist: {prot_file}. Skipping.')
            continue
        if not os.path.exists(lig_file):
            print(f'[WARN] Ligand file does not exist: {lig_file}. Skipping.')
            continue

        ref_prot = cryst_prot_from1
        comp_prot = mda.Universe(prot_file)

        try:
            comp_lig = mda.Universe(lig_file)
        except Exception:
            comp_lig = None

        shape_cryst_prot_chainA = ref_prot.select_atoms('name CA and protein and chainID A').positions.shape
        shape_cryst_prot_chainB = ref_prot.select_atoms('name CA and protein and chainID B').positions.shape
        shape_cryst_prot_NoChain = ref_prot.select_atoms('name CA and protein').positions.shape
        shape_comp_prot = comp_prot.select_atoms('name CA and protein').positions.shape

        try:
            ca_rmsd, n_ca, sc_rmsd, n_sc, all_rmsd, n_all, ref_chain_used, mob_chain_used = seq_assisted_protein_rmsds(
                ref_prot,
                comp_prot,
                aligner,
                ref_chain_prefer='A'
            )
        except Exception:
            ca_rmsd = np.nan
            n_ca = 0
            sc_rmsd = np.nan
            n_sc = 0
            all_rmsd = np.nan
            n_all = 0
            ref_chain_used = None
            mob_chain_used = None

        if cryst_lig is not None and comp_lig is not None:
            try:
                lig_rmsd, n_lig = ligand_rmsd_by_atom_name(cryst_lig, comp_lig)
            except Exception:
                lig_rmsd = np.nan
                n_lig = 0
        else:
            lig_rmsd = np.nan
            n_lig = 0

        results_df_list.append({'pdbid': pdbid,
                                'source': source,
                                'pose': '0001',
                                'bb_sel_shape_cryst_prot-ChainA': shape_cryst_prot_chainA,
                                'bb_sel_shape_cryst_prot-ChainB': shape_cryst_prot_chainB,
                                'bb_sel_shape_cryst_prot-NoChain': shape_cryst_prot_NoChain,
                                'bb_sel_shape_comp_prot': shape_comp_prot,
                                'n_ca_matched': n_ca,
                                'ref_chain_used': ref_chain_used,
                                'mob_chain_used': mob_chain_used,
                                'prot_ca_rmsd': ca_rmsd,
                                'prot_sidechain_rmsd': sc_rmsd,
                                'n_sc_atoms_matched': n_sc,
                                'prot_allheavy_rmsd': all_rmsd,
                                'n_all_atoms_matched': n_all,
                                'lig_rmsd': lig_rmsd,
                                'n_lig_atoms_matched': n_lig})
    else:
        for n in range(npose_dict[source]):
            pose_id = str(n + 1).zfill(4)

            src_dir = f'{base_dir}/{name}_best{pose_id}/{pdbid}'
            prot_file = f'{src_dir}/{pdbid}_protein.pdb'
            lig_file = f'{src_dir}/{pdbid}_ligand.pdb'

            if source in ['GNINA-AF3', 'AF3']:
                ref_prot = cryst_prot_from1
            else:
                ref_prot = cryst_prot

            if not os.path.exists(prot_file):
                print(f'[WARN] Protein file does not exist: {prot_file}. Skipping.')
                continue
            if not os.path.exists(lig_file):
                print(f'[WARN] Ligand file does not exist: {lig_file}. Skipping.')
                continue

            comp_prot = mda.Universe(prot_file)

            try:
                comp_lig = mda.Universe(lig_file)
            except Exception:
                comp_lig = None

            shape_cryst_prot_chainA = ref_prot.select_atoms('name CA and protein and chainID A').positions.shape
            shape_cryst_prot_chainB = ref_prot.select_atoms('name CA and protein and chainID B').positions.shape
            shape_cryst_prot_NoChain = ref_prot.select_atoms('name CA and protein').positions.shape
            shape_comp_prot = comp_prot.select_atoms('name CA and protein').positions.shape

            try:
                ca_rmsd, n_ca, sc_rmsd, n_sc, all_rmsd, n_all, ref_chain_used, mob_chain_used = seq_assisted_protein_rmsds(
                    ref_prot,
                    comp_prot,
                    aligner,
                    ref_chain_prefer='A'
                )
            except Exception:
                ca_rmsd = np.nan
                n_ca = 0
                sc_rmsd = np.nan
                n_sc = 0
                all_rmsd = np.nan
                n_all = 0
                ref_chain_used = None
                mob_chain_used = None

            if cryst_lig is not None and comp_lig is not None:
                try:
                    lig_rmsd, n_lig = ligand_rmsd_by_atom_name(cryst_lig, comp_lig)
                except Exception:
                    lig_rmsd = np.nan
                    n_lig = 0
            else:
                lig_rmsd = np.nan
                n_lig = 0

            results_df_list.append({'pdbid': pdbid,
                                    'source': source,
                                    'pose': pose_id,
                                    'bb_sel_shape_cryst_prot-ChainA': shape_cryst_prot_chainA,
                                    'bb_sel_shape_cryst_prot-ChainB': shape_cryst_prot_chainB,
                                    'bb_sel_shape_cryst_prot-NoChain': shape_cryst_prot_NoChain,
                                    'bb_sel_shape_comp_prot': shape_comp_prot,
                                    'n_ca_matched': n_ca,
                                    'ref_chain_used': ref_chain_used,
                                    'mob_chain_used': mob_chain_used,
                                    'prot_ca_rmsd': ca_rmsd,
                                    'prot_sidechain_rmsd': sc_rmsd,
                                    'n_sc_atoms_matched': n_sc,
                                    'prot_allheavy_rmsd': all_rmsd,
                                    'n_all_atoms_matched': n_all,
                                    'lig_rmsd': lig_rmsd,
                                    'n_lig_atoms_matched': n_lig})

df = pd.DataFrame(results_df_list)
df.to_csv(out_csv, index=False)

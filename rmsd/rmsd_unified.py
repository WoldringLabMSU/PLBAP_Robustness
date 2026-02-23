#!/usr/bin/env python3
import os
import sys
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from Bio.Align import PairwiseAligner


# =========================
# Protein helpers (yours)
# =========================

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


def matched_positions_by_atom_name_unique(ref_atoms, mob_atoms):
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

        rpos, mpos, n = matched_positions_by_atom_name_unique(ref_atoms, mob_atoms)
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
    ref_chain = ref_chain_prefer if ref_chain_prefer in ref_chains else ref_chains[0]

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
    ca_rmsd = float(rms.rmsd(ca_mob, ca_ref, center=True, superposition=True)) if n_ca >= 3 else np.nan

    sc_ref, sc_mob, n_sc = matched_protein_positions_from_pairs(best_pairs, 'not backbone and not name H*')
    sc_rmsd = float(rms.rmsd(sc_mob, sc_ref, center=True, superposition=True)) if n_sc >= 3 else np.nan

    all_ref, all_mob, n_all = matched_protein_positions_from_pairs(best_pairs, 'not name H*')
    all_rmsd = float(rms.rmsd(all_mob, all_ref, center=True, superposition=True)) if n_all >= 3 else np.nan

    return ca_rmsd, n_ca, sc_rmsd, n_sc, all_rmsd, n_all, ref_chain, best_mob_chain


# =========================
# Ligand helpers (robust)
# =========================

def load_u(p):
    u = mda.Universe(str(p))
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


def matched_positions_by_atom_name_first(ref_atoms, mob_atoms):
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

    ref_pos, mob_pos, n = matched_positions_by_atom_name_first(ref, mob)
    if n >= 3:
        val = rms.rmsd(mob_pos, ref_pos, center=True, superposition=True)
        return float(val), int(n), n_ref, n_mob, 'name_intersection'

    return np.nan, 0, n_ref, n_mob, 'no_mapping'


# =========================
# PDB sanitize + safe loader
# =========================

_DECIMAL_RE = re.compile(r'[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?')

def sanitize_pdb_coords(in_pdb: Path, out_pdb: Path) -> bool:
    """
    Try to repair malformed coordinate columns by reconstructing x/y/z in fixed-width fields.
    Uses first 3 decimal numbers on ATOM/HETATM lines as x/y/z.
    Returns True if it wrote a file and changed at least one ATOM/HETATM line.
    """
    changed = False
    out_lines = []

    with in_pdb.open('r', errors='replace') as fh:
        for line in fh:
            if line.startswith(('ATOM  ', 'HETATM')):
                # Try fixed-width parse first
                try:
                    float(line[30:38]); float(line[38:46]); float(line[46:54])
                    out_lines.append(line if line.endswith('\n') else line + '\n')
                    continue
                except Exception:
                    pass

                decs = _DECIMAL_RE.findall(line)
                if len(decs) >= 3:
                    try:
                        x, y, z = map(float, decs[:3])
                        prefix = (line[:30] if len(line) >= 30 else line).ljust(30)
                        suffix = line[54:] if len(line) >= 54 else '\n'
                        new_line = f"{prefix}{x:8.3f}{y:8.3f}{z:8.3f}{suffix}"
                        if not new_line.endswith('\n'):
                            new_line += '\n'
                        out_lines.append(new_line)
                        changed = True
                        continue
                    except Exception:
                        pass

            out_lines.append(line if line.endswith('\n') else line + '\n')

    if not changed:
        return False

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_text(''.join(out_lines))
    return True


def safe_load_u(p: Path, fix_dir: Path | None, label: str):
    """
    Returns (Universe or None, status_str).
    If MDAnalysis fails due to malformed PDB coords, attempt sanitize and retry.
    """
    try:
        return load_u(p), 'ok'
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"

        # Try sanitize only for PDB float parse failures
        if fix_dir is not None and p.suffix.lower() == '.pdb' and 'could not convert string to float' in str(e):
            fixed = fix_dir / (p.stem + '.fixed.pdb')
            try:
                did = sanitize_pdb_coords(p, fixed)
                if did and fixed.exists():
                    try:
                        u = load_u(fixed)
                        return u, f'fixed({label})'
                    except Exception as e2:
                        msg = f"{msg} | retry_failed: {type(e2).__name__}: {e2}"
            except Exception as e3:
                msg = f"{msg} | sanitize_failed: {type(e3).__name__}: {e3}"

        return None, f'load_error({label}): {msg}'


# =========================
# Topdir parsing
# =========================

def parse_topdir(topdir: str):
    if topdir == 'boltz2':
        return 'Boltz', '0001'
    if '_best' in topdir:
        name, pose = topdir.split('_best', 1)
        pose_id = pose.strip()
        src_map = {
            'rosetta': 'Rosetta',
            'gnina': 'GNINA-Crystal',
            'gnina-af3': 'GNINA-AF3',
            'gnina-apo': 'GNINA-Apo',
            'af3': 'AF3',
        }
        return src_map.get(name, name), pose_id
    return topdir, '0000'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topdir', required=True)
    ap.add_argument('--extract_root', required=True)
    ap.add_argument('--pdb_csv', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--base_dir', default='/mnt/scratch/jeaves/CASF-2016')
    ap.add_argument('--ref_chain_prefer', default='A')
    args = ap.parse_args()

    topdir = str(args.topdir).strip()
    extract_root = Path(args.extract_root)
    base_dir = Path(args.base_dir)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    source_label, pose_id = parse_topdir(topdir)

    # extracted topdir root
    extracted_top = extract_root / topdir
    if not extracted_top.exists():
        print(f"[ERROR] extracted_top not found: {extracted_top}", file=sys.stderr)
        raise SystemExit(2)

    # temp fix dir lives under extract_root (auto-cleaned by sbatch trap)
    fix_dir = extract_root / ".pdbfix" / topdir

    # Alignment
    aligner = PairwiseAligner()
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.1

    df_ids = pd.read_csv(args.pdb_csv)
    pdbids = df_ids['pdb_id'].astype(str).str.strip().str.lower().drop_duplicates().tolist()

    cryst_dir = base_dir / 'coreset'
    rows = []

    for pdbid in pdbids:
        comp_dir = extracted_top / pdbid
        if not comp_dir.exists():
            continue

        # reference
        cryst_prot_file = cryst_dir / pdbid / f'{pdbid}_protein.pdb'
        cryst_lig_file = cryst_dir / pdbid / f'{pdbid}_ligand.pdb'
        if not cryst_prot_file.exists():
            continue

        # comp files
        if source_label == 'Boltz':
            prot_file = comp_dir / f'{pdbid}_boltz2_model_0_protein.pdb'
            lig_file = comp_dir / f'{pdbid}_boltz2_model_0_ligand.pdb'
        else:
            prot_file = comp_dir / f'{pdbid}_protein.pdb'
            lig_file = comp_dir / f'{pdbid}_ligand.pdb'

        if not prot_file.exists() or not lig_file.exists():
            continue

        # Load references (should be clean; but keep safe anyway)
        ref_prot_u, ref_status = safe_load_u(cryst_prot_file, fix_dir, 'ref_prot')
        if ref_prot_u is None:
            print(f"[WARN] {pdbid} failed ref protein load: {ref_status}", file=sys.stderr)
            continue

        ref_prot_from1_u, ref1_status = safe_load_u(cryst_prot_file, fix_dir, 'ref_prot_from1')
        if ref_prot_from1_u is None:
            print(f"[WARN] {pdbid} failed ref protein load (from1): {ref1_status}", file=sys.stderr)
            continue
        renumber_protein_resids_from_1_inplace(ref_prot_from1_u)

        cryst_lig_u = None
        if cryst_lig_file.exists():
            cryst_lig_u, _ = safe_load_u(cryst_lig_file, fix_dir, 'ref_lig')

        # Load comparison protein/ligand (this is where gnina-apo blows up)
        comp_prot_u, comp_prot_status = safe_load_u(prot_file, fix_dir, 'comp_prot')
        if comp_prot_u is None:
            print(f"[WARN] {pdbid} failed comp protein load: {comp_prot_status} | file={prot_file}", file=sys.stderr)
            # record a row with error instead of killing the whole archive
            rows.append({
                'pdbid': pdbid, 'source': source_label, 'pose': pose_id,
                'prot_ca_rmsd': np.nan, 'prot_sidechain_rmsd': np.nan, 'prot_allheavy_rmsd': np.nan,
                'lig_rmsd': np.nan,
                'status': 'comp_prot_load_failed',
                'error': comp_prot_status,
            })
            continue

        comp_lig_u, comp_lig_status = safe_load_u(lig_file, fix_dir, 'comp_lig')
        if comp_lig_u is None:
            print(f"[WARN] {pdbid} failed comp ligand load: {comp_lig_status} | file={lig_file}", file=sys.stderr)

        # choose ref protein consistent with earlier logic
        ref_prot = ref_prot_from1_u if source_label in ['GNINA-AF3', 'AF3', 'Boltz'] else ref_prot_u

        # shapes (keep your columns)
        shape_cryst_chainA = ref_prot.select_atoms('name CA and protein and chainID A').positions.shape
        shape_cryst_chainB = ref_prot.select_atoms('name CA and protein and chainID B').positions.shape
        shape_cryst_nochain = ref_prot.select_atoms('name CA and protein').positions.shape
        shape_comp_prot = comp_prot_u.select_atoms('name CA and protein').positions.shape

        # protein RMSDs
        try:
            ca_rmsd, n_ca, sc_rmsd, n_sc, all_rmsd, n_all, ref_chain_used, mob_chain_used = seq_assisted_protein_rmsds(
                ref_prot, comp_prot_u, aligner, ref_chain_prefer=args.ref_chain_prefer
            )
            prot_status = 'ok'
            prot_err = ''
        except Exception as e:
            ca_rmsd = np.nan; n_ca = 0
            sc_rmsd = np.nan; n_sc = 0
            all_rmsd = np.nan; n_all = 0
            ref_chain_used = None; mob_chain_used = None
            prot_status = 'prot_rmsd_failed'
            prot_err = f"{type(e).__name__}: {e}"

        # ligand RMSD
        lig_rmsd = np.nan
        n_lig = 0
        n_ref_lig = 0
        n_mob_lig = 0
        lig_method = 'no_ref_or_load_failed'
        if cryst_lig_u is not None and comp_lig_u is not None:
            try:
                lig_rmsd, n_lig, n_ref_lig, n_mob_lig, lig_method = ligand_rmsd_simple(cryst_lig_u, comp_lig_u)
            except Exception as e:
                lig_method = f'lig_rmsd_error:{type(e).__name__}'

        status = prot_status if prot_status != 'ok' else ('ok' if comp_lig_u is not None else 'comp_lig_load_failed')
        err = prot_err if prot_err else ('' if comp_lig_u is not None else comp_lig_status)

        rows.append({
            'pdbid': pdbid,
            'source': source_label,
            'pose': pose_id,

            'bb_sel_shape_cryst_prot-ChainA': shape_cryst_chainA,
            'bb_sel_shape_cryst_prot-ChainB': shape_cryst_chainB,
            'bb_sel_shape_cryst_prot-NoChain': shape_cryst_nochain,
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
            'n_lig_atoms_matched': n_lig,
            'lig_n_heavy_ref': n_ref_lig,
            'lig_n_heavy_mob': n_mob_lig,
            'lig_rmsd_method': lig_method,

            'status': status,
            'error': err,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"[INFO] topdir={topdir} wrote {len(out_df)} rows -> {out_csv}")


if __name__ == '__main__':
    main()

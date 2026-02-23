#!/bin/bash --login
#SBATCH --job-name=rmsd_by_archive
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=302%1
#SBATCH --output=/mnt/scratch/jeaves/CASF-2016/rmsd_by_archive/logs/%x_%A_%a.out
#SBATCH --error=/mnt/scratch/jeaves/CASF-2016/rmsd_by_archive/logs/%x_%A_%a.err

set -euo pipefail

module purge
module load Miniforge3
conda activate prep_env

PDB_CSV="/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP/preprocessing/casf2016_smiles_seqs.csv"
PREPPED_DIR="/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP/prepped_structures"
BASE_DIR="/mnt/scratch/jeaves/CASF-2016"

PY_SCRIPT="/mnt/research/woldring_lab/Members/Eaves/PLBAP_Robustness/rmsd/rmsd_unified.py"
OUT_DIR="/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP/results/rmsd_by_archive"

mkdir -p "${OUT_DIR}"
mkdir -p "/mnt/scratch/jeaves/CASF-2016/rmsd_by_archive/logs"

# Map task -> (TOPDIR, ARCHIVE)
read -r TOPDIR ARCHIVE_NAME < <(
python3 - <<'PY'
import os
tid = int(os.environ["SLURM_ARRAY_TASK_ID"])
sources = ["rosetta", "gnina", "gnina-af3", "gnina-apo", "af3"]

if tid == 1:
    print("boltz2", "boltz2.tar.gz")
else:
    idx = tid - 2
    src = sources[idx // 100]
    pose = (idx % 100) + 1
    topdir = f"{src}_best{pose:04d}"
    print(topdir, f"{topdir}.tar.gz")
PY
)

ARCHIVE_PATH="${PREPPED_DIR}/${ARCHIVE_NAME}"
if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "[WARN] missing archive: ${ARCHIVE_PATH} -> skip"
  exit 0
fi

TMP_PARENT="/mnt/scratch/jeaves/CASF-2016/rmsd_by_archive/tmp"
mkdir -p "${TMP_PARENT}"
TMPDIR="$(mktemp -d "${TMP_PARENT}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_XXXXXX")"

cleanup() { rm -rf "${TMPDIR}" || true; }
trap cleanup EXIT

echo "[INFO] TOPDIR=${TOPDIR}"
echo "[INFO] ARCHIVE=${ARCHIVE_PATH}"
echo "[INFO] TMPDIR=${TMPDIR}"

first_file="$(tar -tzf "${ARCHIVE_PATH}" | sed 's#^\./##' | grep -v '/$' | head -n 1 || true)"
echo "[INFO] first_file=${first_file}"

if [[ -z "${first_file}" ]]; then
  echo "[WARN] archive seems empty -> skip"
  exit 0
fi

# Extract so that (ideally) we have ${TMPDIR}/${TOPDIR}/...
if [[ "${first_file}" == "${TOPDIR}/"* ]]; then
  tar -xzf "${ARCHIVE_PATH}" -C "${TMPDIR}"
else
  mkdir -p "${TMPDIR}/${TOPDIR}"
  tar -xzf "${ARCHIVE_PATH}" -C "${TMPDIR}/${TOPDIR}"
fi

# ---- Layout fix without sample pdbid ----
TOPDIR_REAL="${TOPDIR}"
EXTRACT_ROOT_REAL="${TMPDIR}"

# If archive is double-nested: TMPDIR/TOPDIR/TOPDIR/<pdbid>/...
if [[ -d "${TMPDIR}/${TOPDIR}/${TOPDIR}" ]]; then
  echo "[WARN] Detected double-nested TOPDIR. Using extract_root=${TMPDIR}/${TOPDIR}"
  EXTRACT_ROOT_REAL="${TMPDIR}/${TOPDIR}"
fi

# Sanity: require the topdir folder exists where python expects it
if [[ ! -d "${EXTRACT_ROOT_REAL}/${TOPDIR_REAL}" ]]; then
  echo "[WARN] Expected extracted folder missing: ${EXTRACT_ROOT_REAL}/${TOPDIR_REAL} -> skip"
  exit 0
fi

OUT_CSV="${OUT_DIR}/${TOPDIR_REAL}_rmsd.csv"

python3 "${PY_SCRIPT}" \
  --topdir "${TOPDIR_REAL}" \
  --extract_root "${EXTRACT_ROOT_REAL}" \
  --pdb_csv "${PDB_CSV}" \
  --out_csv "${OUT_CSV}" \
  --base_dir "${BASE_DIR}"

echo "[INFO] done -> ${OUT_CSV}"

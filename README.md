# Robustness of Protein-Ligand Binding Affinity Prediction Models to Docked and Predicted Structures

**Joelle N. Eaves and Daniel R. Woldring**
Department of Chemical Engineering and Materials Science, Michigan State University
Institute for Quantitative Health Science and Engineering, Michigan State University

*Journal of Chemical Information and Modeling*, 2026

---

## Overview

Structure-based deep learning models for protein-ligand binding affinity prediction (PLBAP) are commonly benchmarked on experimentally resolved co-crystal structures, but real use-cases often rely on computationally generated inputs. This repository contains the code and aggregated results for a systematic study of how PLBAP performance changes when five reproducible pipelines are evaluated on:

- Experimental co-crystal structures (Crystal)
- GNINA rigid-receptor docking into holo co-crystal receptors (GNINA-Crystal)
- GNINA rigid-receptor docking into apo receptors (GNINA-Apo)
- GNINA rigid-receptor docking into AlphaFold3-predicted receptors (GNINA-AF3)
- AlphaFold3 co-folding (AlphaFold3)

The five PLBAP models evaluated are **Dynaformer**, **EGNA**, **EHIGN-PLA**, **GIGN**, and **OnionNet-2**, benchmarked on the CASF-2016 dataset. Analyses address three questions: (1) how much does PLBAP performance degrade across structure sources, (2) does averaging predictions over multiple generated poses recover near-crystal performance, and (3) do protein-ligand interaction distributions shift in ways that explain performance differences.

---

## Repository Structure

```
.
├── performance_comparison/     # Single-pose performance benchmarking
├── multipose/                  # Multi-pose prediction averaging analysis
├── plip/                       # Protein-ligand interaction profiling (PLIP)
└── rmsd/                       # RMSD computation relative to crystal structures
```

---

## Directories

### `performance_comparison/`

Compares PLBAP model performance (Pearson correlation coefficient, PCC; RMSE) across structure sources using paired bootstrap resampling (n = 20,000) with Holm–Bonferroni multiple-testing correction.

| File | Description |
|---|---|
| `fig2-w-bootstrapping.py` | Generates Figure 2: bar charts of per-model PCC by structure source with 95% bootstrap confidence intervals |
| `performance_comparison-Bootstrap_ModelIndependent.py` | Computes model-averaged performance per source and pairwise bootstrap p-values between all source pairs |
| `2026-02-11_CASF2016_Docking-Comparison_All-Results-W-Scores.csv` | **Primary data file.** Raw predictions from all five PLBAP models across all structure sources and poses for the CASF-2016 set. Columns: `pdbid`, `pK_predicted`, `pK_true`, `Model`, `Source`, `pose`, `docking_score` |
| `performance_by_source_modelavg.csv` | Model-averaged PCC and RMSE per structure source |
| `pairwise_pvalues_sources_only_modelavg.csv` | Pairwise bootstrap p-values (Holm-corrected) for all source comparisons |

### `multipose/`

Evaluates whether averaging PLBAP predictions over ranked ensembles of top-N poses (N ∈ {1, 2, 3, 5, 10, 20, 30, 50, 100}) recovers near-crystal performance. Mean and median aggregation strategies are both assessed.

| File | Description |
|---|---|
| `multi-pose_performance.py` | Computes PCC and RMSE for mean/median pose aggregation at each ensemble size N, per model and source; generates PCC vs. N line plots |
| `multi-pose_performance-Bootstrap.py` | Bootstrap resampling (n = 20,000) to compute confidence intervals and pairwise p-values for multi-pose aggregation comparisons |
| `multipose_mean_median_summary.csv` | PCC and RMSE at each N, aggregation method, source, and model |
| `multipose_summary_by_model_bootstrap.csv` | Bootstrap statistics (CI, p-values) per model for multi-pose comparisons |
| `multipose_summary_modelavg_bootstrap.csv` | Bootstrap statistics averaged across models for multi-pose comparisons |
| `multipose_pairwise_sources_by_model.csv` | Pairwise source comparisons per model at each N |
| `multipose_pairwise_sources_modelavg.csv` | Pairwise source comparisons averaged across models at each N |
| `multipose_pcc_mean.png` | Figure 3 (top): PCC vs. ensemble size N for mean aggregation |
| `multipose_pcc_median.png` | Figure 3 (top): PCC vs. ensemble size N for median aggregation |

### `plip/`

Characterizes protein-ligand interaction distributions across structure sources using the Protein-Ligand Interaction Profiler (PLIP). Includes statistical tests comparing interaction count distributions between sources.

| File | Description |
|---|---|
| `plip_distrib-stats.py` | Generates violin plots of per-interaction-type counts by source; computes Kruskal–Wallis tests across sources and Mann–Whitney U pairwise tests; Wilcoxon signed-rank test comparing top-1 vs. top-100 pose distributions |
| `plip_counts_FINAL.csv` | **Primary PLIP data file.** Per-complex, per-pose interaction counts by type (`hydrophobic_interaction`, `hydrogen_bond`, `salt_bridge`, `pi_stacking`, `pi_cation_interaction`, `halogen_bond`) for each source |
| `stats_kruskalwallis_per_inttype.csv` | Kruskal–Wallis test statistics and p-values per interaction type across sources |
| `stats_mannwhitney_pairwise_per_inttype.csv` | Pairwise Mann–Whitney U test results between sources, per interaction type |
| `stats_mannwhitney_total_interactions.csv` | Mann–Whitney U test results for total interaction counts between source pairs |
| `stats_wilcoxon_top1_vs_top100.csv` | Wilcoxon signed-rank test comparing top-1 vs. top-100 pose interaction counts per source |
| `plip_distribs_1pose.png` | Violin plots of interaction distributions for the single best pose per source |
| `plip_distribs_100poses.png` | Violin plots of interaction distributions pooled across all 100 poses per source |
| `plip_distribs_1vs100poses.png` | Overlaid comparison of top-1 vs. top-100 pose interaction distributions |
| `plip_total_interactions_1vs100poses_crystal_full_greywhite.png` | Total interaction count violin plots for top-1 vs. top-100 poses |

### `rmsd/`

Computes structural perturbations of generated structures relative to experimental crystal complexes. Three RMSD metrics are reported: protein Cα RMSD, protein sidechain RMSD, and non-hydrogen ligand heavy-atom RMSD, using sequence alignment to handle receptor differences between sources.

| File | Description |
|---|---|
| `rmsd_unified.py` | Main RMSD computation script. Accepts a protein-ligand structure (PDB/mmCIF) and a crystal reference; performs sequence-alignment-guided Cα superposition; computes Cα, sidechain, and ligand RMSD |
| `rmsd_computation.py` | Supporting functions for protein RMSD: chain identification, sequence-based residue pairing, atom matching by name |
| `rmsd_ligand.py` | Supporting functions for ligand RMSD: ligand atom selection, name-matched position extraction |
| `merge_rmsd_outputs.py` | Merges per-pose RMSD CSV outputs from all sources into a single combined file |
| `submit_rmsd_pipeline.sh` | SLURM array job script for running `rmsd_unified.py` in parallel across all poses and sources on an HPC cluster |

---

## Structure Generation

All five structure sources were generated on the MSU HPCC and applied to the 285 CASF-2016 complexes. Raw structure files are available at [10.5281/zenodo.18701481](https://doi.org/10.5281/zenodo.18701481).

### Crystal
Experimental co-crystal structures were taken directly from the CASF-2016 benchmark set (PDBbind v2016). (Su et al. *J. Chem. Inf. Model.* 2019, 59, 895–913; Wang et al. *J. Med. Chem.* 2005, 48, 4111–4119)

### GNINA Docking (GNINA-Crystal and GNINA-Apo)
Docking was performed with [GNINA v1.3.1](https://github.com/gnina/gnina) (McNutt et al. *J. Cheminform.* 2025, 17, 28) using the [WoldringLabMSU GNINA docking pipeline](https://github.com/WoldringLabMSU/GNINA-Docking-Pipeline). Ligands were specified from OpenEye SMILES strings (RCSB LigandExpo) and converted to SDF with OpenBabel (O'Boyle et al. *J. Cheminform.* 2011, 3, 33). Binding-pocket boxes were defined from PDBbind pocket residue files. GNINA was executed inside a Singularity container with GPU acceleration; up to 1,000 poses were generated per complex and the top 100 retained.

- **GNINA-Crystal**: docking into holo co-crystal receptor structures from CASF-2016.
- **GNINA-Apo**: docking into experimentally resolved apo receptor conformers (aligned to the holo receptor to transfer the binding-site coordinate frame).

### AlphaFold3 Co-folding (AlphaFold3)
Protein-ligand co-folding was performed with AlphaFold3 (Abramson et al. *Nature* 2024, 630, 493–500) using the [WoldringLabMSU AlphaFold3 pipeline](https://github.com/WoldringLabMSU/AlphaFold3-Pipeline). MSAs were generated via ColabFold/MMseqs2 and referenced in per-target AF3 JSON inputs. AF3 inference ran inside a Singularity image on MSU HPCC GPU nodes; 100 predictions were generated per complex.

### GNINA Docking into AlphaFold3 Receptors (GNINA-AF3)
Protein-only AF3 structures were generated (same pipeline as above), aligned to the corresponding crystal receptor to transfer the binding-site coordinate frame, and used as rigid receptors for GNINA docking following the same protocol as GNINA-Crystal.

---

## PLBAP Models

Five models were evaluated in inference-only mode (no retraining) using author-provided pipelines with minimal modifications for batch processing and software-dependency compatibility. Docked/predicted structures were converted to each model's required input format (PDB, MOL2, SDF as needed via OpenBabel) without altering atomic coordinates.

| Model | Citation | Original Repository | Study Fork |
|---|---|---|---|
| Dynaformer | Min et al. *Advanced Science* 2024, 11, 2405404 | [Minys233/Dynaformer](https://github.com/Minys233/Dynaformer) | [jeavesj/Dynaformer](https://github.com/jeavesj/Dynaformer) |
| EGNA | Xia et al. *Briefings in Bioinformatics* 2023, 24, bbac603 | [gnina/EGNA](https://github.com/gnina/EGNA) | [jeavesj/EGNA](https://github.com/jeavesj/EGNA) |
| EHIGN-PLA | Yang et al. *IEEE Trans. Pattern Anal. Mach. Intell.* 2024, 46, 8194–8208 | [guaguaujiaile/EHIGN\_PLA](https://github.com/guaguaujiaile/EHIGN_PLA) | [jeavesj/EHIGN\_PLA](https://github.com/jeavesj/EHIGN_PLA) |
| GIGN | Yang et al. *J. Phys. Chem. Lett.* 2023, 14, 2920–2933 | [guaguaujiaile/GIGN](https://github.com/guaguaujiaile/GIGN) | [jeavesj/GIGN](https://github.com/jeavesj/GIGN) |
| OnionNet-2 | Wang et al. *Front. Chem.* 2021, 9, 753002 | [zehwang/OnionNet-2](https://github.com/zehwang/OnionNet-2) | [jeavesj/OnionNet-2](https://github.com/jeavesj/OnionNet-2) |

All predictions were collected into the standardized CSV described under [Data](#data) below.

---

## Data

The primary input data file (`performance_comparison/2026-02-11_CASF2016_Docking-Comparison_All-Results-W-Scores.csv`) contains model predictions for all CASF-2016 complexes across all structure sources and poses. This file is the input for both the performance comparison and multi-pose analyses.

The PLIP interaction counts (`plip/plip_counts_FINAL.csv`) were generated by running PLIP on all prepared structures and are the input for the PLIP analysis scripts.

Raw structure files (docked poses, AlphaFold3 predictions, preprocessed PDBs) and per-pose RMSD outputs are not included in this repository due to size but were generated on the MSU HPCC and are will be made publicly available upon official publication at 10.5281/zenodo.18701481.

---

## Dependencies

The conda environment used to execute the scripts in this repository is `analysis_environment.yml`.

`conda env create -f analysis_environment.yml`


The SLURM submission script (`submit_rmsd_pipeline.sh`) is specific to the MSU HPCC environment and references internal data paths; it is provided for reproducibility documentation.

---

## Citation

If you use this code or data, please cite:

> Eaves, J. N.; Woldring, D. R. Robustness of Protein-Ligand Binding Affinity Prediction Models to Docked and Predicted Structures. *J. Chem. Inf. Model.* **2026**.

---

## License

This repository is released under CC0 1.0 Universal (public domain dedication). See `LICENSE` for details.

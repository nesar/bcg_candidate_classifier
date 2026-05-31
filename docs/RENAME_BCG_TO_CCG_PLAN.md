# BCG → CCG Terminology Rename Plan (Deferred)

This document captures the planned but **not yet executed** rename of "BCG" to "CCG"
across the codebase, to match the paper's "Cluster Central Galaxy" terminology.

Scope A (probability identifiers — `bcg_prob`/`redmapper_prob`/`candidate_rm_prob`
→ `p_rm`/`p_mem`) has **already been completed** in branch `rename/p_rm_p_mem_ccg`.
Scope B (this document) is deferred due to the scale of changes (~1100 references
across ~40 files).

## Inventory snapshot

A search at the time of the Scope A rename found these unique identifiers
(non-exhaustive):

### Variables (lowercase `bcg_*`)
```
bcg_arcmin_type, bcg_cand, bcg_candidate, bcg_candidates, bcg_clusters,
bcg_clusters_clean, bcg_coord_files, bcg_coords, bcg_csv, bcg_csv_alt,
bcg_csv_clean, bcg_csv_coords, bcg_csv_path, bcg_dataset, bcg_dec,
bcg_deploy, bcg_deployment, bcg_df, bcg_entries, bcg_found, bcg_idx,
bcg_info, bcg_keys, bcg_matched, bcg_not_detected, bcg_pmem,
bcg_pmem_mean, bcg_pmem_median, bcg_pmem_std, bcg_pos, bcg_probs,
bcg_ra, bcg_rank, bcg_rank_by_pmem, bcg_rank_mean, bcg_ranks,
bcg_row, bcg_top, bcg_x, bcg_y
```

### Constants / env names
```
BCG_DATA_DIR, BCG_MATCHED_FILE
```

### Class names (CamelCase)
```
BCGAnalysisRunner, BCGCandidateClassifier, BCGCandidateDataset, BCGDataset,
BCGDatasetManager, BCGEnsembleClassifier, BCGEvaluationAnalyzer,
BCGInference, BCGPathConfig, BCGProbabilisticClassifier
```

### Function names (sample)
```
find_bcg_candidates, create_bcg_datasets, create_bcg_candidate_dataset_from_loader,
prepare_bcg_dataframe (in data_read_bcgs.py)
```

### Dict keys in result/sample dicts
```
'BCG' (in BCGDataset.__getitem__ result)
'all_bcg_candidates' (in test.py metadata)
```

### File names to rename
```
data/data_read_bcgs.py       → data/data_read_ccgs.py
data/candidate_dataset_bcgs.py → data/candidate_dataset_ccgs.py
utils/candidate_based_bcg.py → utils/candidate_based_ccg.py
utils/viz_bcg.py             → utils/viz_ccg.py
create_and_verify_bcg_dataset.py → create_and_verify_ccg_dataset.py
ml_models/candidate_classifier.py (class BCGCandidateClassifier inside)
ml_models/uq_classifier.py (classes BCGEnsembleClassifier, BCGProbabilisticClassifier inside)
```

### Package directory
```
bcg_candidate_classifier/  → ccg_candidate_classifier/
bcg_deployment/            → ccg_deployment/   (separate sub-package)
bcg_deployment/bcg_deploy/ → ccg_deployment/ccg_deploy/
```

## What to NOT rename

Leave the following as-is:

1. **External data CSV column names** in source files:
   - `'BCG RA'`, `'BCG Dec'`, `'BCG Probability'` — these live in the source
     truth-table CSVs and are read by `data_read_bcgs.py:115-120`. Remap them
     to `ccg_ra`/`ccg_dec`/`p_rm` (already done for the last one) on read.

2. **External file paths and filenames**:
   - `bcgs_3p8arcmin_clean_matched.csv`, `bcgs_2p2arcmin_*.csv`
   - `desprior_candidates_*.csv`
   - `/data/bcgs/bcgs/...`
   - `/lcrc/.../BCGs_swing/data/lbleem/bcgs`
   These are paths/filenames on disk owned by external pipelines.

3. **The acronym "BCG" as a noun in documentation/comments** when
   discussing the historical literature (the paper itself notes
   "Brightest Cluster Galaxy" as a misnomer; we use CCG going forward).

## Suggested execution order (when ready)

To minimize broken intermediate states, execute in this order:

1. **Phase 1: Internal variable names + dict keys** (most occurrences,
   no file moves yet). One file at a time, verify imports.
2. **Phase 2: Function names**. Update all call sites in lockstep.
3. **Phase 3: Class names**. Update all instantiations and isinstance checks.
4. **Phase 4: File names**. Rename `.py` files and update `from X import Y`
   statements throughout. Also update string literals that reference
   `from utils.viz_bcg import ...` etc.
5. **Phase 5 (optional): Package directory rename**. Risky because
   external scripts may `cd` into the directory. If done, audit:
   - `bcg_deployment/setup.py` `name=` parameter
   - Any external README/CI/scripts that reference the path
   - The `bcg_deployment` sub-package directory should be renamed too

## Helpful commands

```bash
# Quick audit before starting
grep -rno '\bbcg_[a-zA-Z_]*\|\bBCG[a-zA-Z_]*' --include="*.py" . \
    | grep -v __pycache__ | grep -v backup_old_codes | sort -u

# Verify no regressions after rename
find . -name "*.py" -not -path "*/__pycache__/*" -not -path "*/backup_old_codes/*" \
    -exec python -m py_compile {} \;
```

## Open questions (to resolve before starting)

1. `bcg_coords` / `bcg_ra` / `bcg_dec` → `ccg_*` or `target_*`?
   (These refer to the labeled target's coordinates.)
2. Rename `bcg_deployment/` sub-package directory too?
3. Rename the package root directory `bcg_candidate_classifier/` itself,
   or keep it as a holdover?

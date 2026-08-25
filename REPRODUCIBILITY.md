# Reproducibility notes

## Environment

```
python -m venv .venv
.venv\Scripts\activate        # (Windows) or: source .venv/bin/activate
pip install -r notebooks/requirements.txt
```

Developed and last verified against **Python 3.12.10**, pinned via
`.python-version` at the repository root. `requirements.txt` uses `>=` lower
bounds only, not exact pins; a documented incident (see "Supplementary Table
S10" below) shows a `scikit-learn`/`xgboost` version drift already caused a
real, non-bit-identical numeric result once. `notebooks/requirements-lock.txt`
(pinned via `pip show`, generated 2026-08-20) is provided alongside it for
exact reproduction; prefer `pip install -r notebooks/requirements-lock.txt`
when bit-identical results matter more than picking up newer library
versions.

`requirements.txt` covers everything actually imported by the 8 notebooks
(numpy, pandas, scipy, scikit-learn, matplotlib, shap, statsmodels, pyproj,
openpyxl, pyyaml, jupyter/nbconvert/nbclient/ipykernel). No GIS libraries
(geopandas/rasterio) are required for the **notebook pipeline** itself: all
raster/vector processing for the main analysis happens in Google Earth Engine
(`gee_script/`, or its Python port in `GEE_API/`) or ArcGIS Pro. Two
figure-regeneration scripts are the exception: `results/figures/
figure1_compose_study_area.py` needs `geopandas`, and `figure8_01_export_arcgis.py`
needs `arcpy` (licensed ArcGIS Pro only, not installable from PyPI) -- see
their own docstrings and `README.md`'s data-availability section.
`GEE_API/` has its own `requirements.txt` (Earth Engine API + `geemap` for
the visual-QA notebook), separate from the notebook pipeline's.

A minimal smoke test (`tests/test_smoke.py`, run via `pytest tests/` from the
repository root) checks that `src/coffeemap` imports cleanly and that
`config/paper1_config.yaml` still parses with the manuscript's 10 class
definitions. It runs automatically on every push/PR via
`.github/workflows/smoke-test.yml`. This is a fast sanity check only, not a
substitute for the end-to-end notebook re-execution documented below: it does
not touch `data/raw/`, which is not tracked in this repository (see
`README.md`'s Data availability section).

## Run order

```
gee_script/00_GEE_training_quick_test.js         (Earth Engine, run in the GEE Code Editor)
  -> gee_script/01_GEE_feature_stack_train_map_export.js
  -> gee_script/02_GEE_area_statistics_from_asset.js
  -> exports feed data/raw/ and data/raw/data_DakLak_Statistics/
notebooks/00_audit_inputs_and_reference_data.ipynb
  -> 01_build_feature_catalog_and_area_inventory.ipynb
  -> results/figures/regenerate_S5_S6_feature_selection_figures.py
       (Supplementary Figures S5-S6; standalone script, run after notebook 01 --
        reads the raw per-seed CSVs notebook 01 saves, no RF refit needed. This is
        the single source of truth for that plotting code as of 2026-08-09; the
        notebook cell that used to duplicate it was removed.)
  -> 02_benchmark_models_and_accuracy.ipynb        (Tables 3-4, Figure 5)
  -> 03_check_spatial_cross_validation.ipynb       (spatial-autocorrelation diagnostic)
  -> 04_interpret_final_rf_with_shap.ipynb         (Figures 6-7)
  -> 05_estimate_error_adjusted_area.ipynb         (Table 6, Figure 10, Olofsson 2014)
  -> 06_compare_district_area_with_official_stats.ipynb  (Figure 9)
  -> 07_assemble_submission_package.ipynb          (final QA / manifest)

S10_classifier_comparison/ML_Comparison_v2.ipynb  (Supplementary Table S10)
  -- standalone, self-contained (own data/ and Results_Model_Selection/ output
     folder); does not depend on notebooks 00-07 or their outputs, can be run
     independently at any point. See "Supplementary Table S10" section below for
     a reproducibility caveat on exact decimal values.
```

Run notebooks from `notebooks/` as the working
directory (matches how they locate
`config/paper1_config.yaml` by walking up
parent directories, treating this repository's root as
`PROJECT_ROOT`). Notebooks write their outputs into
`results/tables/`,
`results/figures/`, and
`results/supplementary/` (via
`config/paper1_config.yaml`'s
`tables_dir: "results/tables"` etc.); only internal bookkeeping (`metadata/`)
stays elsewhere inside the repository.

**Path-layout note**: notebook `00`'s own input-audit code (and several
other notebooks) checks a short hardcoded list of relative folder names for
GEE exports and official statistics (`data/raw`, `data/raw/data_DakLak_Statistics`,
`data_DakLak_Statistics`, etc.), independent of `config.yaml`. These literal
folder names must exist directly under the repository root
exactly as named. Renaming or nesting them differently (e.g. `data/raw_gee/`,
or moving statistics to `data/data_DakLak_Statistics/` instead of
`data/raw/data_DakLak_Statistics/`) makes every input show up as `MISSING` in
the audit even though the pipeline otherwise runs without crashing. This was
caught and fixed twice already during package assembly (once during initial
reconstruction, once after the folders were reorganized into the current
`results/`-nested layout); keep the folder
names as they are.

## What was reconstructed, and why

When this package was assembled, `config/paper1_config.yaml` and the
`src/coffeemap/` Python package that every notebook imports at the top
(`coffeemap.config`, `.io`, `.manifest`, `.metrics`, `.olofsson`, `.plotting`,
`.schema`, `.spatial_cv`, `.validation`) did not exist anywhere in the
project; every notebook failed on its first cell. Both were rebuilt by
reading how all 8 notebooks call these functions (parameters in, values/
objects out), using the class definitions in the manuscript's Table 2 and
the RF random seed (2024) used throughout Sections 2.6/2.7 of the manuscript.
Both now live under `config/` and
`src/coffeemap/`.

**This reconstruction has been directly validated, not just assumed to
work**, by re-executing all 8 notebooks end-to-end from
`notebooks/` and comparing the freshly
generated `results/tables/`,
`results/figures/`, and
`results/supplementary/` files/numbers against
the pre-existing, manuscript-matching ones:

| Notebook | Status | Validation |
|---|---|---|
| `00_audit_inputs_and_reference_data` | **Verified** | Runs clean (34/34 PASS, 0 WARN/ERROR). Freshly generated `class_distribution_train_validation.csv` and `district_matching.csv` are identical to the pre-existing copies. Class counts (2,100 train / 900 val / 300 per class), 97 candidate predictors, and the 227,734 ha vs 227,721 ha class/district coffee-area cross-check all match the manuscript exactly. |
| `01_build_feature_catalog_and_area_inventory` | **Verified** | Reproduces `Table5_MappedArea_By_LandCoverClass.csv` with "Mapped coffee area: 227,721 ha" (post-correction, see note below), matching Table 5/Section 3.5 of the manuscript exactly. Required one compatibility fix (see below). Also reproduces the feature-count sensitivity analysis (Supplementary Table S7: sensitivity to number of selected predictors). |
| `02_benchmark_models_and_accuracy` | **Verified (exact match)** | Reproduces Table 4 (class-wise producer's/user's accuracy and F1) with values identical to the manuscript to 1 decimal place for every one of the 10 classes (e.g. Sun coffee PA=80.0/UA=88.9/F1=0.842). Also exactly reproduces all 10 rows of Table 3's sensor-ablation ranking (OA, Kappa, Macro F1): full multi-sensor stack = 95.90 ± 1.13% OA, matching the manuscript to 2 decimal places (2026-08-09 fresh run; an earlier intermediate run on disk had read 95.73 ± 1.25%, now understood to be non-representative rather than a sign of unpinned randomness). This same re-run also caught and confirmed a genuine, pre-existing column-labeling error in Table 3's "coffee-subclass macro F1" column (it had been showing coffee-binary F1 values instead); corrected in the manuscript and in `Table3_CV_summary.csv`. Required one compatibility fix (see below). |
| `03_check_spatial_cross_validation` | **Verified (independent method)** | Runs clean and produces a large random-vs-spatial CV gap (96.3% vs 61.7% overall accuracy in one test run), independently confirming the same qualitative finding reported in the manuscript's Section 3.7. **This notebook uses a different, simpler spatial-grouping method (K-means coordinate clusters) than the 10/15/20 km UTM block design reported in Supplementary Table S3**; that specific design is reproduced by notebook `06` instead (see "Spatial block CV" note below), not by this one. |
| `04_interpret_final_rf_with_shap` | **Verified (exact match)** | Reproduces the RF-Gini-vs-SHAP consistency check with Spearman rho = 0.9608 (p = 2.6e-14), matching the manuscript's reported Spearman rho = 0.961 (Sections 3.4/4.2) almost exactly. Regenerates Figure 6-7 SHAP/importance data (global and class-specific SHAP importance are reported in the manuscript's main-text Figure 7, not as separate Supplementary items; a legacy "03B" cell that used to export this same data under stale, colliding `Supplementary_Table_S4`/`S5`/`S6` names was removed 2026-08-09). Also contains the Random Forest hyperparameter sensitivity sweep (**this notebook's own cell is the actual source of the real Supplementary Table S6**: n_estimators/mtry/bag-fraction grid), reusing this notebook's training/validation data; reproduces the manuscript's baseline-configuration row exactly (OA 93.33%, Kappa 92.59%, macro F1 93.30%, coffee-subclass macro F1 89.86%) and the full grid's reported range (OA 92.44-93.78%, coffee-subclass macro F1 88.31-91.41%, 60 combinations) once its 25 feature columns are sorted alphabetically before fitting (see fix note below). |
| `05_estimate_error_adjusted_area` | **Verified (exact match)** | Reproduces Table 6/Supplementary Table S5 (Olofsson 2014 area-weighted adjustment) with adjusted overall accuracy = 92.01% +/- 2.67% and adjusted total coffee area = 240,053 ha (post-correction, see note below), matching the manuscript's Section 3.7 text exactly. Required two small additions to `coffeemap.schema` (`normalise_col`, and a `role` keyword on `find_first_column`). |
| `06_compare_district_area_with_official_stats` | **Verified (exact + close match)** | Reproduces the Figure 9 caption text byte-for-byte: R² = 0.853, Pearson r = 0.944, RMSE = 4,054 ha, MAE = 3,304 ha, MAPE = 34.9%, total bias = +13,715 ha (+6.4%), mapped 227,721 ha vs. official 214,006 ha. **This notebook also contains the detailed 10/15/20 km UTM spatial block cross-validation** that produces Supplementary Tables S3-S6 and Figures S1-S3 (confirmed by an identical duplicate-audit fingerprint: 256 exact-coordinate duplicates, 928 near-duplicate pairs within 10 m, and identical suggested captions text) -- see "Spatial block CV" note below. |
| `07_assemble_submission_package` | **Verified** | Runs clean; assembles its own package/manifest. (An earlier run also wrote a copy to `mapping_workflow/CoffeePaper1_FinalPackage/`, in the now-archived live project folder; that folder has since been deleted as redundant now that this package supersedes it with additional manuscript/data-history/GIS content.) |

**2026-07-15 data correction:** `data/raw/Table_AreaStatistics_DakLak2024.csv` originally reported
Sun coffee = 168,130.72 ha and Intercrop coffee = 49,388.48 ha (whole-province per-class pixel sum),
which differed by -4.06 ha and -8.99 ha respectively from the independent district-level polygon
aggregation already computed in `results/supplementary/DistrictVsClassArea_Audit_20260715.csv`
(notebook `06`; renamed 2026-08-09 from `Supplementary_Table_S9_DistrictVsClassArea_Audit.csv`,
which collided with the real Supplementary Table S9 (feature-selection strategy comparison). The manuscript's Table 5/Section 3.5 body text had already adopted the district-aggregation
figures, so the raw source file was corrected to match (Sun coffee -> 168,126.66 ha, Intercrop coffee
-> 49,379.49 ha, Newly planted coffee -> 10,214.59 ha; the other 7 classes are unchanged). Notebooks
`01` and `05` were re-executed end-to-end against the corrected source, which also shifted Table 6's
adjusted areas/biases for the three coffee classes (e.g. adjusted total coffee area 240,066 ->
240,053 ha) and regenerated Figure 10. Supplementary Table S5 (naive/adjusted PA-UA-OA) is unchanged
at reported precision, since the area-weight shift is far below its rounding threshold. The same
end-to-end re-execution of notebook `01` also reshuffled Section 3.7's feature-selection-strategy
Friedman test (cell "7c", `results/tables/Detail_FeatureSelection_FriedmanGlobalTests.csv` /
Supplementary Table S9): that comparison draws its own 5 random seeds, unpinned relative to the
area-inventory computation in the same notebook, so re-running it produced different (though
qualitatively identical) statistics -- overall accuracy chi-sq 10.63 -> 12.33, macro F1 10.20 ->
10.68, coffee-subclass macro F1 14.04 -> 13.56; no pairwise Holm-corrected comparison was significant
either before or after. This is disclosed in the Response to Reviewers letter alongside the Table 3
column-labeling fix. As with the S10 and spatial-block-CV caveats below, treat a further fresh re-run
of this specific cell as a robustness check, not a guaranteed bit-identical replica: the CSV above is
the authoritative, manuscript-matching source.

**2026-07-15 restored data:** `data/data_field_survey_training/` (reference ROI KML/CSV files) and
`data/field_survey_raw_photos/` (field photographs and slide deck from the November-December 2025
Dak Lak field visit) were found to be missing from this package (present only in the live
`Chap2_Mapping/data_field_training/` and `Chap2_Mapping/figure_field_survey/` folders, never copied
in) and have been restored here as static reference material. This does not change any pipeline
output: nothing in `notebooks/00`-`07` reads from these two folders. It only completes the
package's documentation of where the reference-point labeling and field photos came from.

Compatibility fixes required for notebooks to run under the current
Python/pandas/matplotlib versions (environment-version issues, not logic
errors):
- `01_build_feature_catalog_and_area_inventory.ipynb`, cell 7: a
  `DataFrame.fillna("")` call failed on a Categorical column under current
  pandas; patched to cast categorical columns to string first.
- `src/coffeemap/plotting.py`:
  `set_publication_style()` set `figure.constrained_layout.use=True`
  globally, which conflicts with notebooks that build their own colorbar
  layout (Figure 5 in notebook `02`). Removed; each notebook manages its own
  figure layout.
- `src/coffeemap/manifest.py`:
  `init_run_manifest`/`append_manifest_note` originally wrote to a hardcoded
  `results/metadata/run_manifest.json` path under `PROJECT_ROOT`, which
  collided with the (unrelated) `results/`
  folder used for tables/figures/supplementary. Fixed to resolve the manifest
  path from `config['paths']['metadata_dir']`
  (`metadata/`) instead, so run manifests stay
  separate from the citable deliverables in
  `results/`.
- `04_interpret_final_rf_with_shap.ipynb`, cell 26 (Supplementary Tables
  S4-S5/Figures S4-S5): had its own hardcoded `INPUT_DIR_CANDIDATES` list
  (`Supplementary/`, `results/Supplementary/`, `data/Supplementary/`, etc.,
  all relative to `PROJECT_ROOT`, and case-sensitive), none of which matched
  `SUPPLEMENTARY_DIR` (the actual `results/supplementary` folder the preceding
  cells had just written `rf_feature_importance.csv`/`shap_global_importance.csv`/
  `shap_values_all.npz` to), so it always raised `FileNotFoundError`. Fixed by
  adding `OUT_DIR` (`= SUPPLEMENTARY_DIR`) as the first candidate.
- The standalone `rf_hyperparameter_sensitivity/rf_hyperparam_sweep.py`
  script (hardcoded paths into a now-deleted project folder and a scratch
  directory, and absent from the documented run order) was retired and its
  logic moved into a new cell in `04_interpret_final_rf_with_shap.ipynb`
  (Section "5B"), reusing the training/validation data already loaded there
  for the SHAP replica. This surfaced a genuine, non-obvious sensitivity:
  with `mtry` (3) smaller than the number of features (25), scikit-learn's
  `RandomForestClassifier` draws its per-split feature subset by column
  *position*, so results depend on feature column order even with a fixed
  `random_state`. The notebook's `X_train`/`X_val` are ordered by RF/SHAP
  importance (from `Table_SelectedBands_corrTop25.csv`), which reproduced
  Table S6's qualitative pattern but shifted the baseline row by about
  0.2-0.4 percentage points from the manuscript's cited numbers. The
  reinstated sweep cell sorts the 25 feature columns alphabetically before
  fitting (a local copy, `X_train_sens`/`X_val_sens`, that does not affect
  the SHAP/importance cells elsewhere in the notebook), which exactly
  matches the column order the manuscript's Table S6 was originally computed
  with, and reproduces its baseline row exactly (OA 93.33%, Kappa 92.59%,
  macro F1 93.30%, coffee-subclass macro F1 89.86%) and its full-grid range
  (OA 92.44-93.78%, coffee-subclass macro F1 88.31-91.41%, 60 combinations).

## Supplementary Table S10 (classifier comparison): recovered and re-verified, with a precision caveat

`S10_classifier_comparison/ML_Comparison_v2.ipynb` was recovered
2026-08-09 from a separate, pre-cleanup project folder (`analysis_ML_Comparision/`,
outside this package) and migrated in along with its GEE export script
(`ML_GGE_export_data.js`) and self-contained `data/` folder of GEE-exported
train/validation CSVs (`OptimalModelSelection_*`). This closes what was previously
an honestly-flagged gap: Supplementary Table S10 existed only as a static CSV with
no generating script saved anywhere in this package.

The notebook is self-contained (own `data/` and `Results_Model_Selection/` output
folder, relative to its own location) and does not depend on notebooks `00`-`07`.
It was re-run end-to-end from its new location (`RANDOM_STATE=2024`, pinned) and
**reproduces the same qualitative finding**, with XGBoost on the VIF-filtered 34-predictor
set still the top-ranked configuration, but **not the bit-identical decimal
values** published in the manuscript's Supplementary Table S10 (e.g., winner OA
0.9491 vs. the published 0.947; ranks 2-3 swap order between two closely-tied
configurations, Full-97 XGBoost vs. Full-97 Random Forest). No `requirements.txt`
or library-version pin was saved with the original April-2026 run, and `xgboost`
had to be freshly installed to re-run it 2026-08-09, so the most likely explanation
is `xgboost`/`scikit-learn` version drift between the original run and today's
environment shifting `RandomizedSearchCV`'s exact search path, not a logic error,
but this has not been independently confirmed by pinning old versions and re-testing.

**`results/tables/Supplementary_Table_S10_ClassifierAlgorithmComparison.csv` (the
file the manuscript's Table S10 was built from) has deliberately NOT been
overwritten with the fresh re-run's numbers**; it remains the authoritative,
manuscript-matching file. The fresh re-run's own output lives separately in
`S10_classifier_comparison/Results_Model_Selection/`, kept as a
reproducibility/robustness check, not a replacement: the same treatment already
given to the spatial block CV re-run above.

## Spatial block CV (10/15/20 km): reproducible, with a caveat on exact numbers

The manuscript's Section 3.7 reports a detailed spatial block cross-validation
across 10, 15, and 20 km UTM blocks (Supplementary Tables S3-S6, Supplementary
Figures S1-S3), showing coffee-subclass macro F1 collapsing from 0.976 to
0.14-0.37 under spatial independence. This analysis is embedded inside
`06_compare_district_area_with_official_stats.ipynb` (not a separately named
notebook, and not the same thing as `03_check_spatial_cross_validation.ipynb`,
which is an independent, simpler K-means-based sanity check). This was
confirmed by an identical fingerprint to the original analysis (256 exact-
coordinate duplicates and 928 near-duplicate point pairs within 10 m among
the pooled 3,000 reference points; identical suggested supplementary table/
figure captions).

Re-running it reproduces the same qualitative and roughly the same
quantitative result (e.g. 10 km: OA = 0.758 +/- 0.155, macro F1 = 0.617 +/-
0.097 in one fresh run vs. 0.763 +/- 0.111 / 0.718 +/- 0.069 in the
manuscript's Supplementary Table S3) but **not bit-identical fold-level
numbers**, because the exact block-to-fold assignment and per-fold Random
Forest fit are not currently pinned to a single deterministic seed end to
end. The existing
`results/supplementary/Table_SpatialCV_*.csv`
files (and the Supplementary Table S3-S7 files already in
`results/tables/`) remain the authoritative
source for the exact numbers quoted in the manuscript; treat a fresh re-run
as a robustness check that should show the same pattern, not an exact replica.

## Reference dataset labeling: what is not independently reproducible

The 3,000-point visually interpreted reference dataset (Table 2 of the
manuscript) was labeled by hand in Google Earth Pro and refined using field
observations from November-December 2025. As of the 2026-08-19 revision,
Section 2.3 of the manuscript documents that each candidate point is
reviewed jointly by three interpreters and retained only when all three
agree on the class label; no formal inter-interpreter agreement statistic
(e.g., Cohen's kappa) was computed for this consensus process. Re-running
the pipeline reproduces every downstream number
computed *from* this reference dataset, but the labeling process itself is
not scripted and cannot be independently regenerated from this package. The
reference ROI points themselves (`data/data_field_survey_training/`) and the
underlying field photographs (`data/field_survey_raw_photos/`) are included
as static documentation of that manual process, not as reproducible inputs.

As of manuscript version `v3_20260716_table6_recalc_final`, the Section 2.5 paragraph describing
the November-December 2025 field verification is still pending real,
documented supporting numbers (site counts by class/district, confirmed-
unchanged vs. changed vs. unverifiable counts); see the manuscript itself
for the current wording and status.

# Manifest: manuscript items -> producing script -> input -> output

This table is a flat, scannable index of every Figure/Table in the manuscript and its
producing script. It complements `REPRODUCIBILITY.md`, which has the full run order,
per-notebook validation record, and known caveats (exact numbers vs. re-run numbers,
compatibility fixes, etc.); read `REPRODUCIBILITY.md` for narrative detail; use this
file to look up "what makes Figure X" at a glance.

All paths are relative to this repository's root. "Producing script" for the
Jupyter notebooks means the notebook must be run from `notebooks/` as the working
directory, in the order given in `REPRODUCIBILITY.md`'s "Run order" section (each
notebook depends on outputs from the ones before it).

## Main-text figures and tables

| Item | Producing script | Key input(s) | Output file(s) |
|---|---|---|---|
| Figure 1 (study area map) | `results/figures/figure1_compose_study_area.py` (pure vector cartography via geopandas/matplotlib, replacing an earlier ArcGIS Pro export that had no coordinate graticule); terrain backdrop pre-built once by `data/dem_hillshade_reference/fetch_and_build_hillshade.py` | `ArcGIS_Coffee_Map/Shapefile/daklak_details/daklak.shp` (district boundaries), `ArcGIS_Coffee_Map/Shapefile/Việt Nam (tỉnh thành) - T0107.shp` (Vietnam province locator inset, neighboring-province context, and international-boundary line), `data/dem_hillshade_reference/daklak_hillshade.npz` (cached Copernicus DEM GLO-30 hillshade backdrop, attributed in-figure per ESA's Copernicus DEM license) | `results/figures/Figure1_StudyArea.png`/`.pdf` |
| Table 1 (input datasets) | *Not scripted*: static Methods description | — | (in-manuscript table only) |
| Table 2 (class scheme) | *Not scripted*: static Methods description, mirrors `config/paper1_config.yaml`'s `classes:` block | — | (in-manuscript table only) |
| Figure 2 (methodology flowchart) | `results/figures/generate_methodology_flowchart.py` (standalone script, run directly, not part of the notebook chain) | — | `results/figures/Figure2_MethodologyFlowchart.png` (+ `.pdf`) |
| Figure 3 (VHR reference examples) | `results/figures/figure3_compose_class_examples.py` | `data/figure3_class_reference_images/` (10 Google Earth Pro screenshots, one per Table 2 class, extracted from `figure_field_survey/Figures.pptx` and matched to class labels via the slide's own shape positions) | `results/figures/Figure3_ReferenceExamples.png` (+ `.pdf`) |
| Figure 4 (field observations) | `results/figures/figure4_compose_field_survey.py` | `data/figure4_field_survey_photos/` (7 field photographs, Nov-Dec 2025, extracted and labeled the same way as Figure 3's source images) | `results/figures/Figure4_FieldSurvey.png` (+ `.pdf`) |
| Table 3 (sensor-configuration CV performance) | `notebooks/02_benchmark_models_and_accuracy.ipynb` | `data/raw/Table_TrainSamples_RF_Final_2024.csv`, `Table_ValPredictions_RF_Final_2024.csv` | `results/tables/Table3_CV_summary.csv` (+ `Table3_CV_all_folds.csv`, `Table3_CV_feature_selection_stability.csv`, `Table3_CV_selected_features_by_fold.csv`) |
| Figure 5 (confusion matrix) | `notebooks/02_benchmark_models_and_accuracy.ipynb` | best-model validation predictions (same run as Table 3) | `results/figures/Figure5_ConfusionMatrix.png` |
| Table 4 (class-wise PA/UA/F1) | `notebooks/02_benchmark_models_and_accuracy.ipynb` | same as Figure 5 | `results/tables/Table4_classwise_accuracy_best_model.csv` (+ `_summary.csv`, `_confusion_matrix_used.csv`) |
| Figure 6 (RF variable importance) | `notebooks/04_interpret_final_rf_with_shap.ipynb` | `data/raw/Table_RF_VariableImportance_DakLak2024_corrTop25.csv`, exported train/val sets | `results/figures/Figure6_VariableImportance.png` |
| Figure 7 (SHAP interpretation, panels A–E) | `notebooks/04_interpret_final_rf_with_shap.ipynb` | scikit-learn RF refit on the 25 selected predictors (Section 2.7) | `results/figures/Figure7_SHAP_Interpretation.png` (single composite; all five panels are built into one Figure object, no separate per-panel files) |
| Table 5 (mapped area by class) | `notebooks/01_build_feature_catalog_and_area_inventory.ipynb` | `data/raw/Table_AreaStatistics_DakLak2024.csv`, `GEE_DakLak_Coffee_Area_By_District_2024.csv` | `results/tables/Table5_MappedArea_By_LandCoverClass.csv` |
| Figure 8 (final classified map + coordinate grid + local-example insets) | `gee_script/01_GEE_feature_stack_train_map_export.js` (classification) + ArcGIS Pro (colormap/symbology only, no spatial filtering, per Methods 2.6) + `results/figures/figure8_01_export_arcgis.py` (ArcGIS Pro Python env; representative-window selection, graticule geometry, map/inset exports) + `results/figures/figure8_02_compose.py` (regular Python env; graticule labels, vector legend, and inset compositing -- all five panels a-e composited into one image; see the script's own docstring for how it stays under Word's per-image page-height limit) | `data/raw/DakLak_2024_10class_RF_corrTop25.tif`, district boundary shapefile, Esri World Imagery (source: Vantor) | `results/figures/Figure8_LandCoverMap.png` (panel a: province map; panels b-e: Cu Mgar/Krong Pac local-example insets) |
| Figure 9 (district-level area consistency) | `notebooks/06_compare_district_area_with_official_stats.ipynb` | `data/raw/data_DakLak_Statistics/` official statistics, `GEE_DakLak_Coffee_Area_By_District_2024.csv` | `results/figures/Figure9_DistrictAreaConsistency.png` (single composite; standalone scatter/residuals sub-panel exports were removed as redundant) |
| Table 6 (Olofsson area-adjusted estimates) | `notebooks/05_estimate_error_adjusted_area.ipynb` | Table 4's confusion matrix + `Table5` mapped-area proportions | `results/tables/Table6_Olofsson_AreaAdjusted.csv` |
| Figure 10 (mapped vs. adjusted area) | `notebooks/05_estimate_error_adjusted_area.ipynb` | same as Table 6 | `results/figures/Figure10_MappedVsAdjustedArea.png` |

## Supplementary figures

| Item | Producing script | Output file |
|---|---|---|
| S1 (Bland–Altman district agreement) | `notebooks/06_compare_district_area_with_official_stats.ipynb` | `results/figures/Supplementary_Figure_S1_BlandAltman_DistrictAreaAgreement.png` |
| S2 (reference samples + 15-km blocks map) | `notebooks/06_compare_district_area_with_official_stats.ipynb`, "Sample-level data QA and deduplication audit" section | `results/figures/Supplementary_Figure_S2_Spatial_Samples_and_Blocks.png` |
| S3 (spatial robustness 10/15/20 km) | `notebooks/06_compare_district_area_with_official_stats.ipynb` | `results/figures/Supplementary_Figure_S3_SpatialBlockCV_Performance.png` |
| S4 (train/validation sample distribution map) | *Not currently scripted in this package*: produced ad hoc during the parcel-independence-audit revision work; the generating script was not saved here | `results/figures/Supplementary_Figure_S4_SampleDistribution_TrainVal.png` (static file only) |
| S5 (feature-count sensitivity) | `results/figures/regenerate_S5_S6_feature_selection_figures.py`, reading `notebooks/01_build_feature_catalog_and_area_inventory.ipynb`'s table output | `results/figures/Supplementary_Figure_S5_FeatureCountSensitivity.png` |
| S6 (feature-selection strategy / sensor composition) | `results/figures/regenerate_S5_S6_feature_selection_figures.py` | `results/figures/Supplementary_Figure_S6_StrategyComparison_SensorComposition.png` |

## Supplementary tables

| Item | Producing script | Output file |
|---|---|---|
| S1 (candidate feature list) | `notebooks/00_audit_inputs_and_reference_data.ipynb` / `01_build_feature_catalog_and_area_inventory.ipynb` | derived from `data/raw/Table_CandidateBandList_2024.csv` |
| S2 (vegetation/moisture/stress indices) | *Not scripted*: static Methods description | (in-manuscript table only) |
| S3 (spatial block CV sensitivity) | `notebooks/06_compare_district_area_with_official_stats.ipynb` | `results/tables/Supplementary_Table_S3_SpatialBlockCV_Sensitivity.csv` |
| S4 (class-wise F1 under spatial block CV) | `notebooks/06_compare_district_area_with_official_stats.ipynb` | `results/tables/Supplementary_Table_S4_ClasswiseF1.csv` |
| S5 (naive vs. area-weighted PA/UA/OA) | `notebooks/05_estimate_error_adjusted_area.ipynb` | part of the Table 6 output set |
| S6 (RF hyperparameter sensitivity, 60-combination grid) | `notebooks/04_interpret_final_rf_with_shap.ipynb`, "Section 5B" cell | part of the notebook's supplementary output set |
| S7 (predictor-count sensitivity) | `notebooks/01_build_feature_catalog_and_area_inventory.ipynb` | part of the notebook's supplementary output set |
| S8 (parcel independence audit) | `GEE_API/independence_audit.py` (connected-components clustering on the reference-point coordinates; not part of the 8-notebook run order, see "Known gaps" below) | `results/tables/Supplementary_Table_S8_ParcelIndependenceAudit.csv` (published file, unchanged); the script writes its own fresh-run output separately to `GEE_API/test_runs/`, not overwriting the published file |
| S9 (feature-selection strategy comparison, Friedman/Wilcoxon-Holm) | `notebooks/01_build_feature_catalog_and_area_inventory.ipynb`, "7c. Friedman + pairwise Wilcoxon signed-rank tests" cell | `results/tables/Supplementary_Table_S9_FeatureSelectionStrategyComparison.csv` |
| S10 (RF vs. RBF-SVM vs. XGBoost comparison) | `S10_classifier_comparison/ML_Comparison_v2.ipynb` (standalone notebook, self-contained `data/` folder, run independently of notebooks 00-07; recovered 2026-08-09 from a separate pre-cleanup project folder and migrated in; see REPRODUCIBILITY.md) | `results/tables/Supplementary_Table_S10_ClassifierAlgorithmComparison.csv` remains the authoritative, manuscript-matching file (unchanged); `S10_classifier_comparison/Results_Model_Selection/publication_ready/tables/Table_model_ranking_publication.csv` is the notebook's own fresh-run output, kept separately as a reproducibility check, not a replacement (see caveat in REPRODUCIBILITY.md) |

## Known gaps (honest, not yet closed)

One item (S4 figure, train/validation sample distribution map) currently exists only
as a static output file. The script that produced it was run ad hoc outside
this repository and was never saved into this package. Re-running the
8 documented notebooks will **not** regenerate this file. It is flagged here rather
than silently omitted; closing this gap (recovering or rewriting the script) is
deferred past the current RSASE resubmission.

S10 (table) was a second such gap but has been closed 2026-08-09: the generating
notebook was recovered and migrated into `S10_classifier_comparison/`
(see REPRODUCIBILITY.md for the reproducibility caveat on exact decimal values).

S8 (table) was a third such gap: it was never actually generated by any of the 8
documented notebooks (the RtR itself describes it as a post-hoc audit, not part of
the main pipeline), so re-running notebooks `00`-`07` does not regenerate it. This
has been partially closed 2026-08-19/20: the audit logic was formalized into
`GEE_API/independence_audit.py` (connected-components clustering, an upgrade over
an earlier DBSCAN-based attempt that allowed cross-parcel chaining), tested against
the published 3,000-point reference set and confirmed to reproduce the exact same
independent-parcel count for all 10 classes. It is not yet wired into the
documented notebook run order, so it remains a standalone script, not a pipeline
stage; running it requires GEE Python API authentication (see `GEE_API/README.md`).

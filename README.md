# Interpretable multi-sensor mapping of coffee production systems in Dak Lak, Vietnam

This is the code/pipeline repository for the manuscript *"Interpretable
multi-sensor mapping of coffee production systems using machine learning in
Dak Lak, Vietnam"*: a Random Forest / SHAP workflow that integrates
Sentinel-1, Sentinel-2, Landsat 8/9, and SRTM-derived topographic predictors
to map coffee production systems (sun coffee, intercrop coffee, newly planted
coffee) and seven other land-cover classes across Dak Lak Province for 2024.
Submitted to *Remote Sensing Applications: Society and Environment* (Elsevier),
manuscript number **RSASE-D-26-02087**, currently under revision.

## Repository scope and data availability

This repository is released under the MIT License (see `LICENSE`) and contains
the **code and pipeline only**: GEE scripts, a Python port of the GEE pipeline
(`GEE_API/`), the notebook-based analysis pipeline, its helper package
(`src/coffeemap/`), configuration, and non-identifying summary results
(metrics tables, figures, diagnostic plots).

It does **not** include:
- Raw reference-point coordinates, field photographs, or the final classified
  rasters (excluded per `.gitignore`; available from the corresponding author
  on reasonable request, consistent with the manuscript's Data Availability
  statement).
- `ArcGIS_Coffee_Map/` (district/province shapefiles, the classified raster,
  and an ArcGIS Pro `.aprx` project), required only to regenerate Figure 1
  and Figure 8 (`results/figures/figure1_compose_study_area.py` and
  `figure8_01_export_arcgis.py`). This is a sibling folder outside this
  repository, not tracked here; also available on request. Figure 8's script
  additionally requires a licensed ArcGIS Pro installation (`arcpy`, not
  installable from PyPI) and a signed-in ArcGIS Online session for its
  imagery basemap.
- Manuscript drafts and reviewer correspondence, kept in a separate,
  non-public location.

## Folder guide

| Folder | Contents |
|---|---|
| `gee_script/` | Google Earth Engine JavaScript scripts: feature-stack build, RF train/classify/export, area-statistics export. Run manually in the GEE Code Editor. |
| `GEE_API/` | Python port of the GEE pipeline via the Earth Engine Python API (feature extraction, RF train/classify/export, a formalized parcel-independence audit, and a points-CSV merge/correction tool), so the same steps can run from a local Python environment instead of the GEE Code Editor. See `GEE_API/README.md`. |
| `notebooks/` | The 8-notebook Python pipeline (`00`-`07`) that turns the GEE exports into every table/figure in the manuscript. See `REPRODUCIBILITY.md` for run order. |
| `data/raw/` | GEE-exported training/validation sample tables, area statistics, and validation predictions that feed the Python pipeline. |
| `data/raw/data_DakLak_Statistics/` | Official district/provincial coffee-area and land-use statistics (raw input). |
| `data/data_field_survey_training/` | Reference ROI points (KML/CSV) used for visual interpretation and labeling. |
| `data/field_survey_raw_photos/` | Field-survey photographs and the slide deck built from them (November-December 2025 field visit). |
| `data/gis_outputs/` | The final classified raster, coffee mask, and colormap produced in ArcGIS Pro. |
| `data/processed/` | Intermediate pipeline artifacts (currently empty: notebooks read/write `data/raw/` and `results/` directly). |
| `config/`, `src/coffeemap/` | Shared configuration and helper Python package imported by every notebook. |
| `results/tables/`, `results/figures/`, `results/supplementary/` | Every table/figure cited in the manuscript, plus pipeline QA/audit outputs, already generated and checked in so the manuscript can be verified without re-running anything. Re-running the notebooks regenerates the same files in place. |
| `metadata/` | Internal pipeline bookkeeping (run manifests), not itself a citable deliverable. |
| `S10_classifier_comparison/` | Self-contained notebook + GEE export script for Supplementary Table S10 (RF vs. RBF-SVM vs. XGBoost comparison), independent of the main 8-notebook pipeline. |

See `MANIFEST.md` for the exact producing script/input/output for every
individual figure and table.

## Quick start

See `REPRODUCIBILITY.md` for the exact run order, environment setup, and a
transparent account of what has been re-verified by re-running the pipeline
end-to-end.

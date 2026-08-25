"""
Regenerate the two feature-selection supplementary figures with correct
sequential numbering (S6, S7) instead of the placeholder "Sx"/"Sy" letters
they were originally exported with (notebooks/01_build_feature_catalog_and_
area_inventory.ipynb, cell 19). Reconstructed from the raw per-seed CSVs
already saved by that notebook (RawMetrics_AllStrategies_AllSeeds.csv,
RawClasswiseMetrics_AllStrategies_AllSeeds.csv, RawSelectedFeatures_
AllStrategies.csv), so no RF retraining is needed - the plotting logic below
is copied verbatim from that cell, only the title strings and output
filenames changed.

Outputs:
  Supplementary_Figure_S5_FeatureCountSensitivity.png/.pdf
  Supplementary_Figure_S6_StrategyComparison_SensorComposition.png/.pdf
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coffeemap.config import load_config, class_info, coffee_class_ids
from coffeemap.plotting import set_publication_style

CONFIG = load_config(PROJECT_ROOT / "config" / "paper1_config.yaml")
CLASS_INFO = class_info(CONFIG)
COFFEE_CLASSES = coffee_class_ids(CONFIG)
CLASS_NAMES = CLASS_INFO

set_publication_style(font="Arial", dpi=600)

FIG_DIR = Path(__file__).resolve().parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

DPI = 600
SENSOR_ORDER = ["Sentinel-2", "Landsat 8/9", "Sentinel-1", "DEM"]
SENSOR_COLORS = {"Sentinel-2": "#2e7d32", "Landsat 8/9": "#ef6c00", "Sentinel-1": "#1565c0", "DEM": "#6a1b9a"}
FIGSIZE_SX = (8.0, 4.0)
FIGSIZE_SY = (8.0, 4.0)
strategy_order = ["RF_Pearson", "RF_VIF10", "RF_rank_only", "VIF10_only"]


def panel_label(ax, label, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")


def clean_axis(ax):
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def save_figure(fig, png_path, pdf_path, dpi=DPI):
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


# --------------------------------------------------------------------------
# Reconstruct the aggregated dataframes the original cell built in-notebook
# --------------------------------------------------------------------------
raw_metrics = pd.read_csv(TABLES_DIR / "RawMetrics_AllStrategies_AllSeeds.csv")
raw_classwise = pd.read_csv(TABLES_DIR / "RawClasswiseMetrics_AllStrategies_AllSeeds.csv")
selected_features_df = pd.read_csv(TABLES_DIR / "RawSelectedFeatures_AllStrategies.csv")

summary_metrics_sorted = (
    raw_metrics.groupby(["strategy", "feature_count_target", "n_selected"], as_index=False)
    .agg(
        n_runs=("seed", "count"),
        OA_mean=("OA", "mean"), OA_sd=("OA", "std"),
        MacroF1_mean=("MacroF1", "mean"), MacroF1_sd=("MacroF1", "std"),
        CoffeeSubclassMacroF1_mean=("CoffeeSubclassMacroF1", "mean"),
        CoffeeSubclassMacroF1_sd=("CoffeeSubclassMacroF1", "std"),
    )
    .sort_values(["strategy", "n_selected"])
)

class_summary = (
    raw_classwise.groupby(["strategy", "feature_count_target", "n_selected", "class_id", "class_name"], as_index=False)
    .agg(f1_mean=("f1", "mean"), f1_sd=("f1", "std"))
)

strategy_stats_df = raw_metrics[
    (
        (raw_metrics["strategy"].isin(["RF_Pearson", "RF_VIF10", "RF_rank_only"]))
        & (raw_metrics["feature_count_target"] == 25)
    )
    | (raw_metrics["strategy"] == "VIF10_only")
].copy()
strategy_stats_df["strategy"] = pd.Categorical(strategy_stats_df["strategy"], categories=strategy_order, ordered=True)

summary_stats = (
    strategy_stats_df.groupby(["strategy", "feature_count_target", "n_selected"], observed=True)
    .agg(
        n_runs=("seed", "count"),
        OA_mean=("OA", "mean"), OA_sd=("OA", "std"),
        MacroF1_mean=("MacroF1", "mean"), MacroF1_sd=("MacroF1", "std"),
        CoffeeF1_mean=("CoffeeSubclassMacroF1", "mean"), CoffeeF1_sd=("CoffeeSubclassMacroF1", "std"),
    )
    .reset_index()
)
for prefix in ["OA", "MacroF1", "CoffeeF1"]:
    summary_stats[f"{prefix}_se"] = summary_stats[f"{prefix}_sd"] / np.sqrt(summary_stats["n_runs"])

summary_stats["strategy_label"] = summary_stats.apply(lambda r: f"{r['strategy']}\n(n={int(r['n_selected'])})", axis=1)
summary_stats = summary_stats.sort_values("strategy")

# ============================================================
# Figure S6 (was "Sx"): feature-count sensitivity
# ============================================================
main_plot = summary_metrics_sorted[summary_metrics_sorted["strategy"] == "RF_Pearson"].sort_values("n_selected")
coffee_cls = class_summary[
    (class_summary["strategy"] == "RF_Pearson") & (class_summary["class_id"].isin(COFFEE_CLASSES))
].copy()

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_SX)

ax = axes[0]
for mean_col, sd_col, label in [
    ("OA_mean", "OA_sd", "Overall accuracy"),
    ("MacroF1_mean", "MacroF1_sd", "Macro F1"),
    ("CoffeeSubclassMacroF1_mean", "CoffeeSubclassMacroF1_sd", "Coffee F1"),
]:
    ax.errorbar(main_plot["n_selected"], main_plot[mean_col], yerr=main_plot[sd_col],
                marker="o", linewidth=1.2, capsize=3, label=label)
ax.axvline(25, linestyle="--", linewidth=1.0)
ax.set_xlabel("Number of selected predictors")
ax.set_ylabel("Score")
ax.set_title("Feature-count sensitivity")
ax.set_ylim(0.0, 1.02)
ax.legend(frameon=False, loc="lower right", fontsize=8)
clean_axis(ax)
panel_label(ax, "a")

ax = axes[1]
for class_id in COFFEE_CLASSES:
    g = coffee_cls[coffee_cls["class_id"] == class_id].sort_values("n_selected")
    ax.errorbar(g["n_selected"], g["f1_mean"], yerr=g["f1_sd"],
                marker="o", linewidth=1.2, capsize=3, label=CLASS_NAMES[class_id])
ax.axvline(25, linestyle="--", linewidth=1.0)
ax.set_xlabel("Number of selected predictors")
ax.set_ylabel("Class-wise F1")
ax.set_title("Coffee-system F1")
ax.set_ylim(0.0, 1.02)
ax.legend(frameon=False, loc="lower right", fontsize=8)
clean_axis(ax)
panel_label(ax, "b")

fig.tight_layout()

fig_s6_png = FIG_DIR / "Supplementary_Figure_S5_FeatureCountSensitivity.png"
fig_s6_pdf = FIG_DIR / "Supplementary_Figure_S5_FeatureCountSensitivity.pdf"
save_figure(fig, fig_s6_png, fig_s6_pdf)

# ============================================================
# Figure S7 (was "Sy"): strategy comparison + sensor composition
# ============================================================
strategy_plot = summary_stats.copy()
strategy_plot["strategy"] = pd.Categorical(strategy_plot["strategy"], categories=strategy_order, ordered=True)
strategy_plot = strategy_plot.sort_values("strategy").reset_index(drop=True)
strategy_plot["label"] = strategy_plot.apply(lambda r: f"{r['strategy']}\n(n={int(r['n_selected'])})", axis=1)

selected_b = []
for _, r in strategy_plot.iterrows():
    tmp = selected_features_df[
        (selected_features_df["strategy"] == r["strategy"])
        & (selected_features_df["feature_count_target"] == r["feature_count_target"])
        & (selected_features_df["n_selected"] == r["n_selected"])
    ].copy()
    tmp["label"] = r["label"]
    selected_b.append(tmp)
selected_b = pd.concat(selected_b, ignore_index=True)

comp = (
    selected_b.groupby(["label", "source_group"]).size().unstack(fill_value=0)
    .reindex(columns=SENSOR_ORDER, fill_value=0).reindex(strategy_plot["label"]).fillna(0)
)

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_SY)

ax = axes[0]
x = np.arange(len(strategy_plot))
bar_w = 0.24
err_kw = dict(lw=0.8, capsize=3, capthick=0.8)
ax.bar(x - bar_w, strategy_plot["OA_mean"], yerr=strategy_plot["OA_sd"], width=bar_w, label="OA", error_kw=err_kw)
ax.bar(x, strategy_plot["MacroF1_mean"], yerr=strategy_plot["MacroF1_sd"], width=bar_w, label="Macro F1", error_kw=err_kw)
ax.bar(x + bar_w, strategy_plot["CoffeeF1_mean"], yerr=strategy_plot["CoffeeF1_sd"], width=bar_w, label="Coffee F1", error_kw=err_kw)
ax.set_xticks(x)
ax.set_xticklabels(strategy_plot["label"], rotation=20, ha="right")
ax.set_ylim(0.0, 1.03)
ax.set_ylabel("Mean score")
ax.set_title("Feature-selection strategy comparison")
ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=8)
clean_axis(ax)
panel_label(ax, "a")

ax = axes[1]
x2 = np.arange(len(comp))
bottom = np.zeros(len(comp))
for sensor in SENSOR_ORDER:
    values = comp[sensor].values
    ax.bar(x2, values, bottom=bottom, label=sensor, color=SENSOR_COLORS[sensor])
    bottom += values
ax.set_xticks(x2)
ax.set_xticklabels(comp.index, rotation=20, ha="right")
ax.set_ylabel("Number of selected predictors")
ax.set_title("Sensor composition")
ax.legend(frameon=False, loc="upper left", fontsize=8)
clean_axis(ax)
panel_label(ax, "b")

fig.tight_layout()

fig_s7_png = FIG_DIR / "Supplementary_Figure_S6_StrategyComparison_SensorComposition.png"
fig_s7_pdf = FIG_DIR / "Supplementary_Figure_S6_StrategyComparison_SensorComposition.pdf"
save_figure(fig, fig_s7_png, fig_s7_pdf)

print("Saved:", fig_s6_png)
print("Saved:", fig_s7_png)

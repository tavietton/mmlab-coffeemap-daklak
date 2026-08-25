"""
Generate Figure 2 - Methodology Flowchart.

Single-column, 10-stage workflow diagram for the coffee-production-system
mapping manuscript (Random Forest + SHAP, Sentinel-1/2, Landsat 8/9, DEM,
10 land-cover classes, Dak Lak, Vietnam). Stage 7 (Validation & interpretation)
is rendered as an inline "swimlane" band containing five peer diagnostic
sub-boxes (7a-7e) arranged 3+2, mirroring the Bourgoin et al. (2020) convention
of keeping accuracy assessment in-line in the main flow rather than as a
disconnected side panel.

Design notes:
- All box geometry is computed top-down from explicit line-height / padding
  constants, so total canvas height is derived (not guessed) and the figure
  is verified to embed at a sane height on a portrait page.
- Left-aligned "bold title + regular detail line(s)" typography throughout.
- Numbered circular badges (neutral slate gray throughout, stages 1-10 and
  sub-stages 7a-7e alike) mark every stage. Stage 7's band and its 7a-7e
  sub-boxes use the same neutral gray box/border styling as every other
  stage, so the whole 10-stage flow reads as one unified family.
- Every one of the 10 main stages, and all five stage-7 sub-boxes, carries a
  restrained, hand-drawn vector icon (matplotlib patches only, no external
  icon fonts/clip-art, to preserve pdf.fonttype 42 editable-vector text) in a
  right-aligned gutter: the two data-bookend stages (1: multi-sensor input,
  8: final classified map) get color-tinted icons matching the manuscript's
  own sensor/class palettes; every other icon (all eight intermediate main
  stages plus all five stage-7 sub-boxes) shares one muted slate-blue-gray
  family (PROCESS_ICON_COLOR), so color stays reserved for the two data
  bookends and no stage box (including inside the swimlane) is left with an
  empty gray zone.
- Stage 10's icon deliberately shows two measured bars with a marked gap
  rather than a checkmark, so it does not visually assert a clean pass
  beyond what the area-consistency result actually reported (Section 3.6:
  a directionally consistent but non-trivial +6.4% bias with real
  district-level scatter).
- The arrow leaving the stage-7 band is dashed, distinct from every other
  (solid) stage-to-stage arrow, matching the caption's statement that step 7
  is diagnostic and does not feed into step 8's classified map.
- Both stage-7 sub-box rows share the same left/right column span (row 1's
  three sub-boxes and row 2's two sub-boxes both run edge-to-edge across the
  same inner width), so the "3+2 grid" is a genuine shared-boundary grid,
  not two independently centered rows.
- Effective printed text size at the manuscript's actual 6.5in embed width
  is kept above ~7pt for every load-bearing line (titles, detail lines, and
  sub-badge digits), and the raster is exported at 600 dpi so the artwork
  clears the journal's stated minimum pixel counts for combination artwork.
- A programmatic QA pass (text-overflow, box/arrow overlap, and icon-zone
  intrusion checks) runs before saving; see `run_qa()` at the bottom.

Outputs (overwritten in place):
  Figure2_MethodologyFlowchart.png   (600 dpi)
  Figure2_MethodologyFlowchart.pdf   (vector, editable text)
"""

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.patches import Arc

# --------------------------------------------------------------------------
# Style (matches the manuscript's established matplotlib conventions)
# --------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

OUT_DIR = Path(__file__).resolve().parent
OUT_STEM = OUT_DIR / "Figure2_MethodologyFlowchart"

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
CLASS_COLORS = {  # official 10-class palette (config/paper1_config.yaml) - used
    "sun_coffee": "#8c3b00",   # 1
    "forest": "#1f4d2b",       # 8
    "water": "#00c8ff",        # 9
    "built": "#9a9a9a",        # 10
}
SENSOR_COLORS = {  # existing manuscript SHAP-figure accent colors, reused verbatim
    "s2": "#2e7d32",
    "s1": "#1565c0",
    "landsat": "#ef6c00",
    "dem": "#6a1b9a",
}

SUBBOX_FILL = "#FFFFFF"
SUBBOX_BORDER = "#999999"
NEUTRAL_FILL = "#F2F2F2"
NEUTRAL_BORDER = "#333333"
# Data-container cylinder fills (bookend stages 1 and 8 only -- the two spots
# that represent a freestanding dataset/artifact rather than a computation
# step; every other stage stays the neutral process-gray). Kept as light
# tints of the manuscript's own established palette families (blue = sensor
# input family, green = land-cover/output family) so the two accents read as
# "belongs to this manuscript's existing color language" rather than new,
# unrelated colors.
DATA_FILL_INPUT = "#E3F2FD"   # stage 1: raw multi-sensor input + reference set
DATA_FILL_OUTPUT = "#E8F5E9"  # stage 8: final classified map
DATA_BORDER = NEUTRAL_BORDER  # keep the same border weight/color as process boxes
BADGE_NEUTRAL = "#3A3A3A"
TITLE_COLOR = "#000000"
DETAIL_COLOR = "#000000"
NOTE_COLOR = "#000000"
ARROW_COLOR = "#222222"
MICRO_LABEL_COLOR = "#000000"
# Single neutral tint shared by every "generic process step" icon: all eight
# intermediate main stages (2, 3, 4, 5, 6, 9, 10, plus the 7-band itself) and
# all five stage-7 sub-boxes (7a-7e). Deliberately a muted slate-blue-gray,
# distinct from both the vivid SENSOR_COLORS family and the CLASS_COLORS
# family, so these read as one coherent "process glyph" family rather than
# competing with the two colored data-bookend icons (sources/minimap).
PROCESS_ICON_COLOR = "#5B6B79"

# --------------------------------------------------------------------------
# Geometry constants (inches; 1 data unit == 1 inch, see figure setup below)
# --------------------------------------------------------------------------
## Page-fit target: the manuscript's actual A4 template (revision.docx /
## submission.docx: 8.269 x 11.694 in, margins T/B 0.787in, L 0.985in,
## R 0.787in) gives a usable area of 6.497 x 10.12 in - the existing 6.5in
## figure embed width already matches this almost exactly. That leaves more
## headroom than a Letter-page assumption would, which is spent below on
## substantially larger fonts (per explicit legibility-over-compactness
## instruction) rather than banked as blank space.
W = 9.5                 # canvas width
MARGIN_X = 0.20
BOX_X0 = MARGIN_X
BOX_W = W - 2 * MARGIN_X

TOP_MARGIN = 0.20
BOTTOM_MARGIN = 0.20
STAGE_GAP = 0.28         # vertical gap (arrow zone) between consecutive stages

TITLE_FS = 13.5
TITLE_LH = 0.263
HERO_TITLE_FS = 14.5
HERO_TITLE_LH = 0.283
DETAIL_FS = 11.0
DETAIL_LH = 0.213
LINE_GAP = 0.05
PAD_TOP = 0.10
PAD_BOTTOM = 0.10

# Sub-stage (7a-7e) typography: sized so effective printed size at the
# manuscript's ~0.68x embed-width shrink still clears ~7pt for every
# load-bearing line (titles, detail lines, badge digits) -- see the Guide for
# Authors' explicit ban on disproportionately small artwork text.
SUB_TITLE_FS = 12.0
SUB_TITLE_LH = 0.232
SUB_DETAIL_FS = 11.0
SUB_DETAIL_LH = 0.214
SUB_LINE_GAP = 0.035
SUB_PAD = 0.075

BAND_TOP_PAD = 0.10
BAND_HEADER_FS = 14.0
BAND_HEADER_LH = 0.273
BAND_NOTE_FS = 11.5
BAND_NOTE_LH = 0.224
BAND_HEADER_NOTE_GAP = 0.045
BAND_NOTE_GRID_GAP = 0.10
BAND_ROW_GAP = 0.10
BAND_BOTTOM_PAD = 0.10
BAND_SIDE_PAD = 0.15
BAND_SUBBOX_GAP = 0.13
BAND_ROUNDING = 0.045

BADGE_R = 0.15
SUB_BADGE_R = 0.125
SUB_BADGE_FS = 9.5
TEXT_INDENT = 0.58        # from box left edge to text left edge (past badge)
SUB_TEXT_INDENT = 0.40
RIGHT_PAD = 0.16
SUB_RIGHT_PAD = 0.10
BOX_ROUNDING = 0.06
CYLINDER_CAP_H = 0.16     # extra vertical allowance for the top/bottom cylinder caps

ICON_GUTTER_SOURCES = 1.70   # stage 1: 4 icons + micro-labels
ICON_GUTTER_SMALL = 0.50     # single-icon gutter: stages 6/8 (meaningful icons)
                              # and every other single-icon main stage
                              # (2, 3, 4, 5, 9, 10 - generic process glyphs)
ICON_GUTTER_SUB = 0.40       # every 7a-7e sub-box icon
ICON_MICRO_LABEL_FS = 9.5    # stage-1 sensor labels + the stage-4 "70:30" label

# Stage numbers (main-stage "num" strings) that get a single right-gutter
# icon using the shared ICON_GUTTER_SMALL width. Centralized here so the
# gutter-width dispatch and the QA icon-zone bookkeeping stay in lock-step.
SINGLE_ICON_NAMES = {"trees", "minimap", "layers", "grid", "split", "funnel", "adjust", "compare"}
# Stage-7 sub-box icon names, all drawn in PROCESS_ICON_COLOR.
SUB_ICON_NAMES = {"shap", "holdout", "blocks", "sensitivity", "cluster"}

# --------------------------------------------------------------------------
# Content (logic/structure preserved verbatim from the approved design;
# only the typographic split - bold title + regular detail line(s) - and the
# `icon=` keys were added)
# --------------------------------------------------------------------------
MAIN_STAGES = [
    dict(num="1", title="Data acquisition (Nov 2023–Oct 2024)", hero=True,
         details=["Sentinel-2, Sentinel-1, Landsat 8/9, SRTM DEM",
                   "3,000-pt reference set (300/class, Google Earth Pro)"],
         icon="sources", shape="cylinder", fill=DATA_FILL_INPUT),
    dict(num="2", title="Preprocessing",
         details=["Seasonal (S2, Landsat) / annual (S1) compositing; 10 m common grid"],
         icon="layers"),
    dict(num="3", title="Feature extraction",
         details=["97 candidate predictors"],
         icon="grid"),
    dict(num="4", title="Train / validation split",
         details=["2,100 / 900, patch-grouped, stratified 70:30"],
         icon="split"),
    dict(num="5", title="Feature selection (training subset only)",
         details=["RF rank → correlation filter → top 25 predictors"],
         icon="funnel"),
    dict(num="6", title="Random Forest classification", hero=True,
         details=["GEE Smile RF, 2,000 trees"],
         icon="trees"),
]

BAND_HEADER = "7.  Validation & interpretation of the step-6 model"
BAND_NOTE = ("Diagnostic stage: reports on the step-6 model; "
             "does not alter step 8's classified map")

SUBSTAGES_ROW1 = [
    dict(num="7a", title="Non-spatial validation",
         details=["Held-out 900-pt, 5-fold CV", "(Table 3–4)"],
         icon="holdout"),
    dict(num="7b", title="Spatial block CV",
         details=["10/15/20 km spatial blocks", "(Table S3–S4)"],
         icon="blocks"),
    dict(num="7c", title="Model sensitivity",
         details=["RF grid; predictor/strategy;", "algorithm choice", "(Suppl. S6–S10)"],
         icon="sensitivity"),
]
SUBSTAGES_ROW2 = [
    dict(num="7d", title="SHAP interpretation",
         details=["scikit-learn RF,", "99.1% agree w/ GEE RF"],
         icon="shap"),
    dict(num="7e", title="Sample independence audit",
         details=["Parcel-clustering check;", "leakage-removed retrain (Suppl. S8)"],
         icon="cluster"),
]

TAIL_STAGES = [
    dict(num="8", title="Final classified map", hero=True,
         details=["10 land-cover classes, 10 m resolution"],
         icon="minimap", shape="cylinder", fill=DATA_FILL_OUTPUT),
    dict(num="9", title="Area-weighted adjustment",
         details=["Olofsson et al. (2014): Table 5 → Table 6"],
         icon="adjust"),
    dict(num="10", title="District-level consistency check",
         details=["vs. official statistics (Figure 9)"],
         icon="compare"),
]

# --------------------------------------------------------------------------
# Height calculators (used both to size the canvas up front and to place
# boxes - keeps the two in lock-step so nothing drifts).
# --------------------------------------------------------------------------

def main_box_height(entry):
    n = len(entry["details"])
    title_lh = HERO_TITLE_LH if entry.get("hero") else TITLE_LH
    extra = CYLINDER_CAP_H if entry.get("shape") == "cylinder" else 0.0
    return PAD_TOP + title_lh + n * (LINE_GAP + DETAIL_LH) + PAD_BOTTOM + extra


def sub_box_height(entry):
    n = len(entry["details"])
    return SUB_PAD + SUB_TITLE_LH + n * (SUB_LINE_GAP + SUB_DETAIL_LH) + SUB_PAD


def band_height():
    row_h = max(sub_box_height(e) for e in SUBSTAGES_ROW1 + SUBSTAGES_ROW2)
    return (BAND_TOP_PAD + BAND_HEADER_LH + BAND_HEADER_NOTE_GAP + BAND_NOTE_LH
            + BAND_NOTE_GRID_GAP + row_h + BAND_ROW_GAP + row_h + BAND_BOTTOM_PAD)


TOTAL_H = (TOP_MARGIN + BOTTOM_MARGIN
           + sum(main_box_height(e) for e in MAIN_STAGES)
           + band_height()
           + sum(main_box_height(e) for e in TAIL_STAGES)
           + STAGE_GAP * (len(MAIN_STAGES) + len(TAIL_STAGES)))  # gaps between all 10 stage slots

H = round(TOTAL_H, 4)

# --------------------------------------------------------------------------
# Figure / axes: 1 data unit == 1 inch (axes fills the whole figure)
# --------------------------------------------------------------------------
fig = plt.figure(figsize=(W, H))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

# bookkeeping for QA
_boxes = []      # (x0, y0, x1, y1, label) for overlap checks
_texts = []      # (Text object, parent_box_bbox, label) for overflow checks
_arrows = []     # (x, y0, y1) vertical segments for crossing checks
_icon_zones = []  # (x0, y0, x1, y1, label) reserved icon-gutter rectangles for overlap QA


def add_box_record(x0, y0, x1, y1, label):
    _boxes.append((x0, y0, x1, y1, label))


def _stage_prefix(label):
    """'stage-2-title' / 'stage-2-icon-zone' / 'sub-7a-icon-zone' -> 'stage-2' / 'sub-7a'."""
    parts = label.split("-")
    return "-".join(parts[:2])


# --------------------------------------------------------------------------
# Icon primitives (small, restrained, hand-drawn vector glyphs)
# --------------------------------------------------------------------------

def icon_satellite(cx, cy, s, color, radar=False):
    bw, bh = s * 0.36, s * 0.50
    ax.add_patch(Rectangle((cx - bw / 2, cy - bh / 2), bw, bh,
                            facecolor=color, edgecolor="none", zorder=4))
    ww, wh = s * 0.34, s * 0.16
    ax.add_patch(Rectangle((cx - bw / 2 - ww, cy - wh / 2), ww, wh,
                            facecolor=color, edgecolor="none", alpha=0.80, zorder=4))
    ax.add_patch(Rectangle((cx + bw / 2, cy - wh / 2), ww, wh,
                            facecolor=color, edgecolor="none", alpha=0.80, zorder=4))
    if radar:
        for rad in (s * 0.26, s * 0.40):
            arc = Arc((cx, cy - bh / 2 - s * 0.02), rad * 2, rad * 2,
                      angle=0, theta1=205, theta2=335, color=color, linewidth=0.9, zorder=4)
            ax.add_patch(arc)


def icon_terrain(cx, cy, s, color):
    base_y = cy - s * 0.24
    tri1 = Polygon([(cx - s * 0.34, base_y), (cx - s * 0.02, base_y + s * 0.46),
                     (cx + s * 0.26, base_y)], closed=True,
                    facecolor=color, edgecolor="none", alpha=0.90, zorder=4)
    tri2 = Polygon([(cx - s * 0.08, base_y), (cx + s * 0.18, base_y + s * 0.34),
                     (cx + s * 0.42, base_y)], closed=True,
                    facecolor=color, edgecolor="none", alpha=0.62, zorder=5)
    ax.add_patch(tri1)
    ax.add_patch(tri2)


def icon_trees(cx, cy, s, color):
    for dx, hgt, alpha in ((-0.30, 0.34, 0.85), (0.0, 0.48, 1.0), (0.30, 0.34, 0.85)):
        base_y = cy - s * 0.24
        tri = Polygon([(cx + dx * s - s * 0.15, base_y), (cx + dx * s, base_y + s * hgt),
                        (cx + dx * s + s * 0.15, base_y)], closed=True,
                       facecolor=color, edgecolor="none", alpha=alpha, zorder=4)
        ax.add_patch(tri)
        ax.add_patch(Rectangle((cx + dx * s - s * 0.02, base_y - s * 0.07), s * 0.04, s * 0.07,
                                facecolor=color, edgecolor="none", alpha=alpha, zorder=4))


def icon_minimap(cx, cy, s, colors):
    # Irregular parcel mosaic (uneven cell sizes/positions) so this reads as a
    # small classified-map thumbnail rather than a uniform 2x2 app-icon grid.
    half = s * 0.40
    x0, y0 = cx - half, cy - half
    full = 2 * half
    # (rel_x, rel_y, rel_w, rel_h) as fractions of the full icon footprint
    parcels = [
        (0.00, 0.55, 0.62, 0.45, colors[0]),
        (0.62, 0.68, 0.38, 0.32, colors[1]),
        (0.62, 0.35, 0.38, 0.33, colors[2]),
        (0.00, 0.00, 0.40, 0.55, colors[3]),
        (0.40, 0.00, 0.60, 0.35, colors[1]),
    ]
    for rx, ry, rw, rh, col in parcels:
        ax.add_patch(Rectangle((x0 + rx * full, y0 + ry * full), rw * full, rh * full,
                                facecolor=col, edgecolor="white", linewidth=0.5, zorder=4))
    ax.add_patch(Rectangle((x0, y0), full, full, facecolor="none",
                            edgecolor="#222222", linewidth=1.0, zorder=5))


def icon_shap_bars(cx, cy, s, color):
    lengths = [0.55, 0.95, 0.35]
    bar_h = s * 0.14
    gap = s * 0.06
    total_h = len(lengths) * bar_h + (len(lengths) - 1) * gap
    y = cy + total_h / 2 - bar_h / 2
    for L in lengths:
        ax.add_patch(Rectangle((cx - s * 0.46, y - bar_h / 2), s * 0.92 * L, bar_h,
                                facecolor=color, edgecolor="none", alpha=0.85, zorder=4))
        y -= (bar_h + gap)


# -- "Generic process step" glyphs, all sharing PROCESS_ICON_COLOR, matching
#    the hand-drawn geometric style (plain patches, no gradients/shadows/
#    clip-art) of the icons above.

def icon_layers(cx, cy, s, color):
    """Stage 2 - Preprocessing: three offset, semi-transparent squares
    stacked diagonally, read as 'multiple time-slice images compositing
    into one common-grid layer'."""
    w, h = s * 0.52, s * 0.38
    offsets = [(-0.15, 0.14), (-0.015, 0.02), (0.12, -0.10)]
    alphas = [0.35, 0.55, 0.90]
    for (dx, dy), a in zip(offsets, alphas):
        rect = Rectangle((cx + dx * s - w / 2, cy + dy * s - h / 2), w, h,
                          facecolor=color, edgecolor=color, linewidth=0.7,
                          alpha=a, zorder=4)
        ax.add_patch(rect)


def icon_grid(cx, cy, s, color):
    """Stage 3 - Feature extraction: a small cell grid (table of candidate
    predictors), with a few cells lightly filled to suggest 'many computed
    columns extracted from the stack'."""
    cols, rows = 4, 3
    w, h = s * 0.64, s * 0.46
    x0, y0 = cx - w / 2, cy - h / 2
    highlight_cells = {(0, 2), (2, 1), (3, 0)}
    for c in range(cols):
        for r in range(rows):
            if (c, r) in highlight_cells:
                cell_w, cell_h = w / cols, h / rows
                ax.add_patch(Rectangle((x0 + c * cell_w, y0 + r * cell_h), cell_w, cell_h,
                                        facecolor=color, edgecolor="none", alpha=0.30, zorder=3))
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none",
                            edgecolor=color, linewidth=1.0, zorder=4))
    for i in range(1, cols):
        x = x0 + w * i / cols
        ax.plot([x, x], [y0, y0 + h], color=color, linewidth=0.6, zorder=4)
    for j in range(1, rows):
        y = y0 + h * j / rows
        ax.plot([x0, x0 + w], [y, y], color=color, linewidth=0.6, zorder=4)


def icon_split(cx, cy, s, color):
    """Stage 4 - Train/validation split: one bar divided ~70:30 by a thin
    gap, the larger (training) segment solid, the smaller (validation)
    segment lighter."""
    w, h = s * 0.68, s * 0.30
    split = 0.70
    gap = s * 0.045
    x0 = cx - w / 2
    train_w = w * split - gap / 2
    val_w = w * (1 - split) - gap / 2
    ax.add_patch(Rectangle((x0, cy - h / 2), train_w, h,
                            facecolor=color, edgecolor=color, linewidth=0.6,
                            alpha=0.85, zorder=4))
    ax.add_patch(Rectangle((x0 + train_w + gap, cy - h / 2), val_w, h,
                            facecolor=color, edgecolor=color, linewidth=0.6,
                            alpha=0.40, zorder=4))


def icon_funnel(cx, cy, s, color):
    """Stage 5 - Feature selection: a funnel narrowing from many candidate
    predictors (dots) at the top to a single retained stream at the bottom."""
    top_w, bottom_w = s * 0.62, s * 0.10
    top_y = cy + s * 0.20
    bottom_y = cy - s * 0.06
    funnel = Polygon([(cx - top_w / 2, top_y), (cx + top_w / 2, top_y),
                       (cx + bottom_w / 2, bottom_y), (cx - bottom_w / 2, bottom_y)],
                      closed=True, facecolor=color, edgecolor="none", alpha=0.55, zorder=4)
    ax.add_patch(funnel)
    stem_w, stem_h = s * 0.10, s * 0.15
    ax.add_patch(Rectangle((cx - stem_w / 2, bottom_y - stem_h), stem_w, stem_h,
                            facecolor=color, edgecolor="none", alpha=0.85, zorder=4))
    for dx in (-0.20, -0.07, 0.07, 0.20):
        ax.add_patch(Circle((cx + dx * s, top_y + s * 0.09), s * 0.028,
                             facecolor=color, edgecolor="none", zorder=5))


def icon_adjust(cx, cy, s, color):
    """Stage 9 - Area-weighted adjustment: a small VERTICAL bar chart with a
    dashed reference line across the tops, reading as 'raw class-area
    estimates reconciled against a corrected reference'."""
    heights = [0.28, 0.55, 0.40]
    bar_w = s * 0.14
    gap = s * 0.09
    total_w = 3 * bar_w + 2 * gap
    x0 = cx - total_w / 2
    base_y = cy - s * 0.22
    for i, hfrac in enumerate(heights):
        x = x0 + i * (bar_w + gap)
        bh = s * hfrac
        ax.add_patch(Rectangle((x, base_y), bar_w, bh,
                                facecolor=color, edgecolor="none", alpha=0.80, zorder=4))
    ref_y = base_y + s * 0.60
    ax.plot([x0 - s * 0.03, x0 + total_w + s * 0.03], [ref_y, ref_y],
            color=color, linewidth=0.9, linestyle=(0, (2, 1.5)), zorder=5)


def icon_compare(cx, cy, s, color):
    """Stage 10 - District-level consistency check: two HORIZONTAL bars
    (mapped estimate vs. official statistic, of different length) with a
    small double-headed arrow marking the measured gap between their ends.
    Deliberately not a checkmark: the icon should read as 'measured against
    a reference, with a quantified gap', not 'confirmed correct', since the
    actual reported result (Section 3.6) is a directionally consistent but
    non-trivial +6.4% bias with real district-level scatter, not a clean
    pass. The horizontal orientation also keeps this visually distinct from
    stage 9's vertical bar chart."""
    bar_h = s * 0.15
    gap_v = s * 0.13
    lengths = [0.62, 0.48]  # mapped (top) longer than official (bottom): matches the overestimate
    x0 = cx - s * 0.34
    y_top = cy + gap_v / 2
    y_bot = cy - gap_v / 2
    ax.add_patch(Rectangle((x0, y_top - bar_h / 2), s * lengths[0], bar_h,
                            facecolor=color, edgecolor="none", alpha=0.85, zorder=4))
    ax.add_patch(Rectangle((x0, y_bot - bar_h / 2), s * lengths[1], bar_h,
                            facecolor=color, edgecolor="none", alpha=0.55, zorder=4))
    x_short = x0 + s * lengths[1]
    x_long = x0 + s * lengths[0]
    ax.annotate("", xy=(x_long, cy), xytext=(x_short, cy),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.0, alpha=0.85), zorder=5)


def icon_holdout(cx, cy, s, color):
    """Sub-stage 7a - Non-spatial validation: a 3x2 grid of small blocks,
    five dim ('training folds') plus one solid ('held-out fold'), reading as
    'one fold set aside from the training pool for held-out evaluation'."""
    cols, rows = 3, 2
    cell = s * 0.20
    gap = s * 0.06
    grid_w = cols * cell + (cols - 1) * gap
    grid_h = rows * cell + (rows - 1) * gap
    x0, y0 = cx - grid_w / 2, cy - grid_h / 2
    holdout = (2, 0)
    for c in range(cols):
        for r in range(rows):
            x = x0 + c * (cell + gap)
            y = y0 + r * (cell + gap)
            is_holdout = (c, r) == holdout
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=color, edgecolor="none",
                                    alpha=0.92 if is_holdout else 0.30, zorder=4))


def icon_blocks(cx, cy, s, color):
    """Sub-stage 7b - Spatial block CV: three nested square outlines of
    different sizes, reading as 'multiple spatial block sizes tested'."""
    for frac, lw, a in ((0.30, 1.3, 0.95), (0.46, 1.0, 0.62), (0.62, 0.8, 0.38)):
        side = s * frac
        ax.add_patch(Rectangle((cx - side / 2, cy - side / 2), side, side,
                                facecolor="none", edgecolor=color, linewidth=lw, alpha=a, zorder=4))


def icon_sensitivity(cx, cy, s, color):
    """Sub-stage 7c - Model sensitivity: a small fan of radiating lines of
    varying length from a common origin, reading as 'a parameter sweep'."""
    base_x, base_y = cx, cy - s * 0.20
    angles = [50, 72, 90, 108, 130]
    lengths = [0.30, 0.42, 0.50, 0.40, 0.28]
    for ang, L in zip(angles, lengths):
        rad = math.radians(ang)
        ex = base_x + s * L * math.cos(rad)
        ey = base_y + s * L * math.sin(rad)
        ax.plot([base_x, ex], [base_y, ey], color=color, linewidth=1.3, alpha=0.75, zorder=4,
                solid_capstyle="round")
    ax.add_patch(Circle((base_x, base_y), s * 0.035, facecolor=color, edgecolor="none", zorder=5))


def icon_cluster(cx, cy, s, color):
    """Sub-stage 7e - Sample independence audit: two small dot-clusters,
    each ringed by a thin boundary, reading as 'points grouped and checked
    by parcel cluster, not treated as independent'."""
    clusters = [(-0.16, 0.06), (0.15, -0.07)]
    dot_offsets = [(-0.05, 0.03), (0.045, 0.045), (0.01, -0.045)]
    for ccx, ccy in clusters:
        cx0, cy0 = cx + ccx * s, cy + ccy * s
        for dx, dy in dot_offsets:
            ax.add_patch(Circle((cx0 + dx * s, cy0 + dy * s), s * 0.030,
                                 facecolor=color, alpha=0.85, edgecolor="none", zorder=5))
        ax.add_patch(Circle((cx0, cy0), s * 0.115, facecolor="none",
                             edgecolor=color, linewidth=0.9, alpha=0.55, zorder=4))


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------

def draw_badge(cx, cy, r, fill, text, fontsize=8.5):
    ax.add_patch(Circle((cx, cy), r, facecolor=fill, edgecolor="none", zorder=6))
    t = ax.text(cx, cy, text, fontsize=fontsize, fontweight="bold", color="white",
                ha="center", va="center", zorder=7, family="Times New Roman")
    return t


def draw_arrow(x, y_start, y_end, linestyle="-"):
    arr = FancyArrowPatch((x, y_start), (x, y_end), arrowstyle="-|>",
                           mutation_scale=9, linewidth=1.15, color=ARROW_COLOR,
                           shrinkA=0, shrinkB=0, zorder=3, linestyle=linestyle)
    ax.add_patch(arr)
    _arrows.append((x, y_start, y_end))


def draw_cylinder(x0, y_bottom, w, h, cap_h, facecolor, edgecolor, linewidth, zorder):
    """Classic 'database cylinder' glyph: straight vertical sides, a fully-drawn
    elliptical top cap, and a front-facing elliptical arc at the bottom (the
    conventional way to signal 'this is a data container', distinct from the
    plain rounded rectangles used for process/computation stages)."""
    cx = x0 + w / 2
    y_top = y_bottom + h
    body_bottom = y_bottom + cap_h / 2
    body_top = y_top - cap_h / 2

    # bottom cap: filled ellipse (back half hidden by the body fill drawn after)
    ax.add_patch(Ellipse((cx, body_bottom), w, cap_h, facecolor=facecolor,
                          edgecolor="none", zorder=zorder))
    # straight body
    ax.add_patch(Rectangle((x0, body_bottom), w, body_top - body_bottom,
                            facecolor=facecolor, edgecolor="none", zorder=zorder + 0.1))
    # top cap: filled + outlined ellipse (the visible "lid")
    ax.add_patch(Ellipse((cx, body_top), w, cap_h, facecolor=facecolor,
                          edgecolor=edgecolor, linewidth=linewidth, zorder=zorder + 0.3))
    # outline: two straight sides + the front (lower) arc of the bottom cap
    ax.plot([x0, x0], [body_bottom, body_top], color=edgecolor, linewidth=linewidth,
            zorder=zorder + 0.2, solid_capstyle="butt")
    ax.plot([x0 + w, x0 + w], [body_bottom, body_top], color=edgecolor, linewidth=linewidth,
            zorder=zorder + 0.2, solid_capstyle="butt")
    ax.add_patch(Arc((cx, body_bottom), w, cap_h, angle=0, theta1=180, theta2=360,
                      edgecolor=edgecolor, linewidth=linewidth, zorder=zorder + 0.2))


def draw_main_box(entry, y_top):
    h = main_box_height(entry)
    y_bottom = y_top - h
    x0, x1 = BOX_X0, BOX_X0 + BOX_W
    is_cylinder = entry.get("shape") == "cylinder"
    if is_cylinder:
        draw_cylinder(x0, y_bottom, BOX_W, h, CYLINDER_CAP_H,
                       facecolor=entry.get("fill", NEUTRAL_FILL), edgecolor=DATA_BORDER,
                       linewidth=1.1, zorder=2)
    else:
        box = FancyBboxPatch((x0, y_bottom), BOX_W, h,
                              boxstyle=f"round,pad=0,rounding_size={BOX_ROUNDING}",
                              linewidth=1.1, edgecolor=NEUTRAL_BORDER, facecolor=NEUTRAL_FILL,
                              zorder=2)
        ax.add_patch(box)
    add_box_record(x0, y_bottom, x1, y_top, f"stage-{entry['num']}")

    badge_cx = x0 + 0.24
    content_top = y_top - PAD_TOP - (CYLINDER_CAP_H / 2 if is_cylinder else 0.0)
    title_lh = HERO_TITLE_LH if entry.get("hero") else TITLE_LH
    title_fs = HERO_TITLE_FS if entry.get("hero") else TITLE_FS
    title_y = content_top - title_lh / 2
    draw_badge(badge_cx, title_y, BADGE_R, BADGE_NEUTRAL, entry["num"], fontsize=9.5)

    icon_gutter = 0.0
    if entry.get("icon") == "sources":
        icon_gutter = ICON_GUTTER_SOURCES
    elif entry.get("icon") in SINGLE_ICON_NAMES:
        icon_gutter = ICON_GUTTER_SMALL

    text_x0 = x0 + TEXT_INDENT
    text_x1 = x1 - RIGHT_PAD - icon_gutter
    tt = ax.text(text_x0, title_y, entry["title"], fontsize=title_fs, fontweight="bold",
                 color=TITLE_COLOR, ha="left", va="center", family="Times New Roman", zorder=4)
    _texts.append((tt, (x0, y_bottom, x1, y_top), f"stage-{entry['num']}-title"))

    cursor = content_top - title_lh
    for d in entry["details"]:
        cursor -= LINE_GAP
        line_y = cursor - DETAIL_LH / 2
        dt = ax.text(text_x0, line_y, d, fontsize=DETAIL_FS, color=DETAIL_COLOR,
                     ha="left", va="center", family="Times New Roman", zorder=4)
        _texts.append((dt, (x0, y_bottom, x1, y_top), f"stage-{entry['num']}-detail"))
        cursor -= DETAIL_LH

    # Record the reserved icon-gutter zone for QA (check 4: no text may
    # intrude into a stage's own icon gutter).
    if icon_gutter > 0:
        gutter_x0 = x1 - RIGHT_PAD - icon_gutter
        _icon_zones.append((gutter_x0, y_bottom, x1 - RIGHT_PAD, y_top,
                             f"stage-{entry['num']}-icon-zone"))

    icon_cy = (y_top + y_bottom) / 2
    icon_name = entry.get("icon")
    if icon_name == "sources":
        icon_names = ["s2", "s1", "landsat", "dem"]
        icon_labels = ["S2", "S1", "L8/9", "DEM"]
        n_icons = len(icon_names)
        cell_w = ICON_GUTTER_SOURCES / n_icons
        gutter_x0 = x1 - RIGHT_PAD - ICON_GUTTER_SOURCES
        for i, (name, lab) in enumerate(zip(icon_names, icon_labels)):
            cx = gutter_x0 + cell_w * (i + 0.5)
            icy = icon_cy + 0.055
            s = 0.36
            if name == "s2":
                icon_satellite(cx, icy, s, SENSOR_COLORS["s2"])
            elif name == "s1":
                icon_satellite(cx, icy, s, SENSOR_COLORS["s1"], radar=True)
            elif name == "landsat":
                icon_satellite(cx, icy, s, SENSOR_COLORS["landsat"])
            elif name == "dem":
                icon_terrain(cx, icy, s, SENSOR_COLORS["dem"])
            lt = ax.text(cx, icy - 0.22, lab, fontsize=ICON_MICRO_LABEL_FS, color=MICRO_LABEL_COLOR,
                         ha="center", va="center", family="Times New Roman", zorder=4)
            _texts.append((lt, (x0, y_bottom, x1, y_top), f"stage-{entry['num']}-icon-label-{lab}"))
    elif icon_name in SINGLE_ICON_NAMES:
        cx = x1 - RIGHT_PAD - ICON_GUTTER_SMALL / 2
        if icon_name == "trees":
            icon_trees(cx, icon_cy, 0.42, PROCESS_ICON_COLOR)
        elif icon_name == "minimap":
            icon_minimap(cx, icon_cy, 0.36,
                         [CLASS_COLORS["sun_coffee"], CLASS_COLORS["forest"],
                          CLASS_COLORS["water"], CLASS_COLORS["built"]])
        elif icon_name == "layers":
            icon_layers(cx, icon_cy, 0.42, PROCESS_ICON_COLOR)
        elif icon_name == "grid":
            icon_grid(cx, icon_cy, 0.42, PROCESS_ICON_COLOR)
        elif icon_name == "split":
            icon_split(cx, icon_cy, 0.42, PROCESS_ICON_COLOR)
            lt = ax.text(cx, icon_cy - 0.24, "70:30", fontsize=ICON_MICRO_LABEL_FS,
                         color=MICRO_LABEL_COLOR, ha="center", va="center",
                         family="Times New Roman", zorder=4)
            _texts.append((lt, (x0, y_bottom, x1, y_top), f"stage-{entry['num']}-icon-label-7030"))
        elif icon_name == "funnel":
            icon_funnel(cx, icon_cy, 0.42, PROCESS_ICON_COLOR)
        elif icon_name == "adjust":
            icon_adjust(cx, icon_cy, 0.42, PROCESS_ICON_COLOR)
        elif icon_name == "compare":
            icon_compare(cx, icon_cy, 0.42, PROCESS_ICON_COLOR)

    return y_bottom


def draw_sub_box(entry, x0, y_top, w):
    h = sub_box_height(entry)
    y_bottom = y_top - h
    x1 = x0 + w
    box = FancyBboxPatch((x0, y_bottom), w, h,
                          boxstyle=f"round,pad=0,rounding_size=0.04",
                          linewidth=0.9, edgecolor=SUBBOX_BORDER, facecolor=SUBBOX_FILL,
                          zorder=4)
    ax.add_patch(box)
    add_box_record(x0, y_bottom, x1, y_top, f"sub-{entry['num']}")

    badge_cx = x0 + 0.19
    content_top = y_top - SUB_PAD
    title_y = content_top - SUB_TITLE_LH / 2
    draw_badge(badge_cx, title_y, SUB_BADGE_R, BADGE_NEUTRAL, entry["num"], fontsize=SUB_BADGE_FS)

    icon_name = entry.get("icon")
    icon_gutter = ICON_GUTTER_SUB if icon_name in SUB_ICON_NAMES else 0.0
    text_x0 = x0 + SUB_TEXT_INDENT
    text_x1 = x1 - SUB_RIGHT_PAD - icon_gutter
    tt = ax.text(text_x0, title_y, entry["title"], fontsize=SUB_TITLE_FS, fontweight="bold",
                 color=TITLE_COLOR, ha="left", va="center", family="Times New Roman", zorder=5,
                 wrap=False)
    _texts.append((tt, (x0, y_bottom, x1, y_top), f"sub-{entry['num']}-title"))

    cursor = content_top - SUB_TITLE_LH
    for d in entry["details"]:
        cursor -= SUB_LINE_GAP
        line_y = cursor - SUB_DETAIL_LH / 2
        dt = ax.text(text_x0, line_y, d, fontsize=SUB_DETAIL_FS, color=DETAIL_COLOR,
                     ha="left", va="center", family="Times New Roman", zorder=5)
        _texts.append((dt, (x0, y_bottom, x1, y_top), f"sub-{entry['num']}-detail"))
        cursor -= SUB_DETAIL_LH

    if icon_gutter > 0:
        gutter_x0 = x1 - SUB_RIGHT_PAD - icon_gutter
        _icon_zones.append((gutter_x0, y_bottom, x1 - SUB_RIGHT_PAD, y_top,
                             f"sub-{entry['num']}-icon-zone"))

    if icon_name in SUB_ICON_NAMES:
        icy = (y_top + y_bottom) / 2
        icon_cx = x1 - SUB_RIGHT_PAD - ICON_GUTTER_SUB / 2
        if icon_name == "shap":
            icon_shap_bars(icon_cx, icy, 0.32, PROCESS_ICON_COLOR)
        elif icon_name == "holdout":
            icon_holdout(icon_cx, icy, 0.32, PROCESS_ICON_COLOR)
        elif icon_name == "blocks":
            icon_blocks(icon_cx, icy, 0.32, PROCESS_ICON_COLOR)
        elif icon_name == "sensitivity":
            icon_sensitivity(icon_cx, icy, 0.32, PROCESS_ICON_COLOR)
        elif icon_name == "cluster":
            icon_cluster(icon_cx, icy, 0.32, PROCESS_ICON_COLOR)

    return y_bottom


def draw_band(y_top):
    h = band_height()
    y_bottom = y_top - h
    x0, x1 = BOX_X0, BOX_X0 + BOX_W
    band = FancyBboxPatch((x0, y_bottom), BOX_W, h,
                           boxstyle=f"round,pad=0,rounding_size={BAND_ROUNDING}",
                           linewidth=1.1, edgecolor=NEUTRAL_BORDER, facecolor=NEUTRAL_FILL, zorder=2)
    ax.add_patch(band)
    add_box_record(x0, y_bottom, x1, y_top, "band-7")

    badge_cx = x0 + 0.24
    header_y = y_top - BAND_TOP_PAD - BAND_HEADER_LH / 2
    draw_badge(badge_cx, header_y, BADGE_R, BADGE_NEUTRAL, "7", fontsize=9.5)
    ht = ax.text(x0 + TEXT_INDENT, header_y, BAND_HEADER, fontsize=BAND_HEADER_FS,
                 fontweight="bold", color="#000000", ha="left", va="center",
                 family="Times New Roman", zorder=4)
    _texts.append((ht, (x0, y_bottom, x1, y_top), "band-header"))

    note_y = header_y - BAND_HEADER_LH / 2 - BAND_HEADER_NOTE_GAP - BAND_NOTE_LH / 2
    nt = ax.text(x0 + TEXT_INDENT, note_y, BAND_NOTE, fontsize=BAND_NOTE_FS, style="italic",
                 color=NOTE_COLOR, ha="left", va="center", family="Times New Roman", zorder=4)
    _texts.append((nt, (x0, y_bottom, x1, y_top), "band-note"))

    grid_top = note_y - BAND_NOTE_LH / 2 - BAND_NOTE_GRID_GAP

    row_h = max(sub_box_height(e) for e in SUBSTAGES_ROW1 + SUBSTAGES_ROW2)
    inner_x0 = x0 + BAND_SIDE_PAD
    inner_w = BOX_W - 2 * BAND_SIDE_PAD

    # Both rows share the identical [inner_x0, inner_x0 + inner_w] span, so
    # row 1 (3 columns) and row 2 (2 columns) form a genuine shared-boundary
    # grid rather than two independently centered rows.
    n1 = len(SUBSTAGES_ROW1)
    sub_w1 = (inner_w - (n1 - 1) * BAND_SUBBOX_GAP) / n1
    cx = inner_x0
    for e in SUBSTAGES_ROW1:
        draw_sub_box(e, cx, grid_top, sub_w1)
        cx += sub_w1 + BAND_SUBBOX_GAP

    n2 = len(SUBSTAGES_ROW2)
    sub_w2 = (inner_w - (n2 - 1) * BAND_SUBBOX_GAP) / n2
    row2_top = grid_top - row_h - BAND_ROW_GAP
    cx = inner_x0
    for e in SUBSTAGES_ROW2:
        draw_sub_box(e, cx, row2_top, sub_w2)
        cx += sub_w2 + BAND_SUBBOX_GAP

    return y_bottom


# --------------------------------------------------------------------------
# Lay out the full column top-down
# --------------------------------------------------------------------------
cursor_y = H - TOP_MARGIN
x_center = W / 2

all_slots = list(MAIN_STAGES) + ["__BAND__"] + list(TAIL_STAGES)
prev_bottom = None
prev_slot = None
for slot in all_slots:
    if prev_bottom is not None:
        cursor_y = prev_bottom - STAGE_GAP
        # The arrow leaving the diagnostic band is dashed, distinct from
        # every other (solid) transition, matching the caption's statement
        # that step 7 does not feed into step 8's classified map.
        arrow_style = "dashed" if prev_slot == "__BAND__" else "solid"
        draw_arrow(x_center, prev_bottom, cursor_y, linestyle=arrow_style)
    if slot == "__BAND__":
        prev_bottom = draw_band(cursor_y)
    else:
        prev_bottom = draw_main_box(slot, cursor_y)
    prev_slot = slot

# --------------------------------------------------------------------------
# QA: programmatic checks (not eyeballed) - text overflow, box overlap,
# arrow/box crossings, and icon-zone intrusion.
# --------------------------------------------------------------------------

def run_qa():
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    problems = []

    # Compute every text bbox once (data coords), reused by checks 1 and 4.
    text_bboxes = []  # (label, x0d, y0d, x1d, y1d, parent_bbox)
    for text_obj, parent_bbox, label in _texts:
        bb = text_obj.get_window_extent(renderer=renderer)
        (x0d, y0d) = inv.transform((bb.x0, bb.y0))
        (x1d, y1d) = inv.transform((bb.x1, bb.y1))
        text_bboxes.append((label, x0d, y0d, x1d, y1d, parent_bbox))

    # 1) text overflow: every text bbox must sit inside its parent box bbox
    for label, x0d, y0d, x1d, y1d, parent_bbox in text_bboxes:
        px0, py0, px1, py1 = parent_bbox
        margin_right = px1 - x1d
        margin_left = x0d - px0
        margin_top = py1 - y1d
        margin_bottom = y0d - py0
        if margin_right < 0.01 or margin_left < -0.02 or margin_top < -0.02 or margin_bottom < -0.02:
            problems.append(
                f"OVERFLOW  {label}: text_x=[{x0d:.3f},{x1d:.3f}] box_x=[{px0:.3f},{px1:.3f}] "
                f"right_margin={margin_right:.3f} top_margin={margin_top:.3f} bottom_margin={margin_bottom:.3f}"
            )

    # 2) box/box overlap (skip band vs. its own sub-boxes: sub-boxes are
    #    legitimately nested inside band-7)
    for i in range(len(_boxes)):
        for j in range(i + 1, len(_boxes)):
            x0a, y0a, x1a, y1a, la = _boxes[i]
            x0b, y0b, x1b, y1b, lb = _boxes[j]
            nested_pair = ("band-7" in (la, lb)) and ("sub-" in la or "sub-" in lb)
            if nested_pair:
                continue
            overlap_x = min(x1a, x1b) - max(x0a, x0b)
            overlap_y = min(y1a, y1b) - max(y0a, y0b)
            if overlap_x > 0.001 and overlap_y > 0.001:
                problems.append(f"OVERLAP   {la} <-> {lb}: dx={overlap_x:.3f} dy={overlap_y:.3f}")

    # 3) arrows must not cross any non-adjacent box (they should live entirely
    #    inside an empty gap by construction; verify that construction held)
    for (ax_, ay0, ay1) in _arrows:
        ylo, yhi = min(ay0, ay1), max(ay0, ay1)
        for (x0b, y0b, x1b, y1b, lb) in _boxes:
            if x0b <= ax_ <= x1b:
                inter = min(yhi, y1b) - max(ylo, y0b)
                if inter > 0.005:
                    problems.append(f"ARROW-CROSS arrow@x={ax_:.2f} y=[{ylo:.2f},{yhi:.2f}] hits {lb}")

    # 4) icon-zone intrusion: no stage's own title/detail text may overlap
    #    that same stage's reserved icon gutter. Icon micro-labels (e.g. "S2",
    #    "S1" under the stage-1 sensor icons, or "70:30" under stage 4's
    #    split icon) are legitimately inside their stage's icon zone, so they
    #    are excluded by construction.
    for label, x0d, y0d, x1d, y1d, parent_bbox in text_bboxes:
        if "icon-label" in label:
            continue
        stage_id = _stage_prefix(label)
        for zx0, zy0, zx1, zy1, zlabel in _icon_zones:
            if _stage_prefix(zlabel) != stage_id:
                continue
            overlap_x = min(x1d, zx1) - max(x0d, zx0)
            overlap_y = min(y1d, zy1) - max(y0d, zy0)
            if overlap_x > 0.001 and overlap_y > 0.001:
                problems.append(
                    f"ICON-OVERLAP {label} intrudes into {zlabel}: dx={overlap_x:.3f} dy={overlap_y:.3f}"
                )

    return problems


qa_problems = run_qa()

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
png_path = OUT_STEM.with_suffix(".png")
pdf_path = OUT_STEM.with_suffix(".pdf")
fig.savefig(png_path, dpi=600, facecolor="white")
fig.savefig(pdf_path, facecolor="white")

print(f"Canvas: {W:.3f} x {H:.3f} in  (embedded at 6.5in width -> {H * 6.5 / W:.3f} in tall)")
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
if qa_problems:
    print(f"\nQA: {len(qa_problems)} problem(s) found:")
    for p in qa_problems:
        print("  -", p)
else:
    print("\nQA: PASS - no text overflow, no box overlap, no arrow/box crossings detected.")

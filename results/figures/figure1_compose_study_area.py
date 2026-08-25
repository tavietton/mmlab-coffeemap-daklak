"""
Figure 1: study-area map (Dak Lak Province, district boundaries, Vietnam
locator inset), rebuilt as pure vector cartography (geopandas + matplotlib)
to replace the earlier ArcGIS Pro export, which had no coordinate graticule.
Reviewer 2 explicitly asked for geographic coordinates on this figure; this
version adds a real lon/lat graticule in the same style as Figure 8's.

Run with any Python environment that has geopandas installed:
  python figure1_compose_study_area.py

Requires ArcGIS_Coffee_Map/ (see Inputs below), a sibling of this repository
under the same parent folder, not part of this repository -- available from
the corresponding author on request, same as the other excluded raw data.

Inputs:
  ArcGIS_Coffee_Map/Shapefile/daklak_details/daklak.shp        (15 districts, EPSG:4326)
  ArcGIS_Coffee_Map/Shapefile/Việt Nam (tỉnh thành) - T0107.shp (63 provinces, EPSG:4326, for
                                                                   neighboring-province context
                                                                   and the Vietnam locator inset)

Output:
  Figure1_StudyArea.png / .pdf

Design notes:
- The Vietnam locator inset originally overlapped the main map's top-right
  corner (a common cartographic convention), positioned by eyeballed figure-
  fraction coordinates. That overlap turned out to collide with the
  rightmost graticule tick labels: the inset axes is opaque and drawn after
  the main map, so any tick label whose text extended under the inset's
  bounding box was silently painted over and invisible. Eyeballing a fixed
  fig-fraction gap is fragile because the graticule's own tick positions
  depend on the data extent/aspect ratio, so a "safe" gap for one province
  shape is not guaranteed for another. Fixed here by giving the inset its
  own dedicated column to the right of the main map, in physical inches,
  with a real gap between the two axes' bounding boxes -- so the two can
  never overlap regardless of how many graticule ticks the main map ends up
  showing. This is the same non-overlapping-panel approach already used for
  Figure 8's legend (figure8_02_compose.py's `rect()` helper).
"""
import math
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Polygon, Rectangle


def draw_compass_rose(ax, cx, cy, r_long, color="#000000", zorder=9):
    """An 8-point compass-rose north arrow (circle + graduation ticks + an
    8-pointed star, long points on the cardinal directions) matching the
    ArcGIS-style compass used in Figure 8's main panel, so both figures in
    the manuscript share one north-arrow visual language rather than mixing
    a plain arrow (Figure 1's original placeholder) with a compass rose
    (Figure 8)."""
    r_short = r_long * 0.62
    r_in = r_long * 0.16
    r_circle = r_long * 0.78

    # 16 vertices, alternating [cardinal tip, inner notch, intercardinal
    # tip, inner notch, ...] at 22.5-degree steps, bearing 0 = north.
    pts = []
    for k in range(16):
        bearing = k * 22.5
        if k % 4 == 0:
            r = r_long
        elif k % 2 == 0:
            r = r_short
        else:
            r = r_in
        theta = math.radians(bearing)
        pts.append((cx + r * math.sin(theta), cy + r * math.cos(theta)))
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=color,
                          linewidth=0.9, zorder=zorder, joinstyle="miter"))

    ax.add_patch(Circle((cx, cy), r_circle, fill=False, edgecolor=color,
                         linewidth=1.4, zorder=zorder))
    # small graduation ticks around the circle, between the star's points
    for k in range(32):
        bearing = k * 11.25
        if bearing % 22.5 == 0:
            continue
        theta = math.radians(bearing)
        x0, y0 = cx + r_circle * math.sin(theta), cy + r_circle * math.cos(theta)
        x1, y1 = cx + r_circle * 1.13 * math.sin(theta), cy + r_circle * 1.13 * math.cos(theta)
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.1, zorder=zorder)

    label_r = r_long * 1.50
    # fontsize=18 targets the same effective printed size as Figure 8's
    # compass/scale-bar "furniture" text (FURNITURE_FONTSIZE=22 at Fig8's
    # 0.517 shrink = ~11.4pt effective; 11.4 / Figure 1's ~0.64 shrink = ~18).
    for label, bearing in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
        theta = math.radians(bearing)
        ax.text(cx + label_r * math.sin(theta), cy + label_r * math.cos(theta), label,
                fontsize=18, ha="center", va="center", color=color,
                family="Arial", zorder=zorder)

mpl.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "savefig.dpi": 400,
})

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../results/figures/ -> .../results/ -> repo root
PARENT_DIR = REPO_ROOT.parent  # sibling folder to this repository, holds ArcGIS_Coffee_Map/
DAKLAK_SHP = PARENT_DIR / "ArcGIS_Coffee_Map/Shapefile/daklak_details/daklak.shp"
VN_SHP = PARENT_DIR / "ArcGIS_Coffee_Map/Shapefile/Việt Nam (tỉnh thành) - T0107.shp"
HILLSHADE_NPZ = REPO_ROOT / "data/dem_hillshade_reference/daklak_hillshade.npz"
OUT_DIR = REPO_ROOT / "results/figures"

hs = np.load(HILLSHADE_NPZ)
HILLSHADE, HS_LON_MIN, HS_LON_MAX, HS_LAT_MIN, HS_LAT_MAX = (
    hs["hillshade"], float(hs["lon_min"]), float(hs["lon_max"]), float(hs["lat_min"]), float(hs["lat_max"]))

FILL_RED = "#b53534"
INSET_YELLOW = "#f5f0c6"

daklak = gpd.read_file(DAKLAK_SHP)
vn = gpd.read_file(VN_SHP)
dak_lak_row = vn[vn["ten_tinh"] == "Đắk Lắk"]
vn_outline = vn.dissolve()  # national boundary, for the international-border line style below

xmin, ymin, xmax, ymax = daklak.total_bounds
pad_x, pad_y = (xmax - xmin) * 0.12, (ymax - ymin) * 0.08
ext_xmin, ext_xmax = xmin - pad_x, xmax + pad_x
ext_ymin, ext_ymax = ymin - pad_y * 1.4, ymax + pad_y
lat0 = (ext_ymin + ext_ymax) / 2
aspect = 1.0 / math.cos(math.radians(lat0))  # y-units per x-unit, for equal-area-looking display

# ---- layout, all in inches: main map | gap | locator inset, as separate
# non-overlapping axes (never share any figure-fraction bounding box) ----
left_margin_in = 1.20     # room for latitude tick labels
top_margin_in = 0.65      # room for longitude tick labels (drawn above the frame)
bottom_margin_in = 0.20
main_w_in = 6.4
main_h_in = main_w_in * (ext_ymax - ext_ymin) * aspect / (ext_xmax - ext_xmin)

gap_in = 0.30
inset_w_in = 1.75
vxmin0, vymin0, vxmax0, vymax0 = vn.total_bounds
inset_h_in = inset_w_in * (vymax0 - vymin0) * (1.0 / math.cos(math.radians((vymin0 + vymax0) / 2))) / (vxmax0 - vxmin0)
inset_top_gap_in = 0.0  # inset's own top aligns with the main map's top

right_margin_in = 0.15

total_w_in = left_margin_in + main_w_in + gap_in + inset_w_in + right_margin_in
total_h_in = top_margin_in + main_h_in + bottom_margin_in

fig = plt.figure(figsize=(total_w_in, total_h_in))
fig.patch.set_facecolor("white")


def rect(x_in, y_top_in, w_in, h_in):
    return [x_in / total_w_in, 1 - (y_top_in + h_in) / total_h_in, w_in / total_w_in, h_in / total_h_in]


# -----------------------------------------------------------------
# Main map: districts + neighboring-province context + graticule
# -----------------------------------------------------------------
ax = fig.add_axes(rect(left_margin_in, top_margin_in, main_w_in, main_h_in))

# Real terrain relief (Copernicus DEM GLO-30 hillshade) as the backdrop for
# everything outside the study area, instead of a flat gray fill. Figure 8's
# ArcGIS-exported main panel already shows real satellite imagery for its
# out-of-province context; this closes the same gap for Figure 1, matching
# the real-terrain-backdrop convention used in comparable published figures
# (e.g. Maskell et al. 2021, Fig. 3). A semi-transparent light overlay mutes
# the raw grayscale relief so it reads as backdrop, not competing content,
# and keeps the graticule/labels legible on top of it.
ax.imshow(HILLSHADE, extent=[HS_LON_MIN, HS_LON_MAX, HS_LAT_MIN, HS_LAT_MAX],
          cmap="gray", vmin=0.05, vmax=0.62, origin="upper", zorder=0, interpolation="bilinear")
ax.add_patch(Rectangle((ext_xmin, ext_ymin), ext_xmax - ext_xmin, ext_ymax - ext_ymin,
                        facecolor="#f7f5ee", alpha=0.35, edgecolor="none", zorder=0.4))

vn.plot(ax=ax, facecolor="none", edgecolor="#8a8a8a", linewidth=0.8, zorder=1)
# International boundary (Vietnam / Cambodia): drawn as its own dash-dot
# black line, per standard map convention where a national border is
# visually distinct from -- and heavier than -- ordinary first-level
# administrative (provincial) boundaries, which stay the light gray line
# above. Without this, Vietnam's western edge here reads as just another
# internal province line, which is not correct map symbology.
vn_outline.boundary.plot(ax=ax, edgecolor="#2a2a2a", linewidth=1.7,
                          linestyle=(0, (5, 1.5, 1, 1.5)), zorder=2.5)
daklak.plot(ax=ax, facecolor=FILL_RED, edgecolor="#1a1a1a", linewidth=0.8, zorder=3)
daklak.dissolve().boundary.plot(ax=ax, edgecolor="#000000", linewidth=2.4, zorder=4)

ax.set_xlim(ext_xmin, ext_xmax)
ax.set_ylim(ext_ymin, ext_ymax)
ax.set_aspect(aspect)

# ---- graticule: real lon/lat ticks, same visual style as Figure 8 ----
# fontsize=13 here (not a flat x1.5 of the original 10.5) targets the SAME
# effective printed size as Figure 8's own graticule labels (16pt at Fig8's
# 0.517 embed-shrink ratio = ~8.3pt effective) via Figure 1's own, milder
# shrink ratio (~0.64): 8.3 / 0.64 = ~13. Canceling each figure's shrink
# independently (what a flat factor does) does not make the two figures'
# effective sizes match each other -- only targeting the same effective
# size does, which is what "one shared visual language" actually requires.
lon_ticks = [round(t, 2) for t in np.arange(107.75, ext_xmax, 0.25) if ext_xmin <= t <= ext_xmax]
lat_ticks = [round(t, 2) for t in np.arange(12.25, ext_ymax, 0.25) if ext_ymin <= t <= ext_ymax]
ax.set_xticks(lon_ticks)
ax.set_yticks(lat_ticks)
ax.set_xticklabels([f"{v:.2f}\N{DEGREE SIGN}E" for v in lon_ticks], fontsize=13, color="#222222")
ax.set_yticklabels([f"{v:.2f}\N{DEGREE SIGN}N" for v in lat_ticks], fontsize=13, color="#222222")
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
ax.tick_params(axis="both", direction="out", length=7, width=1.5, color="#222222")
for spine in ax.spines.values():
    spine.set_edgecolor("#222222")
    spine.set_linewidth(1.5)
ax.grid(False)

# ---- scale bar (bottom-left): bracket-tick style, numbers above the bar,
# tall ticks at major divisions + short ticks at minor divisions -- matching
# the surveyor's-scale convention used in Maskell et al. (2021) Fig. 3,
# rather than the plain three-tick bar used here previously. ----
km_per_deg_lon = 111.32 * math.cos(math.radians(lat0))
# bar_km=40 (not 20): at the corrected 13pt label size, "0"/"10"/"20 km"
# collided ("1(20 km") -- the same fix already applied to Figure 8's scale
# bar (bar_km 20->40), which also keeps both figures' bars showing the same
# round numbers.
bar_km = 40
n_major = 2       # major divisions (0, half, full), each labeled and tall
n_minor_per_major = 2  # short unlabeled ticks between each pair of major ticks
bar_deg = bar_km / km_per_deg_lon
x0 = ext_xmin + (ext_xmax - ext_xmin) * 0.03
y0 = ext_ymin + (ext_ymax - ext_ymin) * 0.06
span_y = ext_ymax - ext_ymin
tall_h = span_y * 0.018
short_h = span_y * 0.009

ax.plot([x0, x0 + bar_deg], [y0, y0], color="black", linewidth=2.1, zorder=9)
n_total = n_major * n_minor_per_major
for i in range(n_total + 1):
    frac = i / n_total
    xt = x0 + bar_deg * frac
    is_major = i % n_minor_per_major == 0
    ax.plot([xt, xt], [y0, y0 + (tall_h if is_major else short_h)], color="black",
            linewidth=2.1 if is_major else 1.5, zorder=9)
    if is_major:
        km_val = bar_km * frac
        label = "0" if km_val == 0 else f"{km_val:g}"
        ax.text(xt, y0 + tall_h + span_y * 0.006,
                label + ("" if km_val == 0 else (" km" if km_val == bar_km else "")),
                fontsize=18, ha="center", va="bottom", family="Arial", zorder=9,
                path_effects=[pe.withStroke(linewidth=4.5, foreground="white", alpha=0.85)])

# ---- terrain-data attribution (required by the Copernicus DEM license) ----
ax.text(ext_xmax - (ext_xmax - ext_xmin) * 0.02, ext_ymin + (ext_ymax - ext_ymin) * 0.012,
        "Terrain: Copernicus DEM GLO-30 (ESA)", fontsize=10, color="#4a4a4a",
        ha="right", va="bottom", family="Arial", zorder=9,
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white", alpha=0.85)])

# ---- north arrow (bottom-right): 8-point compass rose, matching Figure 8 ----
compass_cx = ext_xmax - (ext_xmax - ext_xmin) * 0.09
compass_cy = ext_ymin + (ext_ymax - ext_ymin) * 0.14
draw_compass_rose(ax, compass_cx, compass_cy, r_long=(ext_xmax - ext_xmin) * 0.032)

# ---- provincial capital marker: orients readers unfamiliar with the region ----
bmt_lon, bmt_lat = 108.0293, 12.6581  # Buon Ma Thuot district centroid
ax.plot(bmt_lon, bmt_lat, marker="s", markersize=7.5, markerfacecolor="white",
        markeredgecolor="#000000", markeredgewidth=1.5, zorder=10)
ax.text(bmt_lon + (ext_xmax - ext_xmin) * 0.014, bmt_lat, "Buon Ma Thuot",
        fontsize=13, style="italic", ha="left", va="center", color="#000000",
        family="Arial", zorder=10,
        path_effects=[pe.withStroke(linewidth=4.0, foreground="white")])

# -----------------------------------------------------------------
# Vietnam locator inset: its own column to the right of the main map,
# never overlapping the main map's axes bounding box.
# -----------------------------------------------------------------
ax_inset = fig.add_axes(rect(left_margin_in + main_w_in + gap_in, top_margin_in + inset_top_gap_in,
                              inset_w_in, inset_h_in))
vn.plot(ax=ax_inset, facecolor=INSET_YELLOW, edgecolor="#8a8a6a", linewidth=0.4, zorder=1)
dak_lak_row.plot(ax=ax_inset, facecolor=FILL_RED, edgecolor="#5a1a1a", linewidth=0.8, zorder=2)
ax_inset.set_xlim(vxmin0, vxmax0)
ax_inset.set_ylim(vymin0, vymax0)
ax_inset.set_aspect(1.0 / math.cos(math.radians((vymin0 + vymax0) / 2)))
ax_inset.set_xticks([])
ax_inset.set_yticks([])
for spine in ax_inset.spines.values():
    spine.set_edgecolor("#222222")
    spine.set_linewidth(1.4)

fig.savefig(OUT_DIR / "Figure1_StudyArea.png", facecolor="white")
fig.savefig(OUT_DIR / "Figure1_StudyArea.pdf", facecolor="white")
print("Saved Figure1_StudyArea.png/.pdf, size(in):", total_w_in, total_h_in)

"""
Figure 8, stage 2: composite the ArcGIS exports from figure8_01_export_arcgis.py
into the final embedded figure. Pure matplotlib/PIL, no arcpy -- run with
the project's regular Python environment.

Inputs (produced by stage 1, expected alongside this script, plus stage 0's
background-lightening pass):
  fig8_main_highres_lightened.png (stage 0 output: fig8_main_highres.png with
    the out-of-province satellite imagery desaturated + lightened to match
    Figure 1's hillshade backdrop -- see figure8_00_lighten_background.py),
  graticule_points.json, inset_windows.json,
  inset_<key>_classified.png, inset_<key>_imagery.png

Output:
  Figure8_LandCoverMap.png / .pdf   all five panels a-e in one image

Design notes:
- Individually lettered panels (a-e), per international multi-panel figure
  convention: a = province map; b/c = Cu Mgar imagery/classified; d/e =
  Krong Pac imagery/classified (grouped by site rather than strict
  row-major reading order, since the imagery-classified pair for one site
  is the meaningful analytic unit). Letters are lowercase to match every
  other multi-panel figure in this manuscript.
- The main map's two locator badges use plain numerals (1, 2), deliberately
  decoupled from the a-e panel letters, so a badge is never mistaken for a
  panel label pointing at itself; the caption cross-references numeral to
  letter range (e.g. "window 1, panels b-c").
- The graticule is drawn as labeled tick marks in dedicated top/left margins
  outside the map frame, not as lines crossing the classified content.
- The legend is drawn as its own vector panel (not the ArcGIS-rendered
  legend, which is hidden at export time in stage 1) so its text stays real,
  selectable vector text and its colors are guaranteed to match the values
  below exactly.
- Only minimal in-image text remains (panel letters, district names,
  "High-resolution imagery"/"Classified output" sub-labels, scale bars):
  the full prose description of each panel (selection methodology, imagery
  source and attribution, coordinate reference system, jurisdictional
  disclaimer) lives in the manuscript's own editable Figure 8 caption, not
  baked into the image, per the journal's "keep the amount of text in any
  image to a minimum" guidance.
- All five panels are ONE image (not split across two embedded pictures):
  an earlier version split the province map and the four insets into two
  separate embedded pictures because a single inline picture taller than
  one printed page was found to be silently clipped, rather than
  paginated, by Word's PDF export. To keep this as one image without
  hitting that limit, the insets sit BELOW the main map sharing the map's
  own (wider, legend-inclusive) canvas width -- at a fixed final embed
  width, a wider shared canvas compresses the resulting embed height,
  which is what actually keeps the combined image comfortably under the
  page-height ceiling (uniform embedding rescales the whole canvas
  together, so it is the canvas's aspect ratio, not any absolute size
  choice within it, that sets the final embed height). This was verified
  empirically after embedding: exported PDF page image count/content
  confirmed no clipping.
- The four insets are laid out as a single row (b, c, d, e side by side)
  rather than a 2x2 block: a 2x2 arrangement, sized to fit the page-height
  budget, could not use more than about half the map's own width, leaving
  large empty gutters on both sides that read as visually disconnected
  from the full-width map above. A single row naturally spans nearly the
  full width with no such gutters, and -- being only one square tall
  instead of two -- comfortably fits a larger individual square than the
  2x2 arrangement could at the same safe embed height. Cu Mgar (b, c) and
  Krong Pac (d, e) each get a district-name header spanning their own pair
  of columns, the same grouped-label convention used for Figure 4's
  multi-photo Intercrop coffee group.
"""

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from PIL import Image

mpl.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "savefig.dpi": 550,
})


# Shared sizing for both pieces of map furniture below, so they read as one
# consistent design rather than two independently-tuned elements. Matplotlib
# text/line sizes are always in real points/page-inches regardless of the
# data-space scale they're drawn into, so -- exactly like this file's own
# pre-existing `badge_r_px = 0.13 * base_dpi` pattern -- every geometric size
# here is first chosen in inches, then converted to this image's pixel-data
# space via base_dpi (pixels per inch), rather than picking "pixel-looking"
# numbers that silently come out a different physical size than intended.
FURNITURE_FONTSIZE = 22


def draw_compass_rose_px(ax, cx, cy, r_long_in, base_dpi, color="#000000", zorder=9):
    """Pixel-space twin of Figure 1's draw_compass_rose (figure1_compose_study_area.py):
    identical 8-point compass-rose geometry (circle + graduation ticks + an
    8-pointed star), so the two figures share one north-arrow visual
    language pixel-for-pixel. Drawn here to replace the ArcGIS layout's
    baked-in compass watermark, which was blanked out of the background
    raster in figure8_00_lighten_background.py because it went nearly
    invisible once that background was lightened to match Figure 1. Pixel
    space has y increasing downward (imshow extent=[0, w, h, 0]), so north
    (bearing 0) is placed at -y instead of +y, unlike the lon/lat version.
    """
    r_long = r_long_in * base_dpi
    r_short = r_long * 0.62
    r_in = r_long * 0.16
    r_circle = r_long * 0.78

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
        pts.append((cx + r * math.sin(theta), cy - r * math.cos(theta)))
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=color,
                          linewidth=1.4, zorder=zorder, joinstyle="miter"))

    ax.add_patch(Circle((cx, cy), r_circle, fill=False, edgecolor=color,
                         linewidth=2.0, zorder=zorder))
    for k in range(32):
        bearing = k * 11.25
        if bearing % 22.5 == 0:
            continue
        theta = math.radians(bearing)
        x0, y0 = cx + r_circle * math.sin(theta), cy - r_circle * math.cos(theta)
        x1, y1 = cx + r_circle * 1.13 * math.sin(theta), cy - r_circle * 1.13 * math.cos(theta)
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.5, zorder=zorder)

    label_r = r_long * 1.55
    for label, bearing in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
        theta = math.radians(bearing)
        ax.text(cx + label_r * math.sin(theta), cy - label_r * math.cos(theta), label,
                fontsize=FURNITURE_FONTSIZE, ha="center", va="center", color=color,
                family="Arial", zorder=zorder)


def draw_scalebar_px(ax, x0, y0, bar_km, px_per_m, base_dpi, zorder=9,
                      n_major=2, n_minor_per_major=2):
    """Pixel-space twin of Figure 1's scale bar (figure1_compose_study_area.py):
    bracket-tick style with numbers ABOVE the bar and tall ticks at major
    divisions + short ticks at minor divisions, matching the surveyor's-
    scale convention used in Maskell et al. (2021) Fig. 3 -- replacing both
    this file's own earlier three-tick-below-the-bar design and the ArcGIS
    layout's baked-in ruler, which was blanked out of the background raster
    in figure8_00_lighten_background.py (a baked, low-contrast layout
    element that went nearly invisible once the background was lightened to
    match Figure 1). (x0, y0) is the bar's left end / baseline, in this
    image's own pixel space. bar_km's on-page LENGTH is real (from px_per_m,
    the map's actual ground scale); the tick/text sizing is inch-based via
    base_dpi, matching draw_compass_rose_px's font size and stroke weight so
    the two pieces of furniture read as one design.
    """
    bar_px = bar_km * 1000 * px_per_m
    pad_in = 0.20
    tall_h_in = 0.16
    short_h_in = 0.08
    label_gap_in = 0.06
    pad = pad_in * base_dpi
    tall_h = tall_h_in * base_dpi
    short_h = short_h_in * base_dpi
    label_gap = label_gap_in * base_dpi

    ax.plot([x0, x0 + bar_px], [y0, y0], color="black", linewidth=2.4, zorder=zorder)
    n_total = n_major * n_minor_per_major
    for i in range(n_total + 1):
        frac = i / n_total
        xt = x0 + bar_px * frac
        is_major = i % n_minor_per_major == 0
        h = tall_h if is_major else short_h
        ax.plot([xt, xt], [y0, y0 - h], color="black", linewidth=2.4 if is_major else 1.6, zorder=zorder)
        if is_major:
            km_val = bar_km * frac
            label = "0" if km_val == 0 else f"{km_val:g}"
            if km_val == bar_km:
                label += " km"
            ax.text(xt, y0 - h - label_gap, label, fontsize=FURNITURE_FONTSIZE,
                    ha="center", va="bottom", family="Arial", zorder=zorder,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white", alpha=0.85)])

IN_DIR = Path(__file__).resolve().parent
OUT_DIR = IN_DIR

# Official 10-class palette -- kept in lock-step with the authoritative
# colormap source (ArcGIS_Coffee_Map/Result_2026/DakLak_2024_10class_labels_colormap/
# and its byte-identical copy under mmlab-coffeemap-daklak/data/gis_outputs/, plus
# config/paper1_config.yaml). Rubber and Partially vegetative were adjusted
# from the raw ArcGIS values (#9bd37a, #cde963) to the values below after
# both collapsed to nearly the same hue under simulated deuteranopia/
# protanopia; if this palette changes again, update all of those files too,
# plus the GEE scripts and supplementary_S10 legend CSV/JS that mirror it.
LEGEND_ENTRIES = [
    ("Sun coffee", "#8c3b00"),
    ("Intercrop coffee", "#ff8080"),
    ("Newly planted coffee", "#ffb000"),
    ("Rubber", "#4caf82"),
    ("Partially vegetative", "#d4c86a"),
    ("Rice", "#0087a8"),
    ("Other upland crops", "#f2e6b8"),
    ("Forest", "#1f4d2b"),
    ("Water", "#00c8ff"),
    ("Built", "#9a9a9a"),
]

with open(IN_DIR / "graticule_points.json") as f:
    GRAT = json.load(f)
with open(IN_DIR / "inset_windows.json") as f:
    WINDOWS = json.load(f)

XMIN, YMIN, XMAX, YMAX = GRAT["extent_utm"]

PANELS = {
    "cu_mgar": dict(win_key="cu_mgar", district="Cu Mgar", locator="1",
                     letter_imagery="b", letter_classified="c"),
    "krong_pak": dict(win_key="krong_pak", district="Krong Pac", locator="2",
                       letter_imagery="d", letter_classified="e"),
}


def utm_to_px(x, y, w, h):
    return (x - XMIN) / (XMAX - XMIN) * w, (1 - (y - YMIN) / (YMAX - YMIN)) * h


def edge_crossing(points_px, axis, target):
    for i in range(len(points_px) - 1):
        a, b = points_px[i], points_px[i + 1]
        va, vb = a[axis], b[axis]
        if va == target:
            return a
        if (va - target) * (vb - target) < 0:
            t = (target - va) / (vb - va)
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    return None


def add_scalebar(ax, img_px, ground_m):
    m_per_px = ground_m / img_px
    bar_px = 1000.0 / m_per_px
    margin = img_px * 0.04
    x0, y0 = margin, img_px - margin
    ax.add_patch(Rectangle((x0 - margin * 0.3, y0 - margin * 0.9), bar_px + margin * 0.9, margin * 1.1,
                             facecolor="white", edgecolor="none", alpha=0.72, zorder=8))
    ax.plot([x0, x0 + bar_px], [y0 - margin * 0.15] * 2, color="black", linewidth=2.2, zorder=9)
    ax.plot([x0, x0], [y0 - margin * 0.35, y0 + margin * 0.05], color="black", linewidth=1.4, zorder=9)
    ax.plot([x0 + bar_px, x0 + bar_px], [y0 - margin * 0.35, y0 + margin * 0.05], color="black", linewidth=1.4, zorder=9)
    ax.text(x0 + bar_px / 2, y0 - margin * 1.05, "1 km", fontsize=12.5, ha="center", va="bottom",
             color="black", family="Arial", zorder=9)


def compose_figure8():
    main_img = Image.open(IN_DIR / "fig8_main_highres_lightened.png").convert("RGB")
    w, h = main_img.size
    base_dpi = main_img.info.get("dpi", (400, 400))[0]

    top_margin_in, left_margin_in, right_margin_in, bottom_margin_in = 0.58, 1.05, 3.25, 0.20
    tick_len_px, label_gap_px = 57, 13

    main_w_in, main_h_in = w / base_dpi, h / base_dpi
    total_w_in = left_margin_in + main_w_in + right_margin_in
    map_block_h_in = top_margin_in + main_h_in + bottom_margin_in

    # The four insets (b-e) sit below in a single row spanning nearly the
    # full map width, instead of a 2x2 block centered under it: a 2x2
    # arrangement left large empty gutters on both sides (2 columns cannot
    # use anywhere near the full width without exceeding the page-height
    # budget), which read as visually disconnected from the full-width map
    # above. A single row naturally spans the available width -- and,
    # since it is only one square tall rather than two, comfortably fits a
    # noticeably larger square than the 2x2 arrangement could at the same
    # safe embed height.
    block_gap_in = 0.20
    row_left_margin_in, row_right_margin_in = left_margin_in, 0.20
    inset_gap_in = 0.16
    # group_label_h_in/sublabel_h_in must each be taller than the (bold) text
    # they hold -- fontsize=21/22 respectively -- with a real gap between the
    # two strips; the pre-1.667x-scale-up values (0.24/0.19in, no gap) were
    # never re-checked after that font bump, so "1. Cu Mgar"/"2. Krong Pac"
    # ended up crowding directly into the "b"/"d" panel letters below them.
    group_label_h_in, sublabel_h_in, group_sub_gap_in = 0.40, 0.34, 0.06
    row_w_in = total_w_in - row_left_margin_in - row_right_margin_in
    square_in = (row_w_in - 3 * inset_gap_in) / 4
    insets_block_h_in = group_label_h_in + group_sub_gap_in + sublabel_h_in + square_in

    total_h_in = map_block_h_in + block_gap_in + insets_block_h_in

    fig = plt.figure(figsize=(total_w_in, total_h_in))
    fig.patch.set_facecolor("white")

    def rect(x_in, y_top_in, w_in, h_in):
        return [x_in / total_w_in, 1 - (y_top_in + h_in) / total_h_in, w_in / total_w_in, h_in / total_h_in]

    # -----------------------------------------------------------------
    # Panel a: province map + graticule + inset-locator badges + legend
    # -----------------------------------------------------------------
    ax_main = fig.add_axes(rect(left_margin_in, top_margin_in, main_w_in, main_h_in))
    ax_main.imshow(main_img, extent=[0, w, h, 0], interpolation="none", zorder=0)
    ax_main.set_xlim(0, w)
    ax_main.set_ylim(h, 0)
    ax_main.axis("off")
    ax_main.add_patch(Rectangle((0, 0), w, h, fill=False, edgecolor="#222222", linewidth=1.0, zorder=6, clip_on=False))
    # Panel letter sits in the genuinely blank top-left margin corner (outside
    # the map frame), not overlaid on the map content -- keeps every panel
    # letter in this figure off the imagery/classified pixels, consistent
    # with how Figures 6, 7, 9 and 10 place their letters in chart whitespace.
    fig.text(left_margin_in * 0.28 / total_w_in, 1 - (top_margin_in * 0.55) / total_h_in, "a",
              fontsize=22, fontweight="bold", color="#000000", ha="left", va="center",
              family="Arial")

    for entry in GRAT["lat_lines"]:
        pts = [utm_to_px(x, y, w, h) for x, y in entry["points_utm"]]
        cross = edge_crossing(pts, 0, 0.0)
        if cross is None or not (0 <= cross[1] <= h):
            continue
        _, y0 = cross
        ax_main.plot([0, -tick_len_px], [y0, y0], color="#222222", linewidth=1.0, zorder=6, clip_on=False)
        ax_main.text(-tick_len_px - label_gap_px, y0, f"{entry['value']:.2f}\N{DEGREE SIGN}N",
                     fontsize=16, color="#222222", ha="right", va="center", zorder=6,
                     family="Arial", clip_on=False)

    for entry in GRAT["lon_lines"]:
        pts = [utm_to_px(x, y, w, h) for x, y in entry["points_utm"]]
        cross = edge_crossing(pts, 1, 0.0)
        if cross is None or not (0 <= cross[0] <= w):
            continue
        x0, _ = cross
        ax_main.plot([x0, x0], [0, -tick_len_px], color="#222222", linewidth=1.0, zorder=6, clip_on=False)
        ax_main.text(x0, -tick_len_px - label_gap_px, f"{entry['value']:.2f}\N{DEGREE SIGN}E",
                     fontsize=16, color="#222222", ha="center", va="bottom", zorder=6,
                     family="Arial", clip_on=False)

    badge_r_px = 0.13 * base_dpi
    for info in PANELS.values():
        x0, y0, x1, y1 = WINDOWS[info["win_key"]]["utm_extent"]
        rx0, ry0 = utm_to_px(x0, y1, w, h)
        rx1, ry1 = utm_to_px(x1, y0, w, h)
        ax_main.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False,
                                      edgecolor="#111111", linewidth=1.4, zorder=7))
        ax_main.add_patch(Circle((rx0, ry0), badge_r_px, facecolor="#3A3A3A", edgecolor="white",
                                   linewidth=1.0, zorder=8))
        ax_main.text(rx0, ry0, info["locator"], fontsize=17.5, fontweight="bold", color="white",
                     ha="center", va="center", zorder=9, family="Arial")

    # North arrow: vector compass rose, replacing the ArcGIS layout's baked-in
    # watermark compass that was blanked out of the lightened background (see
    # figure8_00_lighten_background.py) -- same style/geometry as Figure 1's,
    # via draw_compass_rose_px, positioned at that same blanked box's center.
    draw_compass_rose_px(ax_main, cx=3055, cy=2462, r_long_in=0.30, base_dpi=base_dpi, zorder=9)

    # Scale bar: vector bar in Figure 1's own style, replacing the ArcGIS
    # layout's baked-in ruler that was likewise blanked out of the
    # lightened background.
    px_per_m = w / (XMAX - XMIN)
    draw_scalebar_px(ax_main, x0=110, y0=2790, bar_km=40, px_per_m=px_per_m, base_dpi=base_dpi, zorder=9)

    ax_legend = fig.add_axes(rect(left_margin_in + main_w_in + 0.15, top_margin_in + 0.15,
                                    right_margin_in - 0.15, main_h_in - 0.3))
    ax_legend.axis("off")
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    n = len(LEGEND_ENTRIES)
    row_h = 1.0 / (n + 0.5)
    y = 1.0 - row_h * 0.5
    swatch_w, swatch_h = 0.16, row_h * 0.62
    for label, color in LEGEND_ENTRIES:
        ax_legend.add_patch(Rectangle((0.0, y - swatch_h / 2), swatch_w, swatch_h, facecolor=color,
                                        edgecolor="#333333", linewidth=0.6, transform=ax_legend.transAxes, zorder=2))
        ax_legend.text(swatch_w + 0.10, y, label, fontsize=17, color="#000000", ha="left", va="center",
                        family="Arial", transform=ax_legend.transAxes, zorder=2)
        y -= row_h

    # -----------------------------------------------------------------
    # Panels b-e: local-example insets, one row of four (imagery and
    # classified output for each district, side by side) spanning nearly
    # the full map width -- district name is a group header spanning its
    # own pair of columns, matching the grouped-label convention already
    # used for Figure 4's multi-photo Intercrop coffee group.
    # -----------------------------------------------------------------
    y0_in = map_block_h_in + block_gap_in
    columns = [
        ("cu_mgar", "imagery", "b"),
        ("cu_mgar", "classified", "c"),
        ("krong_pak", "imagery", "d"),
        ("krong_pak", "classified", "e"),
    ]
    col_x = [row_left_margin_in + i * (square_in + inset_gap_in) for i in range(4)]

    for group_start, win_key in ((0, "cu_mgar"), (2, "krong_pak")):
        info = PANELS[win_key]
        gx0 = col_x[group_start]
        gw = 2 * square_in + inset_gap_in
        ax_glbl = fig.add_axes(rect(gx0, y0_in, gw, group_label_h_in))
        ax_glbl.axis("off")
        ax_glbl.set_xlim(0, 1)
        ax_glbl.set_ylim(0, 1)
        ax_glbl.text(0.0, 0.5, f"{info['locator']}. {info['district']}", fontsize=21, fontweight="bold",
                     color="#000000", ha="left", va="center", family="Arial")

    y_sub = y0_in + group_label_h_in + group_sub_gap_in
    y_img = y_sub + sublabel_h_in
    for i, (win_key, sub_key, letter) in enumerate(columns):
        info = PANELS[win_key]
        x_in = col_x[i]
        sub_label = "High-resolution imagery" if sub_key == "imagery" else "Classified output"

        ax_sublbl = fig.add_axes(rect(x_in, y_sub, square_in, sublabel_h_in))
        ax_sublbl.axis("off")
        ax_sublbl.set_xlim(0, 1)
        ax_sublbl.set_ylim(0, 1)
        # Panel letter sits in the label strip's own blank space, above the
        # italic sub-label, rather than overlaid on the photo/classified-map
        # content below it.
        ax_sublbl.text(0.0, 0.5, letter, fontsize=22, fontweight="bold", color="#000000",
                        ha="left", va="center", family="Arial")
        ax_sublbl.text(0.16, 0.5, sub_label, fontsize=14, style="italic", color="#333333",
                        ha="left", va="center", family="Arial")

        sub_img = Image.open(IN_DIR / f"inset_{win_key}_{sub_key}.png").convert("RGB")
        ax_img = fig.add_axes(rect(x_in, y_img, square_in, square_in))
        ax_img.imshow(sub_img, interpolation="none", zorder=0)
        ax_img.axis("off")
        ax_img.add_patch(Rectangle((0, 0), sub_img.size[0], sub_img.size[1], fill=False,
                                     edgecolor="black", linewidth=1.3, transform=ax_img.transData, zorder=5))
        add_scalebar(ax_img, sub_img.size[0], 5000.0)

    fig.savefig(OUT_DIR / "Figure8_LandCoverMap.png", facecolor="white")
    fig.savefig(OUT_DIR / "Figure8_LandCoverMap.pdf", facecolor="white")
    print("Saved Figure8_LandCoverMap.png/.pdf, size(in):", total_w_in, total_h_in)


if __name__ == "__main__":
    compose_figure8()

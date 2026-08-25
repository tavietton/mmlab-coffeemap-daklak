"""
Figure 4: field observation photographs collected in November-December 2025
to refine interpretation keys and verify stable land-cover classes.
Pure matplotlib/PIL, no arcpy.

Inputs: mmlab-coffeemap-daklak/data/figure4_field_survey_photos/*.jpg
  Seven original field photographs. Extracted from the original PowerPoint
  source (figure_field_survey/Figures.pptx, slide 2) and matched to their
  class labels via the slide XML's shape positions -- the three Intercrop
  coffee photos and the single-class photos were located this way, not by
  visual guesswork. Two of the seven (Durian, Rubber) exist only inside
  that PowerPoint; the other five are also byte-identical to files under
  data/field_survey_raw_photos/original_image/ (verified by MD5), so this
  folder is now the single, complete, correctly labeled source set for the
  figure.

Output: results/figures/Figure4_FieldSurvey.png / .pdf

Design notes:
- Replaces the previous version's PowerPoint-style yellow-highlighter
  caption boxes with plain bold-text labels on a white ground, consistent
  with the typography used in every other figure in this manuscript.
- Every photo is center-cropped (not stretched) to a single fixed aspect
  ratio shared by both rows, so all seven cells read as one uniform grid
  regardless of each phone photo's native orientation/proportions.
- Layout: row 1 = Sun coffee + the three Intercrop coffee examples (four
  equal-width cells); row 2 = Newly planted coffee, Durian and Rubber
  (three cells, each widened so the row spans the SAME total width as row
  1 -- not left-aligned at row 1's narrower cell width, which previously
  left a visibly empty fourth slot). Durian is a companion crop shown for
  context in the intercrop system, not one of the ten mapped land-cover
  classes itself.
- No panel letters: these are illustrative field photographs, not a set
  of analytic sub-panels cross-referenced by letter in the caption text.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

mpl.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    # Photographic content, not line/text charts -- 300 dpi is standard print
    # quality for photos and keeps the PDF from embedding excessive raster
    # data (550 dpi here produced a >100 MB PDF for no visible quality gain).
    "savefig.dpi": 300,
})

SRC_DIR = Path(__file__).resolve().parents[2] / "data" / "figure4_field_survey_photos"
OUT_DIR = Path(__file__).resolve().parent


def crop_to_aspect(img, target_aspect):
    """Center-crop (never stretch) an image to a target width/height ratio."""
    w, h = img.size
    cur_aspect = w / h
    if cur_aspect > target_aspect:
        new_w = round(h * target_aspect)
        x0 = (w - new_w) // 2
        return img.crop((x0, 0, x0 + new_w, h))
    else:
        new_h = round(w / target_aspect)
        y0 = (h - new_h) // 2
        return img.crop((0, y0, w, y0 + new_h))


ROW1 = [
    ("01_sun_coffee.jpg", "Sun coffee"),
    ("02a_intercrop_coffee.jpg", None),
    ("02b_intercrop_coffee.jpg", None),
    ("02c_intercrop_coffee.jpg", None),
]
ROW1_GROUP_LABEL = "Intercrop coffee (with durian, macadamia, pepper)"
ROW2 = [
    ("03_newly_planted_coffee.jpg", "Newly planted coffee"),
    ("04_durian.jpg", "Durian"),
    ("05_rubber.jpg", "Rubber"),
]

N_COLS_ROW1 = 4
CELL_W_IN, LABEL_H_IN, GAP_IN = 1.86, 0.32, 0.06
IMG_H_IN = 1.46
# Outer margin on all four sides so the grid never touches the embedded
# picture's own edges (the picture frame itself provides no whitespace).
MARGIN_IN = 0.10

row1_total_w_in = N_COLS_ROW1 * CELL_W_IN + (N_COLS_ROW1 - 1) * GAP_IN
# Row 2 has 3 cells but spans the same total width as row 1's 4 cells, so
# each row-2 cell is proportionally wider than a row-1 cell. Row height is
# kept EQUAL between the two rows (not scaled up with the wider cell,
# which previously made row 2 visibly taller than row 1); the wider row-2
# cell instead gets a wider center-crop aspect ratio at the same height.
row2_cell_w_in = (row1_total_w_in - 2 * GAP_IN) / 3
row2_img_h_in = IMG_H_IN

grid_w_in = row1_total_w_in
grid_h_in = IMG_H_IN + LABEL_H_IN + row2_img_h_in + LABEL_H_IN + GAP_IN
total_w_in = grid_w_in + 2 * MARGIN_IN
total_h_in = grid_h_in + 2 * MARGIN_IN

fig = plt.figure(figsize=(total_w_in, total_h_in))
fig.patch.set_facecolor("white")


def rect(x_in, y_top_in, w_in, h_in):
    return [x_in / total_w_in, 1 - (y_top_in + h_in) / total_h_in, w_in / total_w_in, h_in / total_h_in]


def draw_photo(fname, x_in, y_in, w_in, h_in):
    img = ImageOps.exif_transpose(Image.open(SRC_DIR / fname)).convert("RGB")
    img = crop_to_aspect(img, w_in / h_in)
    # Downsample before handing to imshow: matplotlib's PDF backend embeds
    # the raw pixel array at its native resolution regardless of
    # savefig.dpi, so an un-resized ~4000 px phone photo bloats the PDF by
    # tens of MB per image. ~1000 px on the long edge is far more than this
    # cell size needs even at high print dpi.
    max_dim = 1000
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    ax = fig.add_axes(rect(x_in, y_in, w_in, h_in))
    ax.imshow(img, interpolation="none", zorder=0, aspect="auto")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="#222222", linewidth=1.0,
                             transform=ax.transAxes, zorder=5))


def draw_label(text, x_in, y_in, w_in, h_in, fontsize=11):
    ax_lbl = fig.add_axes(rect(x_in, y_in, w_in, h_in))
    ax_lbl.axis("off")
    ax_lbl.set_xlim(0, 1)
    ax_lbl.set_ylim(0, 1)
    ax_lbl.text(0.5, 0.5, text, fontsize=fontsize, fontweight="bold", color="#000000",
                ha="center", va="center", family="Arial")


# Row 1: four equal-width photos
for col, (fname, _) in enumerate(ROW1):
    x_in = MARGIN_IN + col * (CELL_W_IN + GAP_IN)
    draw_photo(fname, x_in, MARGIN_IN, CELL_W_IN, IMG_H_IN)

draw_label("Sun coffee", MARGIN_IN, MARGIN_IN + IMG_H_IN, CELL_W_IN, LABEL_H_IN)
group_x0 = MARGIN_IN + 1 * (CELL_W_IN + GAP_IN)
group_w = 3 * CELL_W_IN + 2 * GAP_IN
draw_label(ROW1_GROUP_LABEL, group_x0, MARGIN_IN + IMG_H_IN, group_w, LABEL_H_IN)

# Row 2: three wider photos spanning the same total width as row 1
row2_y = MARGIN_IN + IMG_H_IN + LABEL_H_IN
for col, (fname, label) in enumerate(ROW2):
    x_in = MARGIN_IN + col * (row2_cell_w_in + GAP_IN)
    draw_photo(fname, x_in, row2_y, row2_cell_w_in, row2_img_h_in)
    draw_label(label, x_in, row2_y + row2_img_h_in, row2_cell_w_in, LABEL_H_IN)

fig.savefig(OUT_DIR / "Figure4_FieldSurvey.png", facecolor="white")
fig.savefig(OUT_DIR / "Figure4_FieldSurvey.pdf", facecolor="white")
print("Saved Figure4_FieldSurvey.png/.pdf, size(in):", total_w_in, total_h_in)

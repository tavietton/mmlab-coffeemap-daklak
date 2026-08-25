"""
Figure 3: very-high-resolution reference examples for the ten land-cover and
coffee production system classes. Pure matplotlib/PIL, no arcpy.

Inputs: mmlab-coffeemap-daklak/data/figure3_class_reference_images/NN_class_name.png
  Ten Google Earth Pro screenshots, one per class, matching the Class ID
  order in Table 2. Extracted from the original PowerPoint source
  (figure_field_survey/Figures.pptx, slide 1) via its slide XML shape
  positions, matched to each class-name text box's on-slide coordinates --
  not by visual guesswork. The "Intercrop coffee" screenshot had a stray
  red Google Earth Pro map-pin icon baked into the pixels (a UI capture
  artifact, not a real map feature); it was removed by patching that small
  region with a same-size crop of the surrounding canopy texture from
  elsewhere in the same image, feathered at the patch edges.

Output: results/figures/Figure3_ReferenceExamples.png / .pdf

Design notes:
- Replaces the previous version's PowerPoint-style yellow-highlighter
  caption boxes with plain bold-text labels on a white ground, consistent
  with the typography used in every other figure in this manuscript.
- Every thumbnail is center-cropped (not stretched, not letterboxed) to the
  same fixed aspect ratio before display, so all ten cells read as a clean,
  uniform grid -- the source screenshots have differing native aspect
  ratios and would otherwise either distort or leave uneven blank margins.
- Keeps the "N. Class name" numbering (not letters): the numbers match
  Table 2's Class ID column one-to-one, so a reader can cross-reference a
  thumbnail directly to its formal class definition. This is a numbered
  index into a table, not a lettered sub-panel scheme, so it does not need
  to follow the a/b/c/d/e panel-letter convention used in the analytic
  figures (Figures 6-10).
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


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

mpl.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    # Photographic content, not line/text charts -- 300 dpi is standard print
    # quality for photos and keeps the PDF from embedding excessive raster
    # data (550 dpi here produced a >100 MB PDF for no visible quality gain).
    "savefig.dpi": 300,
})

SRC_DIR = Path(__file__).resolve().parents[2] / "data" / "figure3_class_reference_images"
OUT_DIR = Path(__file__).resolve().parent

CLASSES = [
    ("01_sun_coffee.png", "1. Sun coffee"),
    ("02_intercrop_coffee.png", "2. Intercrop coffee"),
    ("03_newly_planted_coffee.png", "3. Newly planted coffee"),
    ("04_rubber.png", "4. Rubber"),
    ("05_partially_vegetative.png", "5. Partially vegetative"),
    ("06_rice.png", "6. Rice"),
    ("07_other_upland_crops.png", "7. Other upland crops"),
    ("08_forest.png", "8. Forest"),
    ("09_water.png", "9. Water"),
    ("10_built.png", "10. Built"),
]

N_COLS, N_ROWS = 5, 2
CELL_W_IN, IMG_H_IN, LABEL_H_IN, GAP_IN = 1.62, 1.10, 0.30, 0.06
# Outer margin on all four sides so the grid never touches the embedded
# picture's own edges (the picture frame itself provides no whitespace).
MARGIN_IN = 0.10

grid_w_in = N_COLS * CELL_W_IN + (N_COLS - 1) * GAP_IN
grid_h_in = N_ROWS * (IMG_H_IN + LABEL_H_IN) + (N_ROWS - 1) * GAP_IN
total_w_in = grid_w_in + 2 * MARGIN_IN
total_h_in = grid_h_in + 2 * MARGIN_IN

fig = plt.figure(figsize=(total_w_in, total_h_in))
fig.patch.set_facecolor("white")


def rect(x_in, y_top_in, w_in, h_in):
    return [x_in / total_w_in, 1 - (y_top_in + h_in) / total_h_in, w_in / total_w_in, h_in / total_h_in]


for i, (fname, label) in enumerate(CLASSES):
    row, col = divmod(i, N_COLS)
    x_in = MARGIN_IN + col * (CELL_W_IN + GAP_IN)
    y_in = MARGIN_IN + row * (IMG_H_IN + LABEL_H_IN + GAP_IN)

    img = Image.open(SRC_DIR / fname).convert("RGB")
    img = crop_to_aspect(img, CELL_W_IN / IMG_H_IN)
    ax_img = fig.add_axes(rect(x_in, y_in, CELL_W_IN, IMG_H_IN))
    ax_img.imshow(img, interpolation="none", zorder=0, aspect="auto")
    ax_img.axis("off")
    ax_img.add_patch(Rectangle((0, 0), img.size[0], img.size[1], fill=False,
                                 edgecolor="#222222", linewidth=1.0, transform=ax_img.transData, zorder=5))

    ax_lbl = fig.add_axes(rect(x_in, y_in + IMG_H_IN, CELL_W_IN, LABEL_H_IN))
    ax_lbl.axis("off")
    ax_lbl.set_xlim(0, 1)
    ax_lbl.set_ylim(0, 1)
    ax_lbl.text(0.5, 0.5, label, fontsize=10.5, fontweight="bold", color="#000000",
                ha="center", va="center", family="Arial")

fig.savefig(OUT_DIR / "Figure3_ReferenceExamples.png", facecolor="white")
fig.savefig(OUT_DIR / "Figure3_ReferenceExamples.pdf", facecolor="white")
print("Saved Figure3_ReferenceExamples.png/.pdf, size(in):", total_w_in, total_h_in)

"""
Figure 8, stage 0: lighten the out-of-province satellite-imagery background
in fig8_main_highres.png (the ArcGIS Pro export from figure8_01_export_arcgis.py)
so it visually matches Figure 1's light-gray hillshade backdrop, keeping
both study/output map figures in the same tonal register. Only pixels
OUTSIDE the Dak Lak province boundary are touched (desaturated + blended
toward white); pixels inside the province -- the actual classified
land-cover map -- are left byte-for-byte untouched.

Run with any Python environment that has geopandas + rasterio installed:
  python figure8_00_lighten_background.py

Inputs:
  fig8_main_highres.png   (ArcGIS export, 3307x2826 px, UTM zone 49N extent
                            from graticule_points.json's "extent_utm")
  graticule_points.json   (for the exact extent_utm the export covers)
  ArcGIS_Coffee_Map/Shapefile/daklak_details/daklak.shp

Output:
  fig8_main_highres_lightened.png  (same size/extent, only the background changed)
"""
import json
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_bounds

IN_DIR = Path(__file__).resolve().parent
DAKLAK_SHP = IN_DIR.parents[2] / "ArcGIS_Coffee_Map/Shapefile/daklak_details/daklak.shp"

with open(IN_DIR / "graticule_points.json") as f:
    GRAT = json.load(f)
XMIN, YMIN, XMAX, YMAX = GRAT["extent_utm"]

img = Image.open(IN_DIR / "fig8_main_highres.png").convert("RGB")
w, h = img.size
arr = np.asarray(img).astype(np.float32)

daklak = gpd.read_file(DAKLAK_SHP).to_crs(epsg=32649)
province = daklak.dissolve().geometry.iloc[0]

transform = from_bounds(XMIN, YMIN, XMAX, YMAX, w, h)
inside_mask = rasterize([(province, 1)], out_shape=(h, w), transform=transform,
                         fill=0, dtype="uint8").astype(bool)

# Desaturate (luminosity) + blend toward a warm-white, matching Figure 1's
# hillshade treatment (vmin/vmax-compressed grayscale + ~35% white overlay).
luminosity = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
gray = np.stack([luminosity] * 3, axis=-1)

WHITE = np.array([247.0, 245.0, 238.0])  # matches Figure 1's "#f7f5ee" overlay tint
LIGHTEN_ALPHA = 0.55
lightened_bg = gray * (1 - LIGHTEN_ALPHA) + WHITE * LIGHTEN_ALPHA
# also lift the gray's own floor/ceiling the same way Figure 1's vmin/vmax did,
# so shadow areas don't stay heavy and dark
lightened_bg = 255.0 - (255.0 - lightened_bg) * 0.55

out = np.where(inside_mask[..., None], arr, lightened_bg)

# The ArcGIS layout bakes its map furniture -- the north-arrow watermark AND
# the scale bar -- directly into this same raster, rather than compositing
# them as a separate top layer the way GIS apps normally treat map chrome
# (e.g. AgriGIS: basemap/data layers underneath, scale bar and north arrow
# always pinned to the front as their own layer, so they stay legible no
# matter what the basemap looks like). Baked-in, low-contrast furniture is
# exactly why both went nearly invisible once this background was
# lightened: the naive per-pixel lighten formula above derives "lightened
# background" FROM each pixel's own original luminosity, so a solid black
# symbol just lightens into a pale copy of itself, not a clean background.
# Fixed properly here by following the same layering principle: blank both
# regions out of the raster entirely, then figure8_02_compose.py draws
# fresh, fully opaque vector replacements on top -- a compass rose in
# Figure 1's exact style (draw_compass_rose_px) and a scale bar in Figure
# 1's exact style (white backing chip + tick marks + km labels) -- so both
# figures' map furniture is genuinely one shared front layer, not two
# different baked-in watermarks happening to look similar.
FURNITURE_BOXES = {
    # This box's job is to fully ERASE the ORIGINAL ArcGIS-baked compass
    # watermark -- a separate concern from how big the NEW vector compass
    # drawn on top in figure8_02_compose.py happens to be (currently
    # r_long_in=0.30, label_r=1.55x, centered at (3055, 2462) there -- keep
    # these in sync if either changes). Sized to that compass's label reach
    # with a modest margin, not larger: an earlier, much larger box (needed
    # for a since-reverted, oversized r_long_in=0.50 compass) covered so
    # much area that Telea inpainting -- which reconstructs texture inward
    # from the box's boundary -- produced a visibly smooth, low-detail
    # gradient toward the box's center instead of genuine terrain texture,
    # itself reading as an unwanted "background patch" behind the compass.
    # Kept tight here so inpainting has a short distance to fill and stays
    # texture-faithful throughout.
    "compass": (2740, 2130, 3307, 2794),
    "scalebar": (40, 2600, 830, 2826),
}


def blank_region(out, lightened_bg, inside_mask, box, pad=180):
    """Blank a furniture box using proper inpainting (OpenCV's Telea
    algorithm), not a manually clone-stamped patch. Two earlier, hand-rolled
    approaches both failed in ways inpainting is specifically designed to
    avoid: (1) filling with a single flat averaged color left a smooth,
    texture-free rectangle clearly visible against genuinely textured
    terrain; (2) clone-stamping a same-size rectangular patch from the
    nearest clean neighboring direction fixed the flatness but still left a
    visible seam at the box edge, since a straight patch swap is not the
    same as texture-continuous reconstruction. Real inpainting propagates
    structure inward from the region's boundary, so it blends seamlessly
    without needing a separate blur-based feather pass.

    The classified (inside-province) pixels are ALSO marked as "to fill" in
    the inpainting mask (even though they are not part of the target box and
    keep their own original colors in the final output) purely so the
    algorithm never draws on them as a texture *source* -- otherwise a box
    near the province boundary could still pull classified colors in as
    reference texture, which is exactly what happened with the clone-stamp
    approach. Restricted to a padded crop around the box, not the whole
    image, since inpainting cost scales with the region processed.
    """
    x0, y0, x1, y1 = box
    img_h, img_w = inside_mask.shape
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(img_w, x1 + pad), min(img_h, y1 + pad)

    crop = lightened_bg[cy0:cy1, cx0:cx1].astype(np.uint8)
    crop_inside = inside_mask[cy0:cy1, cx0:cx1]

    fill_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    fill_mask[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = 255
    fill_mask[crop_inside] = 255

    inpainted = cv2.inpaint(crop, fill_mask, inpaintRadius=8, flags=cv2.INPAINT_TELEA)

    result = out.copy()
    result[y0:y1, x0:x1] = inpainted[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0].astype(np.float32)
    return result


for box in FURNITURE_BOXES.values():
    out = blank_region(out, lightened_bg, inside_mask, box)

out = np.clip(out, 0, 255).astype(np.uint8)

Image.fromarray(out, mode="RGB").save(IN_DIR / "fig8_main_highres_lightened.png")
print("saved fig8_main_highres_lightened.png,", "inside px:", inside_mask.sum(), "outside px:", (~inside_mask).sum())

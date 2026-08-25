"""
Figure 8, stage 1: ArcGIS Pro exports (province map + local-example insets).

Must be run with ArcGIS Pro's own Python environment (arcpy is not available
in the project's regular scientific-Python environment, and is only
available with a licensed ArcGIS Pro installation, not installable from
PyPI):
  <path-to-ArcGIS-Pro>/bin/Python/envs/arcgispro-py3/python.exe figure8_01_export_arcgis.py

Requires ArcGIS_Coffee_Map/Coffee_DakLak.aprx (layout "Figure 7.
Daklak_coffee_map_24"), the classified raster
(ArcGIS_Coffee_Map/Result_2026/DakLak_2024_10class_RF_corrTop25.tif), the
district boundary shapefile, and a signed-in ArcGIS Online session (for the
"Hinh anh Ve tinh The gioi" / World Imagery basemap used in the local-example
insets). ArcGIS_Coffee_Map/ is a sibling of this repository under the same
parent folder, not part of this repository -- available from the
corresponding author on request, same as the other excluded raw data.

Produces, into OUT_DIR:
  fig8_main_highres.png              province map, tight extent, no legend
  graticule_points.json              lat/lon graticule tick geometry
  inset_windows.json                 the two selected 5x5 km example windows
  inset_<key>_classified.png         classified-output crop, square 1:1
  inset_<key>_imagery.png            true high-resolution imagery crop, square 1:1

Design notes:
- The map frame/page is resized to the province's own bounding-box aspect
  ratio (instead of the original fixed 210x148mm landscape page, which was
  padded ~31% wider than the study area just to leave room for an in-map
  legend) so the displayed extent stays close to the actual study area, per
  the journal's own "maps should not show an area larger than the study
  area's bounding box" guidance. The legend itself moves to a separate
  vector panel drawn in stage 2 (figure8_02_compose.py), not rendered here.
- Each local-example window is chosen by minimizing the L1 distance between
  the window's 10-class composition and its home district's own province-
  wide-averaged composition, i.e. a *representative* sample, not the most
  visually striking (highest single-class purity) window -- see
  `find_representative_windows` below.
- Inset windows are exported through a temporarily-squared (1:1) map frame,
  not the page's native landscape aspect, so a genuinely square 5x5 km
  ground extent is not anisotropically stretched into a non-square image.
"""

import json
from pathlib import Path

import arcpy

OUT_DIR = Path(__file__).resolve().parent
_ARCGIS_DIR = Path(__file__).resolve().parents[3] / "ArcGIS_Coffee_Map"  # sibling of this repo

APRX_PATH = str(_ARCGIS_DIR / "Coffee_DakLak.aprx")
RASTER_PATH = str(_ARCGIS_DIR / "Result_2026" / "DakLak_2024_10class_RF_corrTop25.tif")
SHP_PATH = str(_ARCGIS_DIR / "Shapefile" / "daklak_details" / "daklak.shp")
UTM = arcpy.SpatialReference(32649)   # WGS 1984 UTM Zone 49N
WGS84 = arcpy.SpatialReference(4326)

# Province raster's own native bounds (confirmed via arcpy.Describe), used as
# the tight main-map extent target, plus a small breathing-room buffer.
RASTER_XMIN, RASTER_YMIN, RASTER_XMAX, RASTER_YMAX = 118440, 1345830, 282340, 1485040
EXTENT_BUFFER_M = 3000

# The two districts featured as enlarged local-example panels: chosen for
# being nationally well-known, high-coffee-area districts (Cu Mgar: 2nd
# highest official district coffee area in the province, classic coffee
# district; Krong Pac: known for coffee/durian intercropping) -- see the
# response-to-reviewers letter for the full rationale.
FEATURED_DISTRICTS = {"cu_mgar": "Cư M'gar", "krong_pak": "Krông Pắc"}
WINDOW_PX = 500      # 500 px * 10 m native cell size = 5 km window
WINDOW_STEP_PX = 150  # sliding-window scan stride


def find_representative_windows():
    """For each featured district, scan 5x5 km candidate windows and return
    the one whose 10-class composition is closest (L1 distance) to that
    district's own overall composition -- a representative, not extremal,
    sample."""
    class_names = {1: "sun_coffee", 2: "intercrop_coffee", 3: "newly_planted_coffee", 4: "rubber",
                   5: "partially_vegetative", 6: "rice", 7: "other_upland_crops", 8: "forest",
                   9: "water", 10: "built"}

    ras_full = arcpy.Raster(RASTER_PATH)
    cell = ras_full.meanCellWidth
    origin_x, origin_y_top = ras_full.extent.XMin, ras_full.extent.YMax
    full_w, full_h = ras_full.width, ras_full.height

    def utm_to_px(x, y):
        return int(round((x - origin_x) / cell)), int(round((origin_y_top - y) / cell))

    def read_window(px_x0, px_y0, px_w, px_h):
        px_x0 = max(0, min(px_x0, full_w - 1))
        px_y0 = max(0, min(px_y0, full_h - 1))
        px_w = min(px_w, full_w - px_x0)
        px_h = min(px_h, full_h - px_y0)
        ll = arcpy.Point(origin_x + px_x0 * cell, origin_y_top - (px_y0 + px_h) * cell)
        return arcpy.RasterToNumPyArray(RASTER_PATH, ll, px_w, px_h, nodata_to_value=0)

    def district_extent_utm(name):
        with arcpy.da.SearchCursor(SHP_PATH, ["District", "SHAPE@"]) as cur:
            for dname, shape in cur:
                if dname == name:
                    e = shape.projectAs(UTM).extent
                    return e.XMin, e.YMin, e.XMax, e.YMax
        raise ValueError(name)

    results = {}
    for key, district_name in FEATURED_DISTRICTS.items():
        xmin, ymin, xmax, ymax = district_extent_utm(district_name)
        dx0, dy1 = utm_to_px(xmin, ymax)
        dx1, dy0 = utm_to_px(xmax, ymin)
        dw, dh = dx1 - dx0, dy0 - dy1

        arr = read_window(dx0, dy1, dw, dh)
        valid = arr[arr != 0]
        target = {class_names[k]: (valid == k).sum() / valid.size for k in range(1, 11)}

        best = None
        for y0 in range(0, max(dh - WINDOW_PX, 1), WINDOW_STEP_PX):
            for x0 in range(0, max(dw - WINDOW_PX, 1), WINDOW_STEP_PX):
                abs_x0, abs_y0 = dx0 + x0, dy1 + y0
                win = read_window(abs_x0, abs_y0, WINDOW_PX, WINDOW_PX)
                win_valid = win[win != 0]
                if win_valid.size < 0.9 * win.size:
                    continue
                fracs = {class_names[k]: (win_valid == k).sum() / win_valid.size for k in range(1, 11)}
                dist = sum(abs(fracs[c] - target[c]) for c in class_names.values())
                if best is None or dist < best["dist"]:
                    ux0 = origin_x + abs_x0 * cell
                    uy1 = origin_y_top - abs_y0 * cell
                    best = {"dist": dist, "fracs": fracs,
                            "utm_extent": [ux0, uy1 - WINDOW_PX * cell, ux0 + WINDOW_PX * cell, uy1]}
        results[key] = best
        print(key, district_name, "-> dist", round(best["dist"], 3), "fracs",
              {k: round(v, 3) for k, v in best["fracs"].items() if v > 0.08})
    return results


def compute_graticule_points(extent_utm):
    xmin, ymin, xmax, ymax = extent_utm
    corners = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]
    bounds = []
    for x, y in corners:
        pt = arcpy.PointGeometry(arcpy.Point(x, y), UTM).projectAs(WGS84)
        bounds.append((pt.firstPoint.Y, pt.firstPoint.X))
    lat_min, lat_max = min(b[0] for b in bounds), max(b[0] for b in bounds)
    lon_min, lon_max = min(b[1] for b in bounds), max(b[1] for b in bounds)

    lat_lines = sorted({round(round(v * 4) / 4, 2)
                         for v in (lat_min + 0.25 * i for i in range(1, 20)) if lat_min < v < lat_max})
    lon_lines = sorted({round(round(v * 4) / 4, 2)
                         for v in (lon_min + 0.25 * i for i in range(1, 20)) if lon_min < v < lon_max})

    def wgs_to_utm(lat, lon):
        pt = arcpy.PointGeometry(arcpy.Point(lon, lat), WGS84).projectAs(UTM)
        return pt.firstPoint.X, pt.firstPoint.Y

    n_samples = 25
    result = {"extent_utm": list(extent_utm), "lat_lines": [], "lon_lines": []}
    lon_lo, lon_hi = lon_min - 0.3, lon_max + 0.3
    for lat in lat_lines:
        pts = [wgs_to_utm(lat, lon_lo + (lon_hi - lon_lo) * i / (n_samples - 1)) for i in range(n_samples)]
        result["lat_lines"].append({"value": lat, "points_utm": [list(p) for p in pts]})
    lat_lo, lat_hi = lat_min - 0.3, lat_max + 0.3
    for lon in lon_lines:
        pts = [wgs_to_utm(lat_lo + (lat_hi - lat_lo) * i / (n_samples - 1), lon) for i in range(n_samples)]
        result["lon_lines"].append({"value": lon, "points_utm": [list(p) for p in pts]})
    return result


def export_main_map(extent_utm):
    aprx = arcpy.mp.ArcGISProject(APRX_PATH)
    lyt = [l for l in aprx.listLayouts() if "Figure 7" in l.name][0]
    mf = [e for e in lyt.listElements() if e.type == "MAPFRAME_ELEMENT"][0]
    legend = [e for e in lyt.listElements() if e.type == "LEGEND_ELEMENT"][0]

    orig_extent = mf.camera.getExtent()
    orig_legend_vis = legend.visible
    orig_mf_w, orig_mf_h = mf.elementWidth, mf.elementHeight
    orig_page_w, orig_page_h = lyt.pageWidth, lyt.pageHeight

    xmin, ymin, xmax, ymax = extent_utm
    aspect = (xmax - xmin) / (ymax - ymin)
    frame_w_mm = 210.0
    frame_h_mm = round(frame_w_mm / aspect, 4)

    lyt.pageHeight = frame_h_mm
    mf.elementWidth = frame_w_mm
    mf.elementHeight = frame_h_mm
    legend.visible = False
    mf.camera.setExtent(arcpy.Extent(xmin, ymin, xmax, ymax))

    out_png = str(OUT_DIR / "fig8_main_highres.png")
    lyt.exportToPNG(out_png, resolution=400)
    print("exported", out_png)

    mf.camera.setExtent(orig_extent)
    mf.elementWidth, mf.elementHeight = orig_mf_w, orig_mf_h
    lyt.pageWidth, lyt.pageHeight = orig_page_w, orig_page_h
    legend.visible = orig_legend_vis


def export_insets(windows):
    aprx = arcpy.mp.ArcGISProject(APRX_PATH)
    lyt = [l for l in aprx.listLayouts() if "Figure 7" in l.name][0]
    mf = [e for e in lyt.listElements() if e.type == "MAPFRAME_ELEMENT"][0]
    m = mf.map

    raster_layer = [l for l in m.listLayers() if l.isRasterLayer][0]
    boundary_layers = [l for l in m.listLayers() if l.isFeatureLayer]
    imagery_layers = [l for l in m.listLayers() if not l.isFeatureLayer and not l.isRasterLayer]
    if not imagery_layers:
        m.addBasemap("Hình ảnh")  # World Imagery basemap; ArcGIS Online sign-in required
        imagery_layers = [l for l in m.listLayers() if not l.isFeatureLayer and not l.isRasterLayer]
    imagery_layer = imagery_layers[0]

    orig_extent = mf.camera.getExtent()
    orig_mf_w, orig_mf_h = mf.elementWidth, mf.elementHeight
    prior_raster_vis = raster_layer.visible
    prior_boundary_vis = {l.longName: l.visible for l in boundary_layers}
    prior_imagery_vis = imagery_layer.visible

    square_mm = 150.0
    mf.elementWidth = square_mm
    mf.elementHeight = square_mm

    for key, win in windows.items():
        x0, y0, x1, y1 = win["utm_extent"]
        assert abs((x1 - x0) - (y1 - y0)) < 1.0, f"{key} window is not square"
        mf.camera.setExtent(arcpy.Extent(x0, y0, x1, y1))

        raster_layer.visible = True
        for l in boundary_layers:
            l.visible = True
        imagery_layer.visible = False
        out_classified = str(OUT_DIR / f"inset_{key}_classified.png")
        mf.exportToPNG(out_classified, resolution=600)
        print("exported", out_classified)

        raster_layer.visible = False
        for l in boundary_layers:
            l.visible = False
        imagery_layer.visible = True
        out_imagery = str(OUT_DIR / f"inset_{key}_imagery.png")
        mf.exportToPNG(out_imagery, resolution=600)
        print("exported", out_imagery)

    mf.elementWidth, mf.elementHeight = orig_mf_w, orig_mf_h
    mf.camera.setExtent(orig_extent)
    raster_layer.visible = prior_raster_vis
    for l in boundary_layers:
        l.visible = prior_boundary_vis[l.longName]
    imagery_layer.visible = prior_imagery_vis


if __name__ == "__main__":
    windows = find_representative_windows()
    with open(OUT_DIR / "inset_windows.json", "w") as f:
        json.dump(windows, f, indent=2)

    main_extent = (RASTER_XMIN - EXTENT_BUFFER_M, RASTER_YMIN - EXTENT_BUFFER_M,
                   RASTER_XMAX + EXTENT_BUFFER_M, RASTER_YMAX + EXTENT_BUFFER_M)
    graticule = compute_graticule_points(main_extent)
    with open(OUT_DIR / "graticule_points.json", "w") as f:
        json.dump(graticule, f)

    export_main_map(main_extent)
    export_insets(windows)
    print("Stage 1 done. Run figure8_02_compose.py (regular Python env) next.")

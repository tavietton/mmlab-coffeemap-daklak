"""Spatial-block cross-validation helpers: coordinate/feature discovery and a
random-vs-spatial-group CV comparison used as a spatial-autocorrelation
robustness diagnostic."""
import json
import types

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold

NON_FEATURE_COLUMNS = {
    "system:index", ".geo", "class_id", "lon", "lat", "longitude", "latitude",
    "x", "y", "X", "Y", "split", "gridId", "grid_id", "rnd", "random",
}


def _parse_geo_lon_lat(geo_value):
    try:
        geo = json.loads(geo_value) if isinstance(geo_value, str) else geo_value
        coords = geo.get("coordinates")
        return coords[0], coords[1]
    except Exception:
        return np.nan, np.nan


def find_coordinate_columns(df):
    """Return (x_col, y_col) column names for sample coordinates. Looks for
    common explicit lon/lat-style columns first; falls back to parsing a
    GEE-style `.geo` GeoJSON Point column into new `_lon_geo`/`_lat_geo`
    columns (added to `df` in place)."""
    lon_candidates = ["lon", "longitude", "x", "X", "POINT_X"]
    lat_candidates = ["lat", "latitude", "y", "Y", "POINT_Y"]
    x_col = next((c for c in lon_candidates if c in df.columns), None)
    y_col = next((c for c in lat_candidates if c in df.columns), None)
    if x_col and y_col:
        return x_col, y_col

    if ".geo" in df.columns:
        lonlat = df[".geo"].apply(_parse_geo_lon_lat)
        df["_lon_geo"] = [v[0] for v in lonlat]
        df["_lat_geo"] = [v[1] for v in lonlat]
        if df["_lon_geo"].notna().any():
            return "_lon_geo", "_lat_geo"

    return None, None


def infer_feature_columns(df, label_col, coord_cols=None):
    """Return the numeric predictor columns in `df`, excluding the label,
    coordinate columns, and known non-feature identifier columns."""
    coord_cols = coord_cols or []
    drop = set(NON_FEATURE_COLUMNS) | {label_col} | set(coord_cols)
    drop |= {c for c in df.columns if str(c).startswith("Unnamed")}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def compare_random_vs_spatial_cv(df, label_col, feature_cols, x_col, y_col,
                                  n_splits=5, random_state=42, n_estimators=500,
                                  n_spatial_groups=20):
    """Compare stratified random k-fold CV against spatial group k-fold CV
    (groups formed by K-means clustering of sample coordinates), as a
    diagnostic for spatial-autocorrelation inflation of validation accuracy.

    Returns an object with `.status` ('ok' or 'error'), `.message`,
    `.fold_scores` (per-fold results, one row per method x fold), and
    `.summary` (mean/SD per method).
    """
    result = types.SimpleNamespace(status="ok", message="", fold_scores=None, summary=None)

    work = df.dropna(subset=[x_col, y_col, label_col] + list(feature_cols)).copy()
    if work[label_col].nunique() < 2 or len(work) < n_splits * 2:
        result.status = "error"
        result.message = "Not enough valid rows/classes to run cross-validation."
        return result

    X = work[feature_cols].to_numpy()
    y = work[label_col].to_numpy()
    coords = work[[x_col, y_col]].to_numpy()

    n_groups = min(n_spatial_groups, max(2, len(work) // 10))
    groups = KMeans(n_clusters=n_groups, random_state=random_state, n_init=10).fit_predict(coords)

    rows = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        acc = float((clf.predict(X[te]) == y[te]).mean())
        rows.append({"method": "Random stratified CV", "fold": fold, "overall_accuracy": acc})

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        acc = float((clf.predict(X[te]) == y[te]).mean())
        rows.append({"method": "Spatial group CV", "fold": fold, "overall_accuracy": acc})

    fold_scores = pd.DataFrame(rows)
    summary = (
        fold_scores.groupby("method")["overall_accuracy"]
        .agg(overall_accuracy_mean="mean", overall_accuracy_sd="std")
        .reset_index()
    )

    result.fold_scores = fold_scores
    result.summary = summary
    result.message = (
        f"Compared {n_splits}-fold random stratified CV against {n_splits}-fold "
        f"spatial group CV over {n_groups} K-means coordinate clusters."
    )
    return result

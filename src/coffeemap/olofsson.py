"""Olofsson et al. (2014) area-weighted, error-adjusted area/accuracy estimation."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def error_matrix_counts(y_true, y_pred, labels):
    """Olofsson-style error matrix: rows = map (predicted) class i, columns =
    reference (true) class j; n_ij = samples mapped as i that are truly j."""
    return confusion_matrix(y_pred, y_true, labels=labels)


def _row_normalized(error_matrix):
    n_i_dot = error_matrix.sum(axis=1)
    K = error_matrix.shape[0]
    q = np.zeros((K, K))
    for i in range(K):
        if n_i_dot[i] > 0:
            q[i, :] = error_matrix[i, :] / n_i_dot[i]
    return q, n_i_dot


def area_adjustment(error_matrix, mapped_areas, labels, class_names=None, total_area=None):
    """Area-weighted, error-adjusted class-area estimates with 95% CIs
    (Olofsson et al. 2014, Eqs. 22-26). Returns a DataFrame with one row per
    class plus a TOTAL row."""
    labels = list(labels)
    K = len(labels)
    q, n_i_dot = _row_normalized(np.asarray(error_matrix, dtype=float))

    if isinstance(mapped_areas, dict):
        A_i = np.array([mapped_areas[c] for c in labels], dtype=float)
    else:
        A_i = np.asarray(mapped_areas, dtype=float)
    A_total = float(total_area) if total_area is not None else float(A_i.sum())
    W_i = A_i / A_total

    p_j = (W_i[:, None] * q).sum(axis=0)
    A_j_hat = A_total * p_j

    var_p_j = np.zeros(K)
    for j in range(K):
        s = 0.0
        for i in range(K):
            if n_i_dot[i] > 1:
                s += (W_i[i] ** 2) * q[i, j] * (1 - q[i, j]) / (n_i_dot[i] - 1)
        var_p_j[j] = s
    ci95_area = 1.96 * np.sqrt(var_p_j) * A_total

    rows = []
    for idx, c in enumerate(labels):
        mapped = A_i[idx]
        adjusted = A_j_hat[idx]
        bias = adjusted - mapped
        rows.append({
            "class_id": c,
            "class_name": (class_names or {}).get(c, str(c)),
            "mapped_area_ha": mapped,
            "mapped_area_pct": 100 * mapped / A_total,
            "adjusted_area_ha": adjusted,
            "adjusted_area_ci95_ha": ci95_area[idx],
            "adjusted_area_pct": 100 * adjusted / A_total,
            "bias_ha": bias,
            "bias_pct": (100 * bias / mapped) if mapped else np.nan,
        })
    df = pd.DataFrame(rows)
    total_row = pd.DataFrame([{
        "class_id": "—", "class_name": "TOTAL",
        "mapped_area_ha": A_total, "mapped_area_pct": 100.0,
        "adjusted_area_ha": A_total, "adjusted_area_ci95_ha": np.nan,
        "adjusted_area_pct": 100.0, "bias_ha": 0.0, "bias_pct": 0.0,
    }])
    return pd.concat([df, total_row], ignore_index=True)


def adjusted_accuracy(error_matrix, mapped_areas, labels, class_names=None, total_area=None):
    """Adjusted overall accuracy, user's accuracy (= row-normalized diagonal)
    and producer's accuracy (Olofsson 2014 Eqs. 23-24) with 95% CIs."""
    labels = list(labels)
    K = len(labels)
    q, n_i_dot = _row_normalized(np.asarray(error_matrix, dtype=float))

    if isinstance(mapped_areas, dict):
        A_i = np.array([mapped_areas[c] for c in labels], dtype=float)
    else:
        A_i = np.asarray(mapped_areas, dtype=float)
    A_total = float(total_area) if total_area is not None else float(A_i.sum())
    W_i = A_i / A_total

    ua = np.diag(q)
    p_j = (W_i[:, None] * q).sum(axis=0)
    pa = np.array([
        (W_i[j] * q[j, j] / p_j[j]) if p_j[j] > 0 else np.nan
        for j in range(K)
    ])

    oa = float((W_i * ua).sum())
    var_oa = float(sum(
        (W_i[i] ** 2) * ua[i] * (1 - ua[i]) / (n_i_dot[i] - 1)
        for i in range(K) if n_i_dot[i] > 1
    ))
    oa_ci95 = 1.96 * np.sqrt(var_oa) * 100

    rows = []
    for idx, c in enumerate(labels):
        rows.append({
            "class_id": c,
            "class_name": (class_names or {}).get(c, str(c)),
            "adjusted_ua_pct": 100 * ua[idx],
            "adjusted_pa_pct": 100 * pa[idx],
            "n_reference": int(n_i_dot[idx]),
        })
    df = pd.DataFrame(rows)
    return df, oa * 100, oa_ci95


def binary_coffee_area_adjustment(error_matrix, mapped_areas, labels, coffee_class_ids, total_area=None):
    """Collapse a multi-class error matrix into a binary coffee/non-coffee
    error matrix and apply the same area-adjustment procedure."""
    labels = list(labels)
    coffee_idx = [i for i, c in enumerate(labels) if c in coffee_class_ids]
    other_idx = [i for i, c in enumerate(labels) if c not in coffee_class_ids]
    em = np.asarray(error_matrix, dtype=float)

    binary_em = np.zeros((2, 2))
    binary_em[0, 0] = em[np.ix_(coffee_idx, coffee_idx)].sum()
    binary_em[0, 1] = em[np.ix_(coffee_idx, other_idx)].sum()
    binary_em[1, 0] = em[np.ix_(other_idx, coffee_idx)].sum()
    binary_em[1, 1] = em[np.ix_(other_idx, other_idx)].sum()

    if isinstance(mapped_areas, dict):
        coffee_area = sum(mapped_areas[c] for c in labels if c in coffee_class_ids)
        other_area = sum(mapped_areas[c] for c in labels if c not in coffee_class_ids)
    else:
        A = np.asarray(mapped_areas, dtype=float)
        coffee_area = A[coffee_idx].sum()
        other_area = A[other_idx].sum()

    return area_adjustment(
        binary_em, {"coffee": coffee_area, "non_coffee": other_area},
        ["coffee", "non_coffee"], total_area=total_area,
    )

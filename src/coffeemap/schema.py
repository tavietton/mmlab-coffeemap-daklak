"""Column-name discovery and lightweight schema-validation helpers for tables
whose exact column names vary across export versions."""
import re

import numpy as np
import pandas as pd


def find_first_column(df, candidates, required=False, role=None):
    """Return the first column name in `candidates` that exists in df.columns,
    or None (or raise if required=True) if none match. `role` is an optional
    human-readable label used only to make the error message clearer."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        label = f" for {role}" if role else ""
        raise KeyError(f"None of the candidate columns {candidates}{label} found in {list(df.columns)}")
    return None


# Alias: some notebooks import this helper as `detect_column`.
detect_column = find_first_column


def normalise_col(s):
    """Lowercase and strip non-alphanumeric characters, for fuzzy column-name matching."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def detect_label_columns(df, true_candidates, pred_candidates):
    """Return (true_label_col, predicted_label_col) detected from candidate
    name lists."""
    return find_first_column(df, true_candidates), find_first_column(df, pred_candidates)


def assert_class_ids(values, expected_ids, label="values"):
    """Raise ValueError if any value in `values` is outside `expected_ids`."""
    observed = set(pd.Series(values).dropna().astype(int).unique().tolist())
    invalid = observed - set(expected_ids)
    if invalid:
        raise ValueError(f"{label} contains unexpected class IDs: {sorted(invalid)}")
    return True


def class_count_table(df, label_col, class_info=None):
    """Return a tidy class_id/class_name/n DataFrame of value counts."""
    counts = df[label_col].value_counts().sort_index()
    return pd.DataFrame({
        "class_id": counts.index,
        "class_name": [(class_info or {}).get(int(i), str(i)) for i in counts.index],
        "n": counts.to_numpy(),
    })


def warn_if_balanced(df, label_col, tolerance=0.01):
    """Return True (and print a note) if class counts are equal within
    `tolerance` of the mean count -- i.e. a deliberately balanced design
    (relevant since balanced validation sets are not map-stratified)."""
    counts = df[label_col].value_counts()
    is_balanced = bool((counts.std() / counts.mean()) < tolerance) if counts.mean() else False
    if is_balanced:
        print(
            f"Note: '{label_col}' classes are balanced ({counts.mean():.0f} rows/class). "
            "This is a deliberately balanced design, not a map-stratified sample; "
            "see the Olofsson area-adjustment caveats."
        )
    return is_balanced


def extract_probability_columns(df, prefix="p_class_"):
    """Return the list of columns holding per-class predicted probabilities."""
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    return sorted(cols)


def assert_probability_matrix(df, prob_cols, atol=1e-2):
    """Raise ValueError if the given probability columns do not sum to ~1 per row."""
    if not prob_cols:
        raise ValueError("No probability columns provided.")
    row_sums = df[prob_cols].sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        bad = int((~np.isclose(row_sums, 1.0, atol=atol)).sum())
        raise ValueError(f"{bad} row(s) of probability columns {prob_cols} do not sum to 1.")
    return True

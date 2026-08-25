"""Classification-performance and prediction-uncertainty metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score


def overall_metrics(y_true, y_pred):
    """Overall accuracy, Cohen's Kappa, and macro F1 for a classification result."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    oa = float(np.mean(y_true == y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {"overall_accuracy": oa, "kappa": kappa, "macro_f1": macro_f1}


def classification_summary(y_true, y_pred, class_info=None):
    """Per-class producer's accuracy (recall), user's accuracy (precision),
    and F1-score, as a tidy DataFrame (one row per class)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    rows = []
    for i, cls in enumerate(labels):
        support = cm[i, :].sum()
        pa = cm[i, i] / support if support > 0 else np.nan          # producer's accuracy = recall
        col_total = cm[:, i].sum()
        ua = cm[i, i] / col_total if col_total > 0 else np.nan       # user's accuracy = precision
        f1 = (2 * pa * ua / (pa + ua)) if (pa and ua and (pa + ua) > 0) else np.nan
        rows.append({
            "class_id": cls,
            "class_name": (class_info or {}).get(cls, str(cls)),
            "producers_accuracy": pa,
            "users_accuracy": ua,
            "f1_score": f1,
            "n_reference": int(support),
        })
    return pd.DataFrame(rows)


def shannon_entropy(probabilities):
    """Normalised Shannon entropy (H / log2(K)) for each row of class
    probabilities. `probabilities` is an (n_samples, n_classes) array."""
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    k = p.shape[1]
    h = -(p * np.log2(p)).sum(axis=1)
    return h / np.log2(k)


def probability_margin(probabilities):
    """Top-1 minus top-2 class probability for each row."""
    p = np.sort(np.asarray(probabilities, dtype=float), axis=1)[:, ::-1]
    return p[:, 0] - p[:, 1]

"""Validation-prediction audit helper (imported by every notebook's shared
bootstrap cell; not directly called by any of the current notebook bodies)."""
import pandas as pd


def audit_validation_predictions(y_true, y_pred, class_ids=None):
    """Quick PASS/WARN/ERROR audit of a validation prediction set: checks
    that predicted and true class IDs are both within the expected class set
    and that no rows are missing predictions."""
    issues = []
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    if y_true.isna().any() or y_pred.isna().any():
        issues.append({"level": "ERROR", "message": "Missing true or predicted class values."})
    else:
        issues.append({"level": "PASS", "message": "No missing true/predicted class values."})

    if class_ids is not None:
        bad_true = set(y_true.dropna().astype(int)) - set(class_ids)
        bad_pred = set(y_pred.dropna().astype(int)) - set(class_ids)
        if bad_true or bad_pred:
            issues.append({"level": "ERROR", "message": f"Unexpected class IDs: true={bad_true}, pred={bad_pred}"})
        else:
            issues.append({"level": "PASS", "message": "All class IDs within expected set."})

    if len(y_true) != len(y_pred):
        issues.append({"level": "ERROR", "message": "True and predicted arrays have different lengths."})

    return pd.DataFrame(issues)

"""File discovery and table read/write helpers."""
from pathlib import Path
import pandas as pd


def find_file(filename_or_candidates, search_dirs=None, required=True, label=None):
    """Search for a file by name (or the first existing name among several
    candidates) across a list of candidate directories.

    Accepts either a single filename or a list of candidate filenames as the
    first argument, for compatibility with both calling styles used across
    the pipeline notebooks.
    """
    if search_dirs is None:
        search_dirs = [Path(".")]
    search_dirs = [Path(d) for d in search_dirs]

    candidates = filename_or_candidates
    if isinstance(candidates, (str, Path)):
        candidates = [candidates]

    for fname in candidates:
        for d in search_dirs:
            p = d / fname
            if p.exists():
                return p

    if required:
        raise FileNotFoundError(
            f"Could not find {label or candidates} in any of: {search_dirs}"
        )
    return None


def read_table(path, **kwargs):
    """Read a CSV or Excel table into a DataFrame based on its extension."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, **kwargs)
    return pd.read_csv(path, encoding=kwargs.pop("encoding", "utf-8-sig"), **kwargs)


def write_table(df, path, index=False, **kwargs):
    """Write a DataFrame to CSV or Excel based on the destination extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(path, index=index, **kwargs)
    else:
        df.to_csv(path, index=index, encoding=kwargs.pop("encoding", "utf-8-sig"), **kwargs)
    return path

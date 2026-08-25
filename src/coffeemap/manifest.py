"""Run-manifest tracking for reproducibility (records which notebook produced what, when)."""
import json
import platform
import sys
from datetime import datetime
from pathlib import Path


def _manifest_path(project_root, config=None):
    metadata_dir = "metadata"
    if config is not None:
        metadata_dir = config.get("paths", {}).get("metadata_dir", "metadata")
    return Path(project_root) / metadata_dir / "run_manifest.json"


def init_run_manifest(config, project_root, notebook_name):
    """Create or update <metadata_dir>/run_manifest.json (path resolved from
    config['paths']['metadata_dir']) with a run record for `notebook_name`,
    and return the manifest dict."""
    path = _manifest_path(project_root, config)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"pipeline": config.get("project", {}).get("name", "coffee_mapping"), "runs": []}

    manifest["runs"].append({
        "notebook": notebook_name,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "random_seed": config.get("project", {}).get("random_seed"),
        "notes": [],
    })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def append_manifest_note(manifest, note, project_root=None):
    """Append a free-text note to the most recent run entry and re-save if
    project_root is given."""
    if manifest.get("runs"):
        manifest["runs"][-1].setdefault("notes", []).append(str(note))
    if project_root is not None:
        path = _manifest_path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest

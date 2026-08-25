"""Configuration loading and path/class helpers for the coffee-mapping pipeline."""
from pathlib import Path
import yaml


def load_config(config_path):
    """Load the project YAML config into a plain dict."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_project_dirs(config, project_root):
    """Create (if needed) and return the project's standard output directories.

    Returns a dict of Path objects keyed by the same names used in
    config['paths'] (e.g. 'tables_dir', 'figures_dir', 'supplementary_dir',
    'metadata_dir', 'input_dir', 'interim_dir').
    """
    project_root = Path(project_root)
    paths_cfg = config.get("paths", {})
    paths = {}
    for key, rel_path in paths_cfg.items():
        p = (project_root / rel_path).resolve()
        p.mkdir(parents=True, exist_ok=True)
        paths[key] = p
    return paths


def class_info(config):
    """Return {class_id: class_name} from config['classes']['definitions']."""
    defs = config.get("classes", {}).get("definitions", {})
    return {int(k): v["name"] for k, v in defs.items()}


def class_colors(config):
    """Return {class_id: color} from config['classes']['definitions']."""
    defs = config.get("classes", {}).get("definitions", {})
    return {int(k): v.get("color", "#999999") for k, v in defs.items()}


def coffee_class_ids(config):
    """Return the list of class IDs that represent coffee production systems."""
    return list(config.get("classes", {}).get("coffee_class_ids", [1, 2, 3]))

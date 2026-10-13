"""Paths and configuration. Every module reads config/sections.json through load()."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"          # pipeline products: SVG, JSON, HTML
BUILD = ROOT / "build"      # figures for the report, frame captures


def load(path=None):
    """Read the configuration. MEDFIG_CONFIG overrides the default path."""
    p = Path(path or os.environ.get("MEDFIG_CONFIG", ROOT / "config" / "sections.json"))
    cfg = json.loads(p.read_text())
    cfg["_path"] = str(p)
    return cfg


def planes_path(cfg):
    """Path of the coronal-planes .npz. MEDFIG_PLANES overrides the configuration."""
    p = Path(os.environ.get("MEDFIG_PLANES", cfg["source"]["planes_npz"]))
    return p if p.is_absolute() else ROOT / p

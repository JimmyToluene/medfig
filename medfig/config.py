"""Paths and configuration. Each modality has one file: config/sections.json (WSI), config/mri.json (MRI), config/ct.json (CT)
and config/pet.json (PET)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"          # pipeline products: SVG, JSON, HTML (WSI in out/, the other modalities in out/<modality>/)
BUILD = ROOT / "build"      # figures for the report, frame captures

# modality -> (file in config/, environment variable for another file, environment variable for another planes file)
MODALITIES = {"wsi": ("sections.json", "MEDFIG_CONFIG", "MEDFIG_PLANES"),
              "mri": ("mri.json", "MEDFIG_MRI_CONFIG", "MEDFIG_MRI_PLANES"),
              "ct": ("ct.json", "MEDFIG_CT_CONFIG", "MEDFIG_CT_PLANES"),
              "pet": ("pet.json", "MEDFIG_PET_CONFIG", "MEDFIG_PET_PLANES")}


def load(modality="wsi"):
    """Read the configuration of a modality ("wsi", "mri", "ct" or "pet"). The file is UTF-8.

    MEDFIG_CONFIG (WSI), MEDFIG_MRI_CONFIG, MEDFIG_CT_CONFIG and MEDFIG_PET_CONFIG override the default path.
    The argument is not a path.
    """
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}: use one of {sorted(MODALITIES)}. "
                         "For another file set MEDFIG_CONFIG (WSI) or MEDFIG_<MODALITY>_CONFIG (MRI, CT, PET).")
    name, env, _ = MODALITIES[modality]
    p = Path(os.environ.get(env, ROOT / "config" / name))
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["_path"] = str(p)
    cfg["_modality"] = modality
    return cfg


def planes_path(cfg):
    """Path of the coronal-planes .npz. MEDFIG_PLANES (WSI) and MEDFIG_<MODALITY>_PLANES override the configuration."""
    p = Path(os.environ.get(MODALITIES[cfg.get("_modality", "wsi")][2], cfg["source"]["planes_npz"]))
    return p if p.is_absolute() else ROOT / p

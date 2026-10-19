"""Step P7, icons: scan thumbnails made from the PET slices of each series (scan.icons(), shared with MRI and CT).

Usage: python -m medfig.pet.icons
Input:  out/pet/<series>/slice_{k}.svg, slices.json
Output: out/pet/<series>/scan_one.svg, scan_strip.svg
"""
import sys

from .. import scan
from ..config import load
from .slices import PET_OUT


def build(cfg=None, log=print):
    cfg = cfg or load("pet")
    for se in cfg["series"]:
        scan.icons(cfg, PET_OUT / se["key"], "PET", log)


if __name__ == "__main__":
    build()
    sys.exit(0)

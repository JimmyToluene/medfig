"""Step C7, icons: scan thumbnails made from the CT slices (scan.icons(), shared with MRI and PET).

Usage: python -m medfig.ct.icons
Input:  out/ct/slice_{k}.svg, out/ct/slices.json
Output: out/ct/scan_one.svg, out/ct/scan_strip.svg
"""
import sys

from .. import scan
from ..config import load
from .slices import CT_OUT


def build(cfg=None, log=print):
    scan.icons(cfg or load("ct"), CT_OUT, "CT", log)


if __name__ == "__main__":
    build()
    sys.exit(0)

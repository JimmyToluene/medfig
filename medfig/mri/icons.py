"""Step M7, icons: scan thumbnails made from the slices, so scan k shows the whole slice k on a dark screen.

Usage: python -m medfig.mri.icons
Input:  out/mri/slice_{k}.svg, out/mri/slices.json
Output: out/mri/scan_one.svg    one scan (126 x 48 px), slice 1
        out/mri/scan_strip.svg  scans 1 to N (244 x 36 px for N = 3) and two card edges behind the last scan

The screen of scan k is a dark rounded square. An inner <svg> with the viewBox of the slices draws slice k in it.
The drawing code is scan.icons(), shared with CT and PET.
"""
import sys

from .. import scan
from ..config import load
from .slices import MRI_OUT


def build(cfg=None, log=print):
    scan.icons(cfg or load("mri"), MRI_OUT, "MRI", log)


if __name__ == "__main__":
    build()
    sys.exit(0)

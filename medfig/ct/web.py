"""Step C7: the oblique 3D stack of CT slices as a standalone web page, and the snippets to put it in another page.

Usage: python -m medfig ct web
Input:  out/ct/slice_{k}.svg, out/ct/slices.json, out/ct/scan_*.svg, config stack_3d
Output: out/ct/stack.html, out/ct/stack_snippet.html, out/ct/stack.css (scan.web(), shared with MRI and PET)
The note of the page credits the CT template and, for the marker positions, holiAtlas.
"""
import sys

from .. import scan
from ..config import load
from .slices import CT_OUT


def build(cfg=None, log=print):
    cfg = cfg or load("ct")
    scan.web(cfg, CT_OUT, "Coronal CT stack", "CT", "of the head (skull and brain, no scalp)",
             extra=f" Marker positions from {cfg['source']['marker_credit']}.", log=log)


if __name__ == "__main__":
    build()
    sys.exit(0)

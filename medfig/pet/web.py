"""Step P7: the stack page of each PET series, and the snippets to put it in another page.

Usage: python -m medfig pet web
Input:  out/pet/<series>/slice_{k}.svg, slices.json, scan_*.svg, key.svg, config stack_3d
Output: out/pet/<series>/stack.html, stack_snippet.html, stack.css (scan.web(), shared with MRI and CT)
The note of the page shows the colour key and credits the templates and, for the outline and the markers, holiAtlas.
"""
import sys

from .. import scan
from ..config import load
from .slices import PET_OUT


def build(cfg=None, log=print):
    cfg = cfg or load("pet")
    for se in cfg["series"]:
        scan.web(cfg, PET_OUT / se["key"], f"Coronal amyloid PET stack, {se['name']}", "PET", f"of the brain, {se['label']}",
                 extra=(f" Brain outline, reference region ({cfg['reference']['name']}) and marker positions from {cfg['source']['outline_credit']}."
                        f"<br><img src=\"key.svg\" alt=\"Colour key\" style=\"margin-top:8px\">"), log=log)


if __name__ == "__main__":
    build()
    sys.exit(0)

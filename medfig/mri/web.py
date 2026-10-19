"""Step M7: the oblique 3D stack of MRI slices as a standalone web page, and the snippets to put it in another page.

Usage: python -m medfig mri web
Input:  out/mri/slice_{k}.svg, out/mri/slices.json, out/mri/scan_*.svg, config stack_3d
Output: out/mri/stack.html          standalone page: Slice / Volume switch, the stack and the scan thumbnails
        out/mri/stack_snippet.html  the .brainbox markup only (for a host page)
        out/mri/stack.css           the CSS only (for a host page)

Slice shows slice 1 with its marker. Volume turns the stack to an oblique view, pushes the back slices in from
behind one after the other, then shows each number. The CSS, the markup and the page template are in stack.py
(shared with WSI); scan.web() writes the three files (shared with CT and PET).
"""
import sys

from .. import scan, stack
from ..config import load
from .slices import MRI_OUT


def css(cfg):
    return stack.css(cfg, len(cfg["slices"]))


def snippet(meta, prefix=""):
    return stack.snippet(meta, prefix, key="slices")


def build(cfg=None, log=print):
    scan.web(cfg or load("mri"), MRI_OUT, "Coronal MRI stack", "MRI", "of the skull-stripped brain", log=log)


if __name__ == "__main__":
    build()
    sys.exit(0)

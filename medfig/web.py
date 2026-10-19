"""Step S9: the oblique 3D stack as a standalone web page, and the snippets to put it in another page.

Usage: python -m medfig.web
Input:  out/section_{k}.svg, out/sections.json, out/slide_*.svg, config stack_3d
Output: out/stack.html        standalone page: Slide-level / Case-level switch, the stack and the slide icons
        out/stack_snippet.html  the .brainbox markup only (for a host page)
        out/stack.css           the CSS only (for a host page)

Slide-level shows section 1 with its block marker. Case-level turns the stack to an oblique view, pushes the
back sections in from behind one after the other, then shows each block number. The CSS, the markup and the page
template are in stack.py (shared with MRI).
"""
import json
import sys

from . import stack
from .config import OUT, load


def css(cfg):
    return stack.css(cfg, len(cfg["sections"]))


def snippet(meta, prefix=""):
    return stack.snippet(meta, prefix)


def build(cfg=None, log=print):
    cfg = cfg or load()
    meta = json.loads((OUT / "sections.json").read_text())
    (OUT / "stack.css").write_text(css(cfg))
    (OUT / "stack_snippet.html").write_text(snippet(meta) + "\n")
    n = len(meta["sections"])
    ys = ", ".join(f"{s['y_mm']:+d}" for s in meta["sections"])
    (OUT / "stack.html").write_text(stack.page(
        cfg, meta, "sections", title="Coronal stack", first="Slide-level", second="Case-level",
        one="slide_one.svg", one_alt="One attached slide", strip="slide_strip.svg", strip_alt=f"Attached slides 1 to {n}",
        note=f"Coronal sections y = {ys} mm, drawn from {meta['license']}. The sections and blocks are a schematic, not a cutting plan."))
    log("stack.html, stack.css, stack_snippet.html")


if __name__ == "__main__":
    build()
    sys.exit(0)

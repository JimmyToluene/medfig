"""Copy the products into a host web page directory (for example the neuropath demo site).

Usage: python -m medfig site SITE_DIR [--prefix brain_sec_]
Writes SITE_DIR/{prefix}{k}.svg, slide_one.svg, slide_strip.svg, and prints the .brainbox markup and the CSS.
It does not edit any page. Put the markup and the CSS into the page by hand, or with the host's own tool.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from . import web
from .config import OUT, load


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("site_dir")
    ap.add_argument("--prefix", default="brain_sec_")
    a = ap.parse_args(argv)
    site = Path(a.site_dir)
    site.mkdir(parents=True, exist_ok=True)
    meta = json.loads((OUT / "sections.json").read_text())
    for s in meta["sections"]:
        shutil.copy(OUT / s["file"], site / f"{a.prefix}{s['k']}.svg")
        s["file"] = f"{a.prefix}{s['k']}.svg"
    for f in ("slide_one.svg", "slide_strip.svg"):
        shutil.copy(OUT / f, site / f)
    print("<!-- .brainbox markup -->")
    print(web.snippet(meta))
    print("/* CSS */")
    print(web.css(load()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

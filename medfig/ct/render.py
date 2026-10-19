"""Step C8, render: capture the CT stack page (medfig/render.py) and draw figures C6 and C7 of the report.

Usage: python -m medfig.ct.render [--gif]
Output: build/frames/ct/*.png, build/fig/ct_frame_sheet.png, build/fig/ct_end_slice.png, ct_end_volume.png,
        build/ct_stack.gif (with --gif), build/fig/ct_step_6_svgs.png, build/fig/ct_step_7_icons.png
Needs Playwright with Chromium (see medfig/render.py).
"""
import sys

from .. import render, scan
from ..config import BUILD, load
from .slices import CT_OUT

CT = scan.target(CT_OUT, BUILD / "frames" / "ct", "ct_")


def report_figs(p, url, log):
    scan.capture_figs(p, url, log, load("ct"), CT_OUT, BUILD / "fig", "ct_step_6_svgs.png", "ct_step_7_icons.png")


def main(argv=None):
    return render.run(CT, report_figs, argv)


if __name__ == "__main__":
    sys.exit(main())

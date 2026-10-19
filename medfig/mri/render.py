"""Step M8, render: capture the MRI stack page. The capture functions are in medfig/render.py (shared with WSI).

Usage: python -m medfig.mri.render [--gif]
Input:  out/mri/stack.html and the files it loads (served over local HTTP), out/mri/slices.json
Output: build/frames/mri/desktop_*.png, phone_*.png        frames at fixed times after each switch
        build/fig/mri_frame_sheet.png                      slice, volume at 0.15/0.4/0.7/1.4 s, back to slice
        build/fig/mri_end_slice.png, mri_end_volume.png    stack only, transparent, 2x
        build/mri_stack.gif                                (with --gif) one full switch and back, 20 fps
        build/fig/mri_step_6_svgs.png                      figure M6: the SVG slices with their markers
        build/fig/mri_step_7_icons.png                     figure M7: the scan thumbnails, 4x

Figures M6 and M7 show the credit line of the source under the drawing ("Source: ..."), scan.capture_figs().
The end states hold the stack only, without text: put the credit line beside them.
Needs Playwright with Chromium (see medfig/render.py).
"""
import sys

from .. import render, scan
from ..config import BUILD, load
from .slices import MRI_OUT

MRI = scan.target(MRI_OUT, BUILD / "frames" / "mri", "mri_")


def report_figs(p, url, log):
    """Figures M6 and M7 of the report: they need a browser to draw the SVG files."""
    scan.capture_figs(p, url, log, load("mri"), MRI_OUT, BUILD / "fig", "mri_step_6_svgs.png", "mri_step_7_icons.png")


def main(argv=None):
    return render.run(MRI, report_figs, argv)


if __name__ == "__main__":
    sys.exit(main())

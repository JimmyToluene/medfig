"""Overview diagram of the pipeline (build/fig/pipeline_overview.png) from the step figures and the frame sheet.

Usage: python -m medfig.overview   (after figures and render)
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

from .config import BUILD, load  # noqa: E402

FIG = BUILD / "fig"


def _crop(path, frac):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    return im.crop((int(frac[0] * w), int(frac[1] * h), int(frac[2] * w), int(frac[3] * h)))


def build(cfg=None, log=print):
    cfg = cfg or load()
    num = json.loads((FIG / "numbers.json").read_text())
    pts = f"{num['contour_points_raw']} → {num['contour_points_simplified']} points"
    th = [("S1", "Stream coronal planes", "gzip, one z-layer at a time", _crop(FIG / "step_1_candidates.png", (0, 0, 1, 1))),
          ("S2", "Pick, downsample", "100 → 300 µm", _crop(FIG / "step_2_plane.png", (.33, .2, .66, 1))),
          ("S3", "Tissue mask", "blur, threshold, clean", _crop(FIG / "step_3_tissue.png", (.33, .2, .66, 1))),
          ("S4", "Grey / white threshold", "Otsu above floor, per section", _crop(FIG / "step_4_grey_white.png", (.69, .2, 1, .85))),
          ("S5", "Round, trace, simplify", pts, _crop(FIG / "step_5_contours.png", (.69, .2, 1, 1))),
          ("S6", "One frame, block markers", f"{num['mm_per_unit']:.1f} mm per unit", _crop(FIG / "step_6_frame.png", (0, .05, 1, 1))),
          ("S7", "Flat SVG sections", "grey pink, white blue", _crop(FIG / "step_7_svgs.png", (0, 0, 1, 1))),
          ("S8", "Slide icons per block", "crop at each marker", _crop(FIG / "step_8_icons.png", (0, 0, 1, 1))),
          ("S9", "Oblique 3D stack", "CSS perspective, stagger", _crop(BUILD / "frames" / "desktop_case_1400.png", (0, 0, 1, 1)))]
    fig = plt.figure(figsize=(15, 7.2), facecolor=cfg["palette"]["page"])
    H = 0.36
    pos = [(0.01, .55), (0.21, .55), (0.41, .55), (0.61, .55), (0.81, .55), (0.81, .06), (0.61, .06), (0.41, .06), (0.11, .06)]
    wid = [.18] * 8 + [.27]
    for (sid, lab, sub, im), (x, y), w in zip(th, pos, wid):
        ax = fig.add_axes([x, y + .075, w, H - .09]); ax.imshow(im); ax.axis("off")
        fig.patches.append(FancyBboxPatch((x - .004, y - .005), w + .008, H + .005, boxstyle="round,pad=0.004,rounding_size=0.01",
                                          transform=fig.transFigure, fc="none", ec="#5b43a8", lw=1.2))
        fig.text(x, y + .045, f"{sid}  {lab}", fontsize=10.5, weight="bold", color="#1a1d21")
        fig.text(x, y + .012, sub, fontsize=8.5, color="#5f5b70")

    def arr(a, b):
        fig.patches.append(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14, color="#5b43a8", lw=1.4))

    for i in range(4):
        x, y = pos[i]; arr((x + .188, y + .2), (pos[i + 1][0] - .008, y + .2))
    arr((pos[4][0] + .09, pos[4][1] - .01), (pos[5][0] + .09, pos[5][1] + H + .005))
    for i in range(5, 8):
        x, y = pos[i]; arr((x - .008, y + .2), (pos[i + 1][0] + wid[i + 1] + .008, y + .2))
    fig.text(.01, .975, "medfig: BigBrain → flat coronal sections → oblique 3D stack", fontsize=13, weight="bold", color="#1a1d21", va="top")
    fig.text(.01, .935, f"Worked example: y = {num['example_y_mm']:+d} mm. Source: {cfg['source']['license']}.", fontsize=9, color="#5f5b70", va="top")
    fig.savefig(FIG / "pipeline_overview.png", dpi=140)
    plt.close(fig)
    log("fig/pipeline_overview.png")


if __name__ == "__main__":
    build()
    sys.exit(0)

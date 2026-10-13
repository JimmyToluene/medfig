"""Figures of steps S1 to S6 for the report. Each panel calls the same functions as the pipeline.

Usage: python -m medfig.figures
Output: build/fig/step_1_candidates.png ... step_6_frame.png, build/fig/numbers.json (values quoted in the report)
The worked example is section 2 (config sections[1]); MEDFIG_EXAMPLE=k picks another section.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage import morphology  # noqa: E402
from skimage.filters import threshold_otsu  # noqa: E402

from . import sections as S  # noqa: E402
from .config import BUILD, OUT, load  # noqa: E402

FIG = BUILD / "fig"
GM, WM = (0.925, 0.776, 0.890), (0.478, 0.651, 0.863)
COLS = ["#5b43a8", "#d0458a", "#2f8f6f", "#c07a1a", "#2f6fbf"]


def _style(page):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "figure.facecolor": page,
                         "axes.facecolor": page, "savefig.facecolor": page})


def _save(fig, name, log):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(name)


def _off(ax, title):
    ax.set_title(title, loc="left")
    ax.axis("off")


def _two(t, w):
    im = np.ones(t.shape + (3,))
    im[t] = GM
    im[w] = WM
    return im


def build(cfg=None, log=print):
    cfg = cfg or load()
    _style(cfg["palette"]["page"])
    FIG.mkdir(parents=True, exist_ok=True)
    r, v = cfg["raster"], cfg["vector"]
    data, meta = S.load_planes(cfg)
    ys = meta["ys"]
    chosen = [s["y_mm"] for s in cfg["sections"]]
    ex = cfg["sections"][int(os.environ.get("MEDFIG_EXAMPLE", 2)) - 1]
    Y = ex["y_mm"]
    num = {"example_y_mm": Y, "planes_shape": list(data.shape)}

    # S1: candidate planes
    n = len(ys)
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(2.15 * cols, 2.2 * rows))
    for ax in np.atleast_1d(axs).flat:
        ax.axis("off")
    for ax, y in zip(np.atleast_1d(axs).flat, ys):
        ax.imshow(data[ys.index(y)][::-1][::8, ::8], cmap="gray", vmin=60, vmax=255)
        ax.set_title(f"y = {y:+d} mm" + ("  ✓" if y in chosen else ""), loc="left")
        if y in chosen:
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#5b43a8"); sp.set_linewidth(2.5)
    _save(fig, "step_1_candidates.png", log)

    # S2: plane and downsampling
    raw = data[ys.index(Y)][::-1]
    a = S.plane(data, meta, Y, r["downsample"])
    num.update(raw_px=[int(raw.shape[1]), int(raw.shape[0])], down_px=[int(a.shape[1]), int(a.shape[0])])
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    axs[0].imshow(raw, cmap="gray", vmin=60, vmax=255); _off(axs[0], f"2a  Raw plane, 100 µm\n{raw.shape[1]} × {raw.shape[0]} px")
    axs[1].imshow(a, cmap="gray", vmin=60, vmax=255); _off(axs[1], f"2b  Block mean {r['downsample']} × {r['downsample']}\n{a.shape[1]} × {a.shape[0]} px")
    t_raw = raw < 238
    rr, cc = np.where(t_raw)
    r0, c0 = int(rr.min() + 0.55 * (rr.max() - rr.min())), int(cc.min() + 0.02 * (cc.max() - cc.min()))
    axs[2].imshow(raw[r0:r0 + 270, c0:c0 + 360], cmap="gray", vmin=60, vmax=255)
    _off(axs[2], "2c  Zoom of 2a (lateral cortex)\nMerker: grey matter dark, white matter pale")
    _save(fig, "step_2_plane.png", log)

    # S3: tissue mask
    t0 = ndi.gaussian_filter(a, r["tissue_blur_sigma"]) < r["tissue_max_intensity"]
    tissue = S.tissue_mask(a, r)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    axs[0].imshow(t0, cmap="gray_r"); _off(axs[0], f"3a  Blur σ {r['tissue_blur_sigma']}, intensity < {r['tissue_max_intensity']}")
    axs[1].imshow(tissue, cmap="gray_r")
    _off(axs[1], f"3b  Drop specks < {r['tissue_min_object_px']}, fill holes < {r['tissue_max_hole_px']},\nopening: ventricles stay open")
    diff = np.ones(t0.shape + (3,)); diff[tissue] = (0.85, 0.85, 0.85); diff[t0 & ~tissue] = (0.85, 0.2, 0.2); diff[~t0 & tissue] = (0.2, 0.4, 0.9)
    axs[2].imshow(diff); _off(axs[2], "3c  Change from 3a to 3b\nred: removed, blue: filled")
    num["tissue_changed_px"] = int((t0 != tissue).sum())
    _save(fig, "step_3_tissue.png", log)

    # S4: grey / white
    wm0, s, otsu, thr = S.grey_white(a, tissue, r)
    num.update(example_otsu=round(otsu, 1), example_threshold=round(thr, 1))
    fig = plt.figure(figsize=(13, 4.6))
    ax0 = fig.add_subplot(1, 3, 1); ax0.imshow(np.where(tissue, s, np.nan), cmap="magma", vmin=150, vmax=240)
    _off(ax0, f"4a  Local mean inside tissue, σ {r['local_mean_sigma']}")
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.hist(s[tissue], bins=120, range=(60, 250), color="#9b8fc8"); ax1.set_xlim(60, 275)
    top = ax1.get_ylim()[1]
    ax1.axvline(r["otsu_floor"], color="#888", ls=":", lw=1)
    ax1.text(r["otsu_floor"] - 2, top * .55, f"values < {r['otsu_floor']}\n(cerebellar granular\nlayer) left out\nof Otsu", fontsize=8, color="#555", ha="right")
    ax1.axvline(otsu, color="#5b43a8", ls="--", lw=1.2); ax1.axvline(thr, color="#d0458a", lw=1.6)
    ax1.text(226, top * .8, f"threshold {thr:.0f}\n= Otsu {otsu:.0f} − {r['wm_offset']}\nabove it:\nwhite matter", fontsize=8, color="#d0458a")
    ax1.set_title("4b  Histogram of 4a: one threshold per section", loc="left"); ax1.set_xlabel("Merker intensity"); ax1.set_yticks([])
    for sp in ("top", "right", "left"):
        ax1.spines[sp].set_visible(False)
    ax2 = fig.add_subplot(1, 3, 3); ax2.imshow(_two(tissue, wm0)); _off(ax2, "4c  Grey matter pink, white matter blue\n(before rounding)")
    _save(fig, "step_4_grey_white.png", log)

    # thresholds of every section, with and without the floor
    th = []
    for sec in cfg["sections"]:
        p = S.plane(data, meta, sec["y_mm"], r["downsample"])
        tt = S.tissue_mask(p, r)
        ss = S.local_mean(p, tt, r["local_mean_sigma"])
        th.append({"y_mm": sec["y_mm"], "otsu_all": round(float(threshold_otsu(ss[tt])), 1),
                   "otsu_floor": round(float(threshold_otsu(ss[tt & (ss > r["otsu_floor"])])), 1)})
    num["thresholds"] = th
    fig, axs = plt.subplots(1, len(th), figsize=(4.3 * len(th), 3.6))
    for ax, sec, t in zip(np.atleast_1d(axs), cfg["sections"], th):
        p = S.plane(data, meta, sec["y_mm"], r["downsample"])
        tt = S.tissue_mask(p, r)
        ss = S.local_mean(p, tt, r["local_mean_sigma"])
        bad = tt & (ss > t["otsu_all"] - r["wm_offset"])
        ax.imshow(_two(tt, bad)); _off(ax, f"y = {sec['y_mm']:+d} mm, no floor: threshold {t['otsu_all'] - r['wm_offset']:.0f}")
    _save(fig, "step_4d_no_floor.png", log)

    # S5: rounding and contour simplification, zoom on the dorsal cortex
    t_r, w_r = S.round_masks(tissue, wm0, r)
    rr, cc = np.where(t_r)
    zr, zc, zh, zw = int(rr.min()) + 5, int(cc.min()) + 40, 110, 140
    sl = (slice(zr, zr + zh), slice(zc, zc + zw))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    axs[0].imshow(_two(tissue[sl], wm0[sl])); _off(axs[0], "5a  Zoom of 4c: pixel edges")
    axs[1].imshow(_two(t_r[sl], w_r[sl]))
    _off(axs[1], f"5b  Masks blurred σ {r['mask_round_sigma']}, cut at 0.5;\nwhite-matter islands < {r['wm_min_island_px']} dropped")
    axs[2].imshow(_two(t_r[sl], w_r[sl]), alpha=.45)
    nraw = nsm = 0
    for c in S.contours(t_r):
        p = S.smooth_contour(c, v)
        if p is None:
            continue
        nraw += len(c); nsm += len(p)
        axs[2].plot(c[:, 1] - zc, c[:, 0] - zr, color="#999", lw=.6)
        axs[2].plot(p[:, 0] - zc, p[:, 1] - zr, color="#5b43a8", lw=1.4)
        axs[2].plot(p[:, 0] - zc, p[:, 1] - zr, "o", ms=2.2, color="#5b43a8")
    axs[2].set_xlim(0, zw); axs[2].set_ylim(zh, 0)
    _off(axs[2], f"5c  Contour (grey) → smoothed σ {v['contour_sigma']} → Douglas–Peucker ε {v['dp_epsilon']}\nall tissue contours: {nraw} → {nsm} points")
    num.update(contour_points_raw=nraw, contour_points_simplified=nsm)
    _save(fig, "step_5_contours.png", log)

    # S6: one frame
    J = json.loads((OUT / "sections.json").read_text())
    ox, oy = J["origin_px"]
    fig, ax = plt.subplots(figsize=(6, 6))
    for sec, col in zip(cfg["sections"], COLS):
        tt, _, _ = S.masks(S.plane(data, meta, sec["y_mm"], r["downsample"]), r)
        for c in S.contours(tt):
            if len(c) > 200:
                ax.plot(c[:, 1] - ox, c[:, 0] - oy, color=col, lw=.8)
        ax.plot([], [], color=col, label=f"y = {sec['y_mm']:+d} mm")
    for sec, col in zip(J["sections"], COLS):
        x, y, w, h = sec["rect"]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ls="--", lw=1.4, color=col))
        ax.text(x + w + 4, y + 12, f"{sec['k']}: {sec['block']}\n({sec['block_mm'][0]:.0f}, {sec['block_mm'][1]:.0f}) mm", color=col, fontsize=7)
    mid = (0 - J["x0_mm"]) / J["mm_per_unit"] - ox
    ax.axvline(mid, color="#aaa", lw=.8, ls=":"); ax.text(mid + 3, 12, "midline x = 0", fontsize=7, color="#777")
    ax.set_xlim(0, J["viewBox"][2]); ax.set_ylim(J["viewBox"][3], 0); ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8, frameon=False); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"6  One frame: viewBox {J['viewBox'][2]} × {J['viewBox'][3]}, 1 unit = {J['mm_per_unit']:.1f} mm", loc="left")
    _save(fig, "step_6_frame.png", log)
    num.update(viewBox=J["viewBox"], mm_per_unit=J["mm_per_unit"],
               svg_kb={s["file"]: round((OUT / s["file"]).stat().st_size / 1024, 1) for s in J["sections"]})
    (FIG / "numbers.json").write_text(json.dumps(num, indent=1))
    log("numbers.json")


if __name__ == "__main__":
    build()
    sys.exit(0)

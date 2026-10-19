"""Step M8, figures: steps M1 to M5 for the report, and the pairing with WSI. Each panel calls the same functions as the pipeline.

Usage: python -m medfig.mri.figures
Input:  the planes .npz (config source.planes_npz), config/mri.json and out/mri/slices.json (run medfig.mri.slices first).
        For the WSI block centres and the pairing: config/sections.json and its planes .npz. No WSI file in out/ is needed.
Output: build/fig/mri_step_1_candidates.png ... mri_step_8_pairing.png, build/fig/mri_numbers.json (values quoted in the report)
The worked example is the slice with k = 2, or the first slice if no slice has k = 2. MEDFIG_MRI_EXAMPLE=k picks another slice.
The step stops if out/mri/slices.json does not hold the configured slices, y values, unit, marker centres and marker size:
run medfig.mri.slices again. It does not see a change of pad_units; `make mri` runs the steps in order.
The figures show a template and its labels. They are a schematic and do not show a patient.
"""
import itertools
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from .. import scan  # noqa: E402
from .. import sections as S  # noqa: E402
from ..config import BUILD, load  # noqa: E402
from ..figures import COLS, _off, _style  # noqa: E402
from ..vector import contours, smooth_contour  # noqa: E402
from . import slices as M  # noqa: E402

FIG = BUILD / "fig"
# panel 2b: one colour per tissue label. Labels that touch in a plane differ for normal and for colour-blind vision.
TISSUE = {0: "#ffffff", 1: "#2a78d6", 2: "#e87ba4", 3: "#eda100", 4: "#008300", 5: "#4a3aa7", 6: "#eb6834", 7: "#1baf7a", 8: "#e34948", 9: "#6b6a66"}
BASE, ADDED, CUT = (0.85, 0.85, 0.85), (0.2, 0.4, 0.9), (0.85, 0.2, 0.2)     # panel 3b, as in WSI panel 3c
MUTED = "#777777"
ZOOM = (-4, 62, 66, 88)     # panels 4a to 4c: row and column from the top left corner of the mask, then height and width (px)
KEY = dict(fontsize=7.5, frameon=False, handlelength=1.2, columnspacing=1.2)   # legend under a panel


def _save(fig, name, credit, log):
    """Write the figure with the credit line under the panels."""
    bb = fig.get_tightbbox()
    fig.text(bb.x0 / fig.get_figwidth(), (bb.y0 - 0.06) / fig.get_figheight(), f"Source: {credit}", fontsize=7, color=MUTED, va="top")
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(name)


def _paint(shape, layers):
    """White image with each (mask, colour) painted in turn."""
    im = np.ones(shape + (3,))
    for m, col in layers:
        im[m] = to_rgb(col)
    return im


def _tint(mask, col, alpha):
    """Transparent image with the mask in one colour."""
    im = np.zeros(mask.shape + (4,))
    im[mask] = to_rgb(col) + (alpha,)
    return im


def _grid(meta, cfg):
    """holiAtlas plane after slices.plane(): x (mm) of column 0, z (mm) of row 0, pixel size (mm), x step (mm) from one
    column to the next (negative in the radiological orientation). The same as in slices.build()."""
    return scan.grid(meta, cfg)


def _wsi_grid(data, meta, ds):
    """BigBrain plane after the block mean of sections.plane(): the same four values (x ascends: WSI is neurological).

    A block pixel is the mean of ds x ds voxels. Its centre is (ds - 1) / 2 voxels inside the first voxel.
    sections.build() places its frame without this half-pixel term. Only the pairing uses this grid.
    """
    vs = meta.get("vs", 0.1)
    half = vs * (ds - 1) / 2
    return meta["srow"][0][3] + half, meta["srow"][2][3] + vs * (data.shape[1] - 1) - half, vs * ds, vs * ds


def _extent(shape, grid):
    """Edges (left, right, bottom, top) in mm of an image on a grid, for imshow(extent=...). In the radiological
    orientation left > right, so the image lands at its mm place on an x axis that ascends."""
    x0, z_top, vs, dx = grid
    return x0 - dx / 2, x0 + (shape[1] - 0.5) * dx, z_top - (shape[0] - 0.5) * vs, z_top + vs / 2


def _resample(mask, src, dst, shape):
    """Nearest-neighbour copy of a mask from the grid `src` to the grid `dst`."""
    rr, cc = np.indices(shape)
    c = np.rint((dst[0] + cc * dst[3] - src[0]) / src[3]).astype(int)
    r = np.rint((src[1] - dst[1] + rr * dst[2]) / src[2]).astype(int)
    ok = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1])
    out = np.zeros(shape, bool)
    out[ok] = mask[r[ok], c[ok]]
    return out


def _sides(cfg):
    """The side letters (left of the image, right of the image): R, L in the radiological orientation."""
    return ("R", "L") if scan.x_sign(cfg) < 0 else ("L", "R")


def _dice(a, b):
    return 2 * float((a & b).sum()) / float(a.sum() + b.sum())


def _tissue_names(cfg):
    """Label -> name, from config tissue_labels.names."""
    return {int(k): n for k, n in cfg["tissue_labels"]["names"].items()}


def _traced(mask, v):
    """Pairs (traced contour (row, col), smoothed and simplified polygon (x, y)) of a mask: the contours that path_d() draws."""
    pairs = ((c, smooth_contour(c, v)) for c in contours(mask))
    return [(c, p) for c, p in pairs if p is not None]


def _mask_steps(p, c, r):
    """The steps of classes() one by one. Returns (pixels added to grey and white matter, pixels cut by the limit, intracranial volume)."""
    filled = ndi.binary_fill_holes(ndi.binary_closing(c["brain"], iterations=r["mask_close_px"]))
    icv = ndi.binary_fill_holes(p["tissue"] > 0)
    return c["mask"] & ~c["brain"], filled & ~icv, icv


def _candidates(cfg, data, meta, vmax, credit, log):
    """M1: the candidate planes."""
    ys, chosen = meta["ys"], [s["y_mm"] for s in cfg["slices"]]
    cols = min(6, len(ys))
    rows = (len(ys) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(2.15 * cols, 2.2 * rows))
    for ax in np.atleast_1d(axs).flat:
        ax.axis("off")
    for ax, y in zip(np.atleast_1d(axs).flat, ys):
        ax.imshow(M.plane(data, meta, y, cfg)["t1w"], cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(f"y = {y:+d} mm" + ("  ✓" if y in chosen else ""), loc="left")
        if y in chosen:
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(cfg["palette"]["accent"]); sp.set_linewidth(2.5)
    _save(fig, "mri_step_1_candidates.png", credit, log)
    return {"n_candidates": len(ys), "candidate_y_mm": list(ys), "slice_y_mm": chosen}


def _strip(cfg, p, c, vs, vmax, credit, log):
    """M2 and M3: the plane of the example, its labels, and the T1w image inside the brain mask."""
    names = _tissue_names(cfg)
    h, w = p["t1w"].shape
    present = [int(k) for k in np.unique(p["tissue"]) if k]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    axs[0].imshow(p["t1w"], cmap="gray", vmin=0, vmax=vmax)
    for x, t in zip((0.03, 0.97), _sides(cfg)):
        axs[0].text(x, 0.96, t, transform=axs[0].transAxes, color="w", fontsize=9, ha="center", va="top")
    _off(axs[0], f"2a  T1w template, whole head\n{w} × {h} px, {vs} mm")
    axs[1].imshow(p["tissue"], cmap=ListedColormap([TISSUE[k] for k in sorted(TISSUE)]), vmin=-0.5, vmax=max(TISSUE) + 0.5, interpolation="nearest")
    keys = [Patch(color=TISSUE[k], label=f"{k}  {names.get(k, '')}") for k in present]
    axs[1].legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, **KEY)
    _off(axs[1], "2b  Tissue labels of the template\nwhite: no label (scalp, skull, air)")
    axs[2].imshow(np.where(c["mask"], p["t1w"], 0), cmap="gray", vmin=0, vmax=vmax)
    for q in contours(c["mask"]):
        axs[2].plot(q[:, 1], q[:, 0], color=cfg["palette"]["card_edge"], lw=0.9)
    _off(axs[2], "2c  T1w inside the brain mask only\n(skull-stripped); line: outline of the mask")
    _save(fig, "mri_step_2_strip.png", credit, log)
    return {"plane_px": [int(w), int(h)], "t1w_window": [0, round(vmax, 1)], "tissue_names": {str(k): n for k, n in names.items()},
            "example_labels": present, "example_px": {"intracranial": int((p["tissue"] > 0).sum()), "mask": int(c["mask"].sum())}}


def _classes(cfg, p, c, vs, credit, log):
    """M3: the classes of the example."""
    r, pal = cfg["raster"], cfg["palette"]
    added, cut, icv = _mask_steps(p, c, r)
    shape = c["mask"].shape
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    axs[0].imshow(_paint(shape, [(c["brain"], pal["grey_matter"]), (c["wm"], pal["white_matter"])]))
    _off(axs[0], "3a  Grey and white matter from the labels\n(no threshold)")
    axs[1].imshow(_paint(shape, [(c["brain"], BASE), (added, ADDED), (cut, CUT)]))
    for q in contours(icv):
        axs[1].plot(q[:, 1], q[:, 0], color=MUTED, lw=0.7, ls=":")
    keys = [Patch(color=BASE, label="grey and white matter (3a)"), Patch(color=ADDED, label=f"added: {int(added.sum())} px"),
            Patch(color=CUT, label=f"cut by the limit: {int(cut.sum())} px"),
            Line2D([], [], color=MUTED, lw=0.7, ls=":", label="intracranial volume")]
    axs[1].legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, **KEY)
    _off(axs[1], f"3b  Brain mask: closing by {r['mask_close_px']} px, holes filled,\nlimited to the intracranial volume")
    axs[2].imshow(_paint(shape, [(c[k], pal[col]) for k, col in M.LAYERS]))
    keys = [Patch(color=pal[col], label="CSF" if col == "csf" else col.replace("_", " ")) for _, col in M.LAYERS]
    axs[2].legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, **KEY)
    _off(axs[2], "3c  Three classes in the palette colours\n(before rounding)")
    _save(fig, "mri_step_3_classes.png", credit, log)
    return {"mask_close_px": r["mask_close_px"], "closing_added_px": int(added.sum()), "closing_added_mm2": round(float(added.sum()) * vs * vs),
            "icv_removed_px": int(cut.sum())}


def _contours(cfg, c, m, vs, credit, log):
    """M4: rounding and contour simplification, zoom on the cortex of the example."""
    r, v, pal = cfg["raster"], cfg["vector"], cfg["palette"]
    rr, cc = np.where(m["mask"])
    zr, zc, zh, zw = max(int(rr.min()) + ZOOM[0], 0), int(cc.min()) + ZOOM[1], ZOOM[2], ZOOM[3]
    sl = (slice(zr, zr + zh), slice(zc, zc + zw))
    three = lambda k: _paint((zh, zw), [(k[n][sl], pal[col]) for n, col in M.LAYERS])  # noqa: E731
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    axs[0].imshow(three(c)); _off(axs[0], f"4a  Zoom of 3c: pixel edges\n{zw * vs:.0f} × {zh * vs:.0f} mm, {vs} mm per pixel")
    axs[1].imshow(three(m)); _off(axs[1], f"4b  Masks blurred σ {r['mask_round_sigma']}, cut at 0.5;\nobjects ≤ {r['min_object_px']} px dropped")
    axs[2].imshow(three(m), alpha=.45)
    for q, s in _traced(m["mask"], v):
        axs[2].plot(q[:, 1] - zc, q[:, 0] - zr, color="#999", lw=.6)
        axs[2].plot(s[:, 0] - zc, s[:, 1] - zr, color=pal["accent"], lw=1.4)
        axs[2].plot(s[:, 0] - zc, s[:, 1] - zr, "o", ms=2.2, color=pal["accent"])
    axs[2].set_xlim(-0.5, zw - 0.5); axs[2].set_ylim(zh - 0.5, -0.5)
    n = {k: {"raw": sum(len(q) for q, _ in t), "simplified": sum(len(s) for _, s in t)} for k, t in ((k, _traced(m[k], v)) for k, _ in M.LAYERS)}
    _off(axs[2], f"4c  Contour (grey) → smoothed σ {v['contour_sigma']}\n→ Douglas–Peucker ε {v['dp_epsilon']} (purple)\n"
                 f"all mask contours: {n['mask']['raw']} → {n['mask']['simplified']} points")
    _save(fig, "mri_step_4_contours.png", credit, log)
    return {"contour_points_raw": n["mask"]["raw"], "contour_points_simplified": n["mask"]["simplified"], "contour_points_layers": n}


def _frame(cfg, J, items, grid, credit, log):
    """M5: the outlines of all slices in the shared frame, the markers and the WSI block centres. The colours repeat if there are more slices than colours."""
    v, (x0, z_top, vs, dx), sx = cfg["vector"], grid, scan.x_sign(cfg)
    vw, vh, unit = J["viewBox"][2], J["viewBox"][3], J["mm_per_unit"]
    ux = lambda x: sx * (np.asarray(x) - J["frame_mm"][0]) / unit  # noqa: E731  x (mm) -> frame units
    uz = lambda z: (J["frame_mm"][1] - np.asarray(z)) / unit  # noqa: E731  z (mm) -> frame units
    fig, ax = plt.subplots(figsize=(6, 6))
    keys = []
    for it, col in zip(items, itertools.cycle(COLS)):
        s = it["J"]
        for _, q in _traced(it["m"]["mask"], v):
            ax.plot(ux(x0 + q[:, 0] * dx), uz(z_top - q[:, 1] * vs), color=col, lw=.8)
        text = f"{s['k']}  y = {s['y_mm']:+d} mm"
        if s["rect"]:
            x, y, w, h = s["rect"]
            roi = np.isin(it["p"]["structure"], it["s"]["roi_labels"])
            e = _extent(roi.shape, grid)
            ax.imshow(_tint(roi, col, .18), extent=(ux(e[0]), ux(e[1]), uz(e[2]), uz(e[3])), interpolation="nearest")
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ls="--", lw=1.4, color=col))
            ax.plot(x + w / 2, y + h / 2, "o", ms=3, color=col)
            ax.plot(x, y, "o", ms=11, color=col)
            ax.text(x, y, str(s["k"]), color="w", fontsize=7, fontweight="bold", ha="center", va="center_baseline")
            text += f": {s['roi']}, centroid ({s['roi_mm'][0]:.1f}, {s['roi_mm'][1]:.1f}) mm"
        if s["rect"] and "distance_mm" in it:          # the line ends at the marker centre of this slice
            bx, bz = it["wsi"]["block_mm"]
            ax.plot([ux(bx), x + w / 2], [uz(bz), y + h / 2], color=col, lw=.8)
            ax.plot(ux(bx), uz(bz), "x", ms=7, mew=1.6, color=col)
            text += f"\n     × WSI block centre ({bx:.0f}, {bz:.0f}) mm: {it['distance_mm']:.1f} mm from the centroid"
        keys.append(Line2D([], [], color=col, lw=1.6, label=text))
    ax.axvline(ux(0), color="#aaa", lw=.8, ls=":"); ax.text(ux(0) + 4, 14, "midline x = 0", fontsize=7, color=MUTED)
    left, right = _sides(cfg)
    ax.text(8, 14, left, fontsize=8, color=MUTED); ax.text(vw - 8, 14, right, fontsize=8, color=MUTED, ha="right")
    ax.set_xlim(0, vw); ax.set_ylim(vh, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=keys, loc="upper left", bbox_to_anchor=(-0.01, -0.01), fontsize=7.5, frameon=False)
    ax.set_title(f"5  One frame: viewBox {vw} × {vh}, 1 unit = {unit:.1f} mm\n"
                 f"marker: {v['roi_units']} units square; its centre is the centroid of the shaded labels", loc="left")
    _save(fig, "mri_step_5_frame.png", credit, log)


def _pairing(cfg, wcfg, J, items, grid, wgrid, credit, log):
    """The MRI brain mask and the WSI tissue outline of the same y in the same mm coordinates."""
    pal = cfg["palette"]
    shared = [it for it in items if it["wsi"]]
    if not shared:
        log("mri_step_8_pairing.png: no slice at the y of a WSI section, not drawn")
        return
    fx, fz, unit, sx = J["frame_mm"][0], J["frame_mm"][1], J["mm_per_unit"], scan.x_sign(cfg)
    box = (fx - sx * 8, fx + sx * (J["viewBox"][2] * unit + 8), fz - J["viewBox"][3] * unit - 8, fz + 8)     # the frame and 8 mm around it, left to right as on the slices
    fig, axs = plt.subplots(1, len(shared), figsize=(4.3 * len(shared), 4.5), sharey=True)
    for ax, it, a in zip(np.atleast_1d(axs), shared, "abcdefgh"):
        ax.imshow(_tint(it["m"]["mask"], pal["grey_matter"], .6), extent=_extent(it["m"]["mask"].shape, grid), interpolation="nearest")
        for _, q in _traced(it["wsi_t"], wcfg["vector"]):
            ax.plot(wgrid[0] + q[:, 0] * wgrid[3], wgrid[1] - q[:, 1] * wgrid[2], color=pal["accent"], lw=.9)
        ax.axvline(0, color="#aaa", lw=.8, ls=":"); ax.axhline(0, color="#aaa", lw=.8, ls=":")
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect("equal")
        ax.set_xticks(range(-60, 61, 30)); ax.set_yticks(range(-60, 91, 30)); ax.tick_params(labelsize=7, color="#aaa", labelcolor=MUTED)
        ax.set_xlabel("x (mm)", fontsize=7.5, color=MUTED)
        for sp in ax.spines.values():
            sp.set_edgecolor("#ccc")
        ax.set_title(f"8{a}  y = {it['J']['y_mm']:+d} mm: Dice {it['dice']:.2f}", loc="left")
    np.atleast_1d(axs)[0].set_ylabel("z (mm)", fontsize=7.5, color=MUTED)
    keys = [Patch(color=to_rgb(pal["grey_matter"]) + (0.6,), label="MRI brain mask (holiAtlas)"),
            Line2D([], [], color=pal["accent"], lw=.9, label="WSI tissue outline (BigBrain)")]
    fig.legend(handles=keys, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=8, frameon=False)
    _save(fig, "mri_step_8_pairing.png", credit, log)


def _wsi(it, wcfg, wdata, wmeta, wgrid, grid):
    """Add the WSI section of the same y to a slice: tissue mask, Dice with the MRI masks, distance of the block centre."""
    ws = [q for q in wcfg["sections"] if q["y_mm"] == it["s"]["y_mm"] and q["y_mm"] in wmeta["ys"]]
    if not ws:
        return
    it["wsi"] = ws[0]
    it["wsi_t"] = S.masks(S.plane(wdata, wmeta, ws[0]["y_mm"], wcfg["raster"]["downsample"]), wcfg["raster"])[0]
    on_mri = _resample(it["wsi_t"], wgrid, grid, it["m"]["mask"].shape)
    it["dice"], it["dice_tissue"] = _dice(it["m"]["mask"], on_mri), _dice(it["m"]["brain"], on_mri)
    it["wsi_tissue_mm2"] = round(float(it["wsi_t"].sum()) * wgrid[2] ** 2)
    if it["centre_mm"]:
        it["offset_mm"] = [ws[0]["block_mm"][0] - it["centre_mm"][0], ws[0]["block_mm"][1] - it["centre_mm"][1]]
        it["distance_mm"] = float(np.hypot(*it["offset_mm"]))


def _stale(cfg, what):
    """The error for an out/mri/slices.json that another configuration wrote."""
    return ValueError(f"{what}: {M.MRI_OUT / 'slices.json'} does not agree with {cfg['_path']}. Run `python -m medfig mri slices` again.")


def _items(cfg, wcfg, J, data, meta, grid):
    """One record per configured slice: planes, masks, marker centre in mm and, where WSI has a section at the same y, the WSI values.

    Raises ValueError if slices.json does not hold the configured slices: a missing slice, another slice, another y or another marker.
    """
    r = cfg["raster"]
    wdata, wmeta = S.load_planes(wcfg)
    wgrid = _wsi_grid(wdata, wmeta, wcfg["raster"]["downsample"])
    by_k = {q["k"]: q for q in J["slices"]}
    for s in cfg["slices"]:
        if s["k"] not in by_k or by_k[s["k"]]["y_mm"] != s["y_mm"]:
            raise _stale(cfg, f"slice {s['k']} (y = {s['y_mm']} mm)")
    other = sorted(set(by_k) - {s["k"] for s in cfg["slices"]})
    if other:
        raise _stale(cfg, f"slices {other} are not in the configuration")
    if J["mm_per_unit"] != cfg["vector"]["unit_mm"]:
        raise _stale(cfg, "unit_mm")
    out = []
    for s in cfg["slices"]:
        p = M.plane(data, meta, s["y_mm"], cfg)
        c = M.classes(p, cfg)
        it = {"s": s, "J": by_k[s["k"]], "p": p, "c": c, "m": M.rounded(c, r), "wsi": None, "centre_mm": None}
        if s.get("roi_labels"):
            cr, cc = M.roi_centre(p["structure"], s["roi_labels"], r)
            it["centre_mm"] = scan.to_mm(cr, cc, grid)
        want = [round(q, 1) for q in it["centre_mm"]] if it["centre_mm"] else None      # as roi_mm in slices.build()
        if it["J"]["roi_mm"] != want or (it["J"]["rect"] and it["J"]["rect"][2] != cfg["vector"]["roi_units"]):
            raise _stale(cfg, f"marker of slice {s['k']}")
        _wsi(it, wcfg, wdata, wmeta, wgrid, grid)
        out.append(it)
    return out, wgrid


def _numbers(cfg, items, vs):
    """The values of each slice for the report."""
    num = {"area_mm2": [], "fractions": [], "mask_steps": [], "roi": [], "wsi_block": [], "pairing": []}
    for it in items:
        s, js = it["s"], it["J"]
        head = {"k": s["k"], "y_mm": s["y_mm"]}
        a = {k: round(float(it["m"][k].sum()) * vs * vs) for k, _ in M.LAYERS}           # as area_mm2 in slices.json
        a = {"mask": a["mask"], "csf": a["mask"] - a["brain"], "grey_matter": a["brain"] - a["wm"], "white_matter": a["wm"]}
        num["area_mm2"].append({**head, **a})
        num["fractions"].append({**head, **{k: round(a[k] / a["mask"], 3) for k in ("csf", "grey_matter", "white_matter")}})
        added, cut, _ = _mask_steps(it["p"], it["c"], cfg["raster"])
        num["mask_steps"].append({**head, "brain_px": int(it["c"]["brain"].sum()), "closing_added_px": int(added.sum()),
                                  "icv_removed_px": int(cut.sum()), "mask_px": int(it["c"]["mask"].sum())})
        if s.get("roi_labels"):
            num["roi"].append({**head, "name": s["roi"], "labels": list(s["roi_labels"]), "roi_mm": js["roi_mm"], "rect": js["rect"],
                               "px": int(np.isin(it["p"]["structure"], s["roi_labels"]).sum())})
        if "distance_mm" in it:
            num["wsi_block"].append({**head, "block": it["wsi"]["block"], "block_mm": list(it["wsi"]["block_mm"]), "roi_mm": js["roi_mm"],
                                     "offset_mm": [round(d, 1) for d in it["offset_mm"]], "distance_mm": round(it["distance_mm"], 1)})
        if it["wsi"]:
            num["pairing"].append({**head, "dice": round(it["dice"], 3), "dice_tissue": round(it["dice_tissue"], 3),
                                   "mri_mask_mm2": a["mask"], "wsi_tissue_mm2": it["wsi_tissue_mm2"]})
    return num


def build(cfg=None, log=print):
    cfg = cfg or load("mri")
    wcfg = load("wsi")
    _style(cfg["palette"]["page"])
    FIG.mkdir(parents=True, exist_ok=True)
    data, meta = M.load_planes(cfg)
    J = json.loads((M.MRI_OUT / "slices.json").read_text(encoding="utf-8"))
    grid = _grid(meta, cfg)
    lic, wlic = cfg["source"]["license"], wcfg["source"]["license"]
    vmax = float(np.percentile(data["t1w"][data["tissue"] > 0], 99.9))     # grey window of the T1w panels: 0 to this value
    items, wgrid = _items(cfg, wcfg, J, data, meta, grid)
    by_k = {it["s"]["k"]: it for it in items}
    k = int(os.environ.get("MEDFIG_MRI_EXAMPLE", 2 if 2 in by_k else items[0]["s"]["k"]))
    if k not in by_k:
        raise ValueError(f"MEDFIG_MRI_EXAMPLE={k}: the configured slices are k = {sorted(by_k)}")
    ex = by_k[k]
    num = {"license": lic, "wsi_license": wlic, "example_k": ex["s"]["k"], "example_y_mm": ex["s"]["y_mm"],
           "planes_shape": list(data["t1w"].shape), "voxel_mm": grid[2], "x0_mm": grid[0], "z_top_mm": grid[1]}
    num.update(_candidates(cfg, data, meta, vmax, lic, log))
    num.update(_strip(cfg, ex["p"], ex["c"], grid[2], vmax, lic, log))
    num.update(_classes(cfg, ex["p"], ex["c"], grid[2], lic, log))
    num.update(_contours(cfg, ex["c"], ex["m"], grid[2], lic, log))
    _frame(cfg, J, items, grid, f"{lic}; {wlic}", log)
    _pairing(cfg, wcfg, J, items, grid, wgrid, f"{lic}; {wlic}", log)
    num.update(_numbers(cfg, items, grid[2]))
    num.update(viewBox=J["viewBox"], mm_per_unit=J["mm_per_unit"], frame_mm=J["frame_mm"], units_per_px=round(grid[2] / J["mm_per_unit"], 4),
               roi_units=cfg["vector"]["roi_units"], svg_kb={s["file"]: round((M.MRI_OUT / s["file"]).stat().st_size / 1024, 1) for s in J["slices"]})
    (FIG / "mri_numbers.json").write_text(json.dumps(num, indent=1), encoding="utf-8")
    log("mri_numbers.json")
    return num


if __name__ == "__main__":
    build()
    sys.exit(0)

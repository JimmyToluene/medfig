"""Step C8, figures: steps C1 to C5 for the report, and the pairing with MRI. Each panel calls the same functions as the pipeline.

Usage: python -m medfig.ct.figures
Input:  the CT planes .npz, the MRI planes .npz, config/ct.json, config/mri.json and out/ct/slices.json (run ct slices first).
Output: build/fig/ct_step_1_planes.png, ct_step_3_classes.png, ct_step_5_frame.png, ct_step_8_pairing.png, build/fig/ct_numbers.json (values quoted in the report)
The worked example is the slice with k = figures.example_k (MEDFIG_CT_EXAMPLE=k picks another slice).
The figures show a template. They are a schematic and do not show a patient.
"""
import itertools
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from .. import scan  # noqa: E402
from ..config import BUILD, load  # noqa: E402
from ..figures import COLS, _off, _style  # noqa: E402
from ..mri import figures as MF  # noqa: E402
from ..mri import slices as M  # noqa: E402
from ..vector import contours  # noqa: E402
from . import slices as C  # noqa: E402
from .extract import largest  # noqa: E402

FIG = BUILD / "fig"
BONE, ICV, ADDED, DROPPED = (0.95, 0.95, 0.95), (0.66, 0.65, 0.74), (0.2, 0.4, 0.9), (0.85, 0.2, 0.2)


def _stale(cfg, what):
    return ValueError(f"{what}: {C.CT_OUT / 'slices.json'} does not agree with {cfg['_path']}. Run `python -m medfig ct slices` again.")


def _items(cfg, mcfg, J, data, meta, mdata, mmeta):
    """One record per configured slice: CT plane, classes, rounded masks, holiAtlas masks on the same rows, marker centre."""
    r, g = cfg["raster"], scan.grid(meta, cfg)
    by_k = {q["k"]: q for q in J["slices"]}
    if sorted(by_k) != sorted(s["k"] for s in cfg["slices"]) or J["mm_per_unit"] != cfg["vector"]["unit_mm"]:
        raise _stale(cfg, "slices or unit")
    out = []
    for s in cfg["slices"]:
        if by_k[s["k"]]["y_mm"] != s["y_mm"]:
            raise _stale(cfg, f"slice {s['k']}")
        p = C.plane(data, meta, s["y_mm"], cfg)
        c = C.classes(p, cfg)
        rows = p["hu"].shape[0]
        mp = M.plane(mdata, mmeta, s["y_mm"], cfg)
        pad = lambda a: np.pad(a[:rows], ((0, max(0, rows - a.shape[0])), (0, 0)))  # noqa: E731  holiAtlas rows on the CT rows
        holi = {"icv": pad(ndi.binary_fill_holes(mp["tissue"] > 0)), "mask": pad(M.rounded(M.classes(mp, mcfg), mcfg["raster"])["mask"]),
                "structure": pad(mp["structure"])}
        centre = scan.slice_centre(s, holi["structure"], r)
        want = [round(q, 1) for q in scan.to_mm(*centre, g)] if centre else None
        if by_k[s["k"]]["roi_mm"] != want:
            raise _stale(cfg, f"marker of slice {s['k']}")
        out.append({"s": s, "J": by_k[s["k"]], "p": p, "c": c, "m": C.rounded(c, r), "holi": holi})
    return out


def _planes(cfg, items, cut_r, credit, log):
    """C1 and C2: the configured planes in a brain window and a bone window, with the cut of the field of view."""
    f = cfg["figures"]
    fig, axs = plt.subplots(2, len(items), figsize=(4.2 * len(items), 8.4), squeeze=False)
    for j, it in enumerate(items):
        for i, (win, name) in enumerate(((f["brain_window_hu"], "brain"), (f["bone_window_hu"], "bone"))):
            ax = axs[i, j]
            ax.imshow(it["p"]["hu"], cmap="gray", vmin=win[0], vmax=win[1])
            ax.axhline(cut_r + 0.5, color=cfg["palette"]["accent"], lw=1, ls="--")
            for x, t in zip((0.04, 0.96), MF._sides(cfg)):
                ax.text(x, 0.97, t, transform=ax.transAxes, color="w", fontsize=9, ha="center", va="top")
            _off(ax, f"1{'abcdef'[j + i * len(items)]}  y = {it['s']['y_mm']:+d} mm, {name} window {win[0]} to {win[1]} HU")
    fig.text(0.01, 0.005, f"Dashed: lowest plane of the CT volume (z = {cfg['_cut_z']:g} mm); the rows below it repeat the last row.",
             fontsize=8, color=MF.MUTED)
    MF._save(fig, "ct_step_1_planes.png", credit, log)


def _classes(cfg, it, credit, log):
    """C3: the classes of the worked example."""
    p, c, pal, r = it["p"], it["c"], cfg["palette"], cfg["raster"]
    icv = (p["icv"] > 0) & ~c["bone"]
    cran = c["bone"] | icv
    closed = ndi.binary_closing(cran, iterations=r["suture_close_px"]) | cran
    added, dropped = closed & ~cran, closed & ~c["ring"]
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.8))
    w = cfg["figures"]["brain_window_hu"]
    axs[0].imshow(p["hu"], cmap="gray", vmin=w[0], vmax=w[1])
    for q in contours(c["bone"]):
        axs[0].plot(q[:, 1], q[:, 0], color=pal["outline"], lw=.6)
    for q in contours(p["icv"] > 0):
        axs[0].plot(q[:, 1], q[:, 0], color="#f08a4b", lw=.8)
    keys = [Line2D([], [], color=pal["outline"], lw=1, label=f"bone: HU > {cfg['hu']['bone']}"),
            Line2D([], [], color="#f08a4b", lw=1, label="intracranial mask (3D, planes file)")]
    axs[0].legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, **MF.KEY)
    _off(axs[0], f"3a  HU, brain window {w[0]} to {w[1]} HU")
    axs[1].imshow(MF._paint(cran.shape, [(cran & c["bone"], BONE), (icv, ICV), (added, ADDED), (dropped, DROPPED)]))
    keys = [Patch(color=BONE, label="bone"), Patch(color=ICV, label="intracranial, without bone"),
            Patch(color=ADDED, label=f"added by the closing: {int(added.sum())} px"), Patch(color=DROPPED, label=f"not in the ring: {int(dropped.sum())} px")]
    for k in keys[:1]:
        k.set_edgecolor("#999")
    axs[1].legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, **MF.KEY)
    _off(axs[1], f"3b  Cranium: bone or intracranial,\nclosed by {r['suture_close_px']} px; ring: its largest part")
    axs[2].imshow(MF._paint(cran.shape, [(np.ones(cran.shape, bool), pal["screen"])] + [(c[k], pal[col]) for k, col in C.LAYERS]))
    keys = [Patch(facecolor=pal[col], edgecolor="#999", label={"ring": "bone (ring)", "brain": "intracranial space", "csf": f"CSF: HU < {cfg['hu']['csf']}"}[k])
            for k, col in C.LAYERS]
    axs[2].legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, **MF.KEY)
    _off(axs[2], "3c  Three classes in the palette colours,\non the screen colour (before rounding)")
    MF._save(fig, "ct_step_3_classes.png", credit, log)
    return {"closing_added_px": int(added.sum()), "ring_dropped_px": int(dropped.sum()), "bone_px": int(c["bone"].sum()),
            "icv_px": int(icv.sum()), "csf_px": int(c["csf"].sum())}


def _frame(cfg, J, items, g, credit, log):
    """C5: the outlines of the rings in the shared frame, the markers and the cut."""
    v, sx, unit = cfg["vector"], scan.x_sign(cfg), J["mm_per_unit"]
    vw, vh = J["viewBox"][2], J["viewBox"][3]
    ux = lambda x: sx * (np.asarray(x) - J["frame_mm"][0]) / unit  # noqa: E731
    uz = lambda z: (J["frame_mm"][1] - np.asarray(z)) / unit  # noqa: E731
    fig, ax = plt.subplots(figsize=(6, 6.4))
    keys = []
    for it, col in zip(items, itertools.cycle(COLS)):
        s = it["J"]
        for _, q in MF._traced(it["m"]["ring"], v):
            ax.plot(ux(g[0] + q[:, 0] * g[3]), uz(g[1] - q[:, 1] * g[2]), color=col, lw=.8)
        text = f"{s['k']}  y = {s['y_mm']:+d} mm"
        if s["rect"]:
            x, y, w, h = s["rect"]
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ls="--", lw=1.4, color=col))
            ax.plot(x + w / 2, y + h / 2, "o", ms=3, color=col)
            text += f": {s['roi']}, centroid ({s['roi_mm'][0]:.1f}, {s['roi_mm'][1]:.1f}) mm"
        keys.append(Line2D([], [], color=col, lw=1.6, label=text))
    ax.axvline(ux(0), color="#aaa", lw=.8, ls=":")
    left, right = MF._sides(cfg)
    ax.text(8, 14, left, fontsize=8, color=MF.MUTED); ax.text(vw - 8, 14, right, fontsize=8, color=MF.MUTED, ha="right")
    ax.axhline(vh, color=cfg["palette"]["accent"], lw=1, ls="--")
    ax.set_xlim(0, vw); ax.set_ylim(vh + 30, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=keys, loc="upper left", bbox_to_anchor=(-0.01, -0.01), fontsize=7.5, frameon=False)
    ax.set_title(f"5  One frame: viewBox {vw} × {vh}, 1 unit = {unit:.1f} mm; dashed line: the bottom edge\n"
                 f"(z = {J['cut_z_mm']:g} mm, the lowest CT plane); markers: centroids of holiAtlas labels", loc="left")
    MF._save(fig, "ct_step_5_frame.png", credit, log)


def _pairing(cfg, items, g, credit, log):
    """The CT intracranial space (brain class) against the holiAtlas intracranial mask and the MRI brain mask, in mm."""
    pal = cfg["palette"]
    fig, axs = plt.subplots(1, len(items), figsize=(4.3 * len(items), 4.6), sharey=True, squeeze=False)
    out = []
    for ax, it, a in zip(axs[0], items, "abcdefgh"):
        b = it["m"]["brain"]
        d_icv, d_mask = MF._dice(b, it["holi"]["icv"]), MF._dice(b, it["holi"]["mask"])
        out.append({"k": it["s"]["k"], "y_mm": it["s"]["y_mm"], "dice_icv": round(d_icv, 3), "dice_mri_mask": round(d_mask, 3),
                    "ct_brain_mm2": round(float(b.sum()) * g[2] ** 2), "holi_icv_mm2": round(float(it["holi"]["icv"].sum()) * g[2] ** 2)})
        ax.imshow(MF._tint(b, pal["brain"], .7), extent=MF._extent(b.shape, g), interpolation="nearest")
        for q in contours(it["holi"]["icv"]):
            ax.plot(g[0] + q[:, 1] * g[3], g[1] - q[:, 0] * g[2], color=pal["accent"], lw=.9)
        for q in contours(it["holi"]["mask"]):
            ax.plot(g[0] + q[:, 1] * g[3], g[1] - q[:, 0] * g[2], color="#f08a4b", lw=.7, ls="--")
        ax.axvline(0, color="#aaa", lw=.8, ls=":"); ax.axhline(0, color="#aaa", lw=.8, ls=":")
        sx = scan.x_sign(cfg)
        ax.set_xlim(-sx * 85, sx * 85); ax.set_ylim(-75, 95); ax.set_aspect("equal")
        ax.tick_params(labelsize=7, color="#aaa", labelcolor=MF.MUTED)
        ax.set_xlabel("x (mm)", fontsize=7.5, color=MF.MUTED)
        ax.set_title(f"8{a}  y = {it['s']['y_mm']:+d} mm: Dice {d_icv:.2f} (ICV), {d_mask:.2f} (MRI mask)", loc="left", fontsize=9)
    axs[0][0].set_ylabel("z (mm)", fontsize=7.5, color=MF.MUTED)
    keys = [Patch(color=pal["brain"], alpha=.7, label="CT intracranial space (class 'brain', rounded)"),
            Line2D([], [], color=pal["accent"], lw=.9, label="holiAtlas intracranial volume (label > 0, filled)"),
            Line2D([], [], color="#f08a4b", lw=.7, ls="--", label="MRI brain mask (medfig MRI)")]
    fig.legend(handles=keys, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8, frameon=False)
    MF._save(fig, "ct_step_8_pairing.png", credit, log)
    return out


def build(cfg=None, log=print):
    cfg = cfg or load("ct")
    mcfg = load("mri")
    _style(cfg["palette"]["page"])
    FIG.mkdir(parents=True, exist_ok=True)
    data, meta = C.load_planes(cfg)
    mdata, mmeta = M.load_planes(mcfg)
    C.check(cfg, meta, mmeta)
    J = json.loads((C.CT_OUT / "slices.json").read_text(encoding="utf-8"))
    g = scan.grid(meta, cfg)
    cfg = {**cfg, "_cut_z": meta["ct"]["z_min_mm"]}
    items = _items(cfg, mcfg, J, data, meta, mdata, mmeta)
    by_k = {it["s"]["k"]: it for it in items}
    k = int(os.environ.get("MEDFIG_CT_EXAMPLE", cfg["figures"]["example_k"] if cfg["figures"]["example_k"] in by_k else items[0]["s"]["k"]))
    if k not in by_k:
        raise ValueError(f"MEDFIG_CT_EXAMPLE={k}: the configured slices are k = {sorted(by_k)}")
    lic = f"{cfg['source']['license']}; markers and MRI masks: {cfg['source']['marker_credit']}"
    num = {"license": cfg["source"]["license"], "example_k": k, "example_y_mm": by_k[k]["s"]["y_mm"], "planes_shape": list(data["hu"].shape),
           "voxel_mm": g[2], "ct": meta["ct"], "icv_ml": meta["icv_ml"], "hu": meta["hu"], "cut_row": C.cut_row(meta)}
    _planes(cfg, items, C.cut_row(meta), lic, log)
    num.update(_classes(cfg, by_k[k], lic, log))
    num["pairing"] = _pairing(cfg, items, g, lic, log)
    _frame(cfg, J, items, g, lic, log)
    vs = g[2]
    num["area_mm2"] = [{"k": it["s"]["k"], "y_mm": it["s"]["y_mm"], **{n: round(float(it["m"][n].sum()) * vs * vs) for n, _ in C.LAYERS}} for it in items]
    num["bone_mm2"] = [{"k": a["k"], "y_mm": a["y_mm"], "bone": a["ring"] - a["brain"]} for a in num["area_mm2"]]
    num.update(viewBox=J["viewBox"], mm_per_unit=J["mm_per_unit"], frame_mm=J["frame_mm"], cut_z_mm=J["cut_z_mm"],
               roi=[{"k": s["k"], "roi": s["roi"], "roi_mm": s["roi_mm"]} for s in J["slices"]],
               svg_kb={s["file"]: round((C.CT_OUT / s["file"]).stat().st_size / 1024, 1) for s in J["slices"]})
    (FIG / "ct_numbers.json").write_text(json.dumps(num, indent=1), encoding="utf-8")
    log("ct_numbers.json")
    return num


if __name__ == "__main__":
    build()
    sys.exit(0)

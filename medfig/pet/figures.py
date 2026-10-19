"""Step P8, figures: steps P1 to P3 for the report. Each panel calls the same functions as the pipeline.

Usage: python -m medfig.pet.figures
Input:  the PET planes .npz, the MRI planes .npz, config/pet.json, config/mri.json and out/pet/<series>/slices.json
        (run pet slices first).
Output: build/fig/pet_step_1_suvr.png, pet_step_3_bands.png, build/fig/pet_numbers.json (values quoted in the report)
The worked example is the slice with k = figures.example_k (MEDFIG_PET_EXAMPLE=k picks another slice).
The figures show group templates. They are a schematic and do not show a patient.
"""
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
from ..figures import _off, _style  # noqa: E402
from ..mri import figures as MF  # noqa: E402
from ..mri import slices as M  # noqa: E402
from ..vector import contours  # noqa: E402
from . import slices as P  # noqa: E402

FIG = BUILD / "fig"


def _stale(cfg, key, what):
    return ValueError(f"{what}: {P.PET_OUT / key / 'slices.json'} does not agree with {cfg['_path']}. Run `python -m medfig pet slices` again.")


def _items(cfg, mcfg, data, meta, mdata, mmeta):
    """One record per configured slice: the MRI plane and masks, and for each series the SUVR plane and the bands."""
    vs = mmeta["vs"]
    js = {se["key"]: scan.read_slices(P.PET_OUT / se["key"]) for se in cfg["series"]}
    for key, J in js.items():
        if [(s["k"], s["y_mm"]) for s in J["slices"]] != [(s["k"], s["y_mm"]) for s in cfg["slices"]] or J["bands"] != cfg["bands"]["edges"]:
            raise _stale(cfg, key, "slices or bands")
    out = []
    for s in cfg["slices"]:
        mp = M.plane(mdata, mmeta, s["y_mm"], cfg)
        c = M.classes(mp, mcfg)
        mask = P.brain(mp, mcfg)
        it = {"s": s, "mp": mp, "c": c, "mask": mask, "suvr": {}, "bands": {}}
        for se in cfg["series"]:
            sv = P.suvr(data, meta, se["channel"], s["y_mm"], mmeta, cfg)
            it["suvr"][se["key"]], it["bands"][se["key"]] = sv, P.bands(sv, mask, cfg, vs)
        out.append(it)
    return out, js


def _suvr(cfg, items, credit, log):
    """P1 and P2: the SUVR planes of every series on the MRI grid, with the brain outline and the reference region."""
    f, ref = cfg["figures"], cfg["reference"]
    n, m = len(cfg["series"]), len(items)
    fig, axs = plt.subplots(n, m, figsize=(4.1 * m, 4.3 * n), squeeze=False)
    for i, se in enumerate(cfg["series"]):
        for j, it in enumerate(items):
            ax = axs[i, j]
            im = ax.imshow(it["suvr"][se["key"]], cmap="gray", vmin=0, vmax=f["suvr_max"])
            for q in contours(it["mask"]):
                ax.plot(q[:, 1], q[:, 0], color="#f7d35c", lw=.8)
            refm = np.isin(it["mp"]["tissue"], ref["tissue_labels"])
            for q in contours(refm):
                ax.plot(q[:, 1], q[:, 0], color="#4fb3d9", lw=.8, ls="--")
            for x, t in zip((0.04, 0.96), MF._sides(cfg)):
                ax.text(x, 0.97, t, transform=ax.transAxes, color="w", fontsize=9, ha="center", va="top")
            _off(ax, f"1{'abcdefgh'[i * m + j]}  {se['name']}, y = {it['s']['y_mm']:+d} mm")
    cb = fig.colorbar(im, ax=axs, shrink=.6, pad=.02)
    cb.set_label(cfg["bands"]["unit"], fontsize=8)
    keys = [Line2D([], [], color="#f7d35c", lw=1, label="MRI brain mask (the outline of the figure)"),
            Line2D([], [], color="#4fb3d9", lw=1, ls="--", label=f"reference region ({ref['name']}, holiAtlas labels {', '.join(map(str, ref['tissue_labels']))})")]
    fig.legend(handles=keys, loc="lower left", bbox_to_anchor=(0.02, -0.02), ncol=2, fontsize=8, frameon=False)
    MF._save(fig, "pet_step_1_suvr.png", credit, log)


def _bands(cfg, it, vs, credit, log):
    """P3 on the worked example: the smoothed SUVR of each series with the band edges, the bands, and the histograms."""
    b, f, pal = cfg["bands"], cfg["figures"], cfg["palette"]
    n = len(cfg["series"])
    fig, axs = plt.subplots(1, n + 1, figsize=(4.8 * (n + 1), 4.8))
    hist = {}
    edges = np.linspace(0, f["suvr_max"], f["hist_bins"] + 1)
    for ax, se in zip(axs, cfg["series"]):
        ls = it["bands"][se["key"]]
        ax.imshow(MF._paint(it["mask"].shape, list(zip(ls, pal["bands"]))))
        for q in contours(it["mask"]):
            ax.plot(q[:, 1], q[:, 0], color=pal["outline"], lw=1)
        _off(ax, f"3{'abcd'[cfg['series'].index(se)]}  {se['name']}: bands\n(smoothed σ {b['smooth_sigma_mm']} mm, before tracing)")
        v = it["suvr"][se["key"]][it["mask"]]
        hist[se["key"]] = np.histogram(v, edges)[0]
    ax = axs[-1]
    for se, col in zip(cfg["series"], ("#c0458a", "#5b43a8")):
        ax.stairs(hist[se["key"]] * vs * vs, edges, color=col, lw=1.4, label=se["name"])
    for e in b["edges"]:
        ax.axvline(e, color="#999", lw=.8, ls=":")
    ax.set_xlabel(f"{b['unit']} (unsmoothed), inside the brain mask", fontsize=8)
    ax.set_ylabel("area (mm²) per bin", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"3{'abcd'[n]}  Histogram, y = {it['s']['y_mm']:+d} mm;\ndotted: band edges", loc="left")
    keys = [Patch(color=c, label=(f"< {b['edges'][0]:.1f}" if i == 0 else f"≥ {b['edges'][i - 1]:.1f}")) for i, c in enumerate(pal["bands"])]
    fig.legend(handles=keys, loc="lower left", bbox_to_anchor=(0.02, -0.04), ncol=len(keys), fontsize=8, frameon=False, title=b["unit"], title_fontsize=8)
    MF._save(fig, "pet_step_3_bands.png", credit, log)


def _numbers(cfg, mcfg, items, vs):
    """Per slice and series: the share of the brain mask in each band, and the mean SUVR of cortex and white matter."""
    tl = mcfg["tissue_labels"]
    rows, tissue = [], []
    for it in items:
        cortex = np.isin(it["mp"]["tissue"], tl["grey_matter"]) & it["mask"]
        white = np.isin(it["mp"]["tissue"], tl["white_matter"]) & it["mask"]
        for se in cfg["series"]:
            ls = it["bands"][se["key"]]
            a = [float(m.sum()) for m in ls] + [0.0]
            sv = it["suvr"][se["key"]]
            rows.append({"k": it["s"]["k"], "y_mm": it["s"]["y_mm"], "series": se["key"], "mask_mm2": round(a[0] * vs * vs),
                         "share": [round((a[j] - a[j + 1]) / a[0], 3) for j in range(len(ls))]})
            tissue.append({"k": it["s"]["k"], "y_mm": it["s"]["y_mm"], "series": se["key"],
                           "grey_matter": round(float(sv[cortex].mean()), 2), "white_matter": round(float(sv[white].mean()), 2)})
    return {"band_share": rows, "tissue_suvr": tissue}


def build(cfg=None, log=print):
    cfg = cfg or load("pet")
    mcfg = load("mri")
    _style(cfg["palette"]["page"])
    FIG.mkdir(parents=True, exist_ok=True)
    data, meta = P.load_planes(cfg)
    mdata, mmeta = M.load_planes(mcfg)
    P.check(cfg, meta)
    items, js = _items(cfg, mcfg, data, meta, mdata, mmeta)
    by_k = {it["s"]["k"]: it for it in items}
    k = int(os.environ.get("MEDFIG_PET_EXAMPLE", cfg["figures"]["example_k"] if cfg["figures"]["example_k"] in by_k else items[0]["s"]["k"]))
    if k not in by_k:
        raise ValueError(f"MEDFIG_PET_EXAMPLE={k}: the configured slices are k = {sorted(by_k)}")
    credit = f"{cfg['source']['license']}; outline and reference region: {cfg['source']['outline_credit']}"
    _suvr(cfg, items, credit, log)
    _bands(cfg, by_k[k], mmeta["vs"], credit, log)
    first = js[cfg["series"][0]["key"]]
    num = {"license": cfg["source"]["license"], "example_k": k, "example_y_mm": by_k[k]["s"]["y_mm"],
           "planes_shape": list(next(iter(data.values())).shape), "native_vs": meta["vs"], "native_dims": meta["dims"],
           "reference": meta["reference"], "series": cfg["series"], "edges": cfg["bands"]["edges"],
           "viewBox": first["viewBox"], "mm_per_unit": first["mm_per_unit"], "frame_mm": first["frame_mm"],
           "area_mm2": {key: [{"k": s["k"], "y_mm": s["y_mm"], **s["area_mm2"]} for s in J["slices"]] for key, J in js.items()},
           "svg_kb": {key: {s["file"]: round((P.PET_OUT / key / s["file"]).stat().st_size / 1024, 1) for s in J["slices"]} for key, J in js.items()}}
    num.update(_numbers(cfg, mcfg, items, mmeta["vs"]))
    (FIG / "pet_numbers.json").write_text(json.dumps(num, indent=1), encoding="utf-8")
    log("pet_numbers.json")
    return num


if __name__ == "__main__":
    build()
    sys.exit(0)

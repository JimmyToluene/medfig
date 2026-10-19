"""Steps S2 to S7: coronal planes -> flat SVG sections in one frame, with block markers.

Usage: python -m medfig.sections
Input:  the planes .npz (config source.planes_npz) and config/sections.json.
Output: out/section_{k}.svg   grey matter, white matter, outline
        out/sections.json     shared viewBox, mm per unit, block rectangle of each section, thresholds

S2  plane(): flip to rows superior -> inferior, block mean D x D (100 um -> 300 um for D = 3).
S3  tissue mask: blur, intensity < limit, drop specks, fill small holes, one opening (ventricles stay open).
S4  grey/white: local mean inside tissue, Otsu of values above the floor (leaves out the dark cerebellar
    granular layer), minus an offset. White matter = tissue brighter than the threshold (Merker: white matter is pale).
S5  round the masks (blur, cut at 0.5), trace contours (marching squares), smooth them (Gaussian, wrap),
    simplify them (Douglas-Peucker). Tracing, smoothing and path data are in vector.py (shared with MRI).
S6  one frame for all sections: same scale, union bounding box plus padding; block centres mm -> units.
S7  write SVG: grey matter fill (even-odd, so holes stay open), white matter fill, outline stroke.
"""
import json
import sys

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import threshold_otsu

from .config import OUT, load, planes_path
from .vector import contours, path_d, smooth_contour  # noqa: F401  (figures.py and the tests use them from here)


def load_planes(cfg):
    z = np.load(planes_path(cfg))
    return z["data"], json.loads(str(z["meta"]))


def plane(data, meta, y, ds):
    """S2. Coronal plane at y mm: rows superior -> inferior, columns x ascending, block mean ds x ds."""
    a = data[meta["ys"].index(y)][::-1].astype(np.float32)
    h, w = a.shape[0] // ds * ds, a.shape[1] // ds * ds
    return a[:h, :w].reshape(h // ds, ds, w // ds, ds).mean(axis=(1, 3))


def tissue_mask(a, r):
    """S3."""
    t = ndi.gaussian_filter(a, r["tissue_blur_sigma"]) < r["tissue_max_intensity"]
    t = morphology.remove_small_objects(t, max_size=r["tissue_min_object_px"])
    t = morphology.remove_small_holes(t, max_size=r["tissue_max_hole_px"])
    return ndi.binary_opening(t, iterations=1)


def local_mean(a, tissue, sigma):
    """Mean intensity of the tissue around each pixel. Background does not leak in at the edges."""
    return ndi.gaussian_filter(np.where(tissue, a, 0), sigma) / np.maximum(ndi.gaussian_filter(tissue.astype(float), sigma), 1e-3)


def grey_white(a, tissue, r):
    """S4. Returns (white matter mask before rounding, local mean, Otsu value, threshold)."""
    s = local_mean(a, tissue, r["local_mean_sigma"])
    otsu = float(threshold_otsu(s[tissue & (s > r["otsu_floor"])]))
    thr = otsu - r["wm_offset"]
    wm = tissue & (s > thr)
    wm = morphology.remove_small_objects(wm, max_size=r["wm_min_object_px"])
    wm = morphology.remove_small_holes(wm, max_size=r["wm_max_hole_px"])
    return wm, s, otsu, thr


def round_masks(tissue, wm, r):
    """S5, first half. Blur each mask and cut at one half. Small white-matter islands are dropped."""
    t = ndi.gaussian_filter(tissue.astype(float), r["mask_round_sigma"]) > 0.5
    w = t & (ndi.gaussian_filter(wm.astype(float), r["mask_round_sigma"]) > 0.5)
    w = morphology.remove_small_objects(w, max_size=r["wm_min_island_px"])
    return t, w


def masks(a, r):
    """S3 to S5 (rasters). Returns (tissue, white matter, info)."""
    tissue = tissue_mask(a, r)
    wm, _, otsu, thr = grey_white(a, tissue, r)
    t, w = round_masks(tissue, wm, r)
    return t, w, {"otsu": round(otsu, 2), "threshold": round(thr, 2)}


def build(cfg=None, log=print):
    cfg = cfg or load()
    r, v, pal = cfg["raster"], cfg["vector"], cfg["palette"]
    data, meta = load_planes(cfg)
    ds = r["downsample"]
    unit_mm = meta.get("vs", 0.1) * ds
    nz = data.shape[1]
    x0 = meta["srow"][0][3]                                   # x (mm) of column 0
    z_top = meta["srow"][2][3] + meta.get("vs", 0.1) * (nz - 1)  # z (mm) of row 0 after the flip
    secs = cfg["sections"]
    planes = [plane(data, meta, s["y_mm"], ds) for s in secs]
    ms = [masks(a, r) for a in planes]
    union = np.any([t for t, _, _ in ms], axis=0)
    rr, cc = np.where(union)
    pad = v["pad_units"]
    oy, ox = int(rr.min()) - pad, int(cc.min()) - pad
    vw, vh = int(cc.max() - cc.min()) + 2 * pad, int(rr.max() - rr.min()) + 2 * pad
    OUT.mkdir(parents=True, exist_ok=True)
    lic = cfg["source"]["license"]
    out = {"viewBox": [0, 0, vw, vh], "mm_per_unit": round(unit_mm, 4), "origin_px": [ox, oy],
           "x0_mm": x0, "z_top_mm": z_top, "license": lic, "config": cfg["_path"], "sections": []}
    B = v["block_units"]
    for s, (t, w, info) in zip(secs, ms):
        bx, bz = s["block_mm"]
        cx = (bx - x0) / unit_mm - ox
        cz = (z_top - bz) / unit_mm - oy
        dt = path_d(t, ox, oy, v)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw} {vh}" role="img" '
               f'aria-label="Coronal brain section, y = {s["y_mm"]} mm, {s["label"]}"><!-- Drawn from {lic} -->'
               f'<path d="{dt}" fill="{pal["grey_matter"]}" fill-rule="evenodd"/>'
               f'<path d="{path_d(w, ox, oy, v)}" fill="{pal["white_matter"]}" fill-rule="evenodd"/>'
               f'<path d="{dt}" fill="none" stroke="{pal["outline"]}" stroke-width="{v["outline_width"]}" stroke-linejoin="round"/></svg>')
        f = OUT / f"section_{s['k']}.svg"
        f.write_text(svg)
        out["sections"].append({**{k: s[k] for k in ("k", "y_mm", "label", "block", "block_mm")},
                                "rect": [round(cx - B / 2, 1), round(cz - B / 2, 1), B, B], **info,
                                "file": f.name})
        log(f"{f.name}: y {s['y_mm']:+d} mm, threshold {info['threshold']}, {len(svg) // 1024} KB")
    (OUT / "sections.json").write_text(json.dumps(out, indent=1))
    log(f"sections.json: viewBox {vw} x {vh}, {unit_mm:.2f} mm per unit")
    return out


if __name__ == "__main__":
    build()
    sys.exit(0)

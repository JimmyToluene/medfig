"""Steps C2 to C6: CT planes -> flat SVG slices of the head (bone ring and brain, no scalp) in one frame, with markers.

Usage: python -m medfig ct slices
Input:  the CT planes .npz (config source.planes_npz), the MRI planes .npz (structure labels for the markers) and config/ct.json.
Output: out/ct/slice_{k}.svg   screen, bone ring, intracranial space (brain), CSF, outline
        out/ct/slices.json     shared viewBox, mm per unit, frame position in mm, orientation, marker rectangle of each slice

C2  plane(): rows superior -> inferior, columns in the configured orientation (scan.py). Rows below the lowest plane of
    the CT volume (z_min_mm) are outside its field of view: the plane is cut there and its last row is repeated
    cut_pad_px times, so every mask runs out of the bottom of the frame and no outline is drawn along the cut.
C3  classes from Hounsfield units: bone = HU > hu.bone; intracranial = the 3D mask of the planes file without bone;
    cranium = bone or intracranial, closed by suture_close_px (the sutures dip under the bone threshold);
    ring = the largest part of the cranium (the skull with its content; the facial bones and the spine drop out);
    brain = intracranial inside the ring; CSF = brain with HU < hu.csf (ventricles and cisterns). The scalp is not drawn.
C4  round the masks (blur, cut at 0.5, each inside the layer below), trace, smooth and simplify (vector.py).
C5  one frame for all slices: 1 unit = vector.unit_mm, union bounding box of the rings plus padding; the bottom edge is
    the cut if the rings reach it. Markers: centroid of the roi_labels of holiAtlas, as in MRI (the grids are the same).
C6  write SVG: dark screen, bone, brain, CSF, outline of the ring.

build() checks every y, the grids and the thresholds of the planes file before it writes the first file.
"""
import json
import sys

import numpy as np
from scipy import ndimage as ndi

from .. import scan
from ..config import OUT, load, planes_path
from ..mri import slices as M
from ..vector import path_d, round_mask
from .extract import MASK_KEYS, largest

CT_OUT = OUT / "ct"
CHANNELS = ("hu", "icv")
LAYERS = (("ring", "bone"), ("brain", "brain"), ("csf", "csf"))       # mask name, palette key; back to front


def load_planes(cfg):
    z = np.load(planes_path(cfg))
    return {k: z[k] for k in CHANNELS}, json.loads(str(z["meta"]))


def cut_row(meta):
    """The row (after the flip) of the lowest plane of the CT volume: rows below it are outside the field of view."""
    return int(round((meta["origin_mm"][2] + meta["vs"] * (meta["dims"][2] - 1) - meta["ct"]["z_min_mm"]) / meta["vs"]))


def plane(data, meta, y, cfg):
    """C2. HU and intracranial mask of the plane at y: rows superior -> inferior, cut at the field of view, padded."""
    i, r, n = meta["ys"].index(y), cut_row(meta), cfg["raster"]["cut_pad_px"]
    return {k: np.pad(scan.orient(a[i], cfg)[:r + 1], ((0, n), (0, 0)), mode="edge") for k, a in data.items()}


def structure(mdata, mmeta, y, cfg, rows):
    """The holiAtlas structure labels of the plane at y on the CT rows (cut and padded with 0)."""
    s = M.plane(mdata, mmeta, y, cfg)["structure"]
    out = np.zeros((rows, s.shape[1]), s.dtype)
    out[:min(rows, s.shape[0])] = s[:rows]
    return out


def classes(p, cfg):
    """C3. Returns the masks ring (skull and content), brain (intracranial space inside the ring) and csf, and bone."""
    h, r = cfg["hu"], cfg["raster"]
    bone = p["hu"] > h["bone"]
    icv = (p["icv"] > 0) & ~bone
    cran = bone | icv
    cran = ndi.binary_closing(cran, iterations=r["suture_close_px"]) | cran
    ring = largest(cran)
    brain = icv & ring
    return {"ring": ring, "brain": brain, "csf": brain & (p["hu"] < h["csf"]), "bone": bone}


def rounded(masks, r):
    """C4, first half. Every layer lies inside the layer below it."""
    out, below = {}, None
    for k, _ in LAYERS:
        m = round_mask(masks[k], r["mask_round_sigma"], r["min_object_px"])
        out[k] = below = m if below is None else m & below
    return out


def check(cfg, meta, mmeta):
    """The planes file must be on the MRI grid and made with the thresholds of the configuration."""
    if [meta[k] for k in ("origin_mm", "dims", "vs")] != [mmeta[k] for k in ("origin_mm", "dims", "vs")]:
        raise ValueError(f"{planes_path(cfg)}: the grid is not the grid of the MRI planes file. Run `python -m medfig ct extract CT_DIR` again.")
    want = {k: cfg["hu"][k] for k in MASK_KEYS}
    if meta["hu"] != want:
        raise ValueError(f"{planes_path(cfg)} was made with hu {meta['hu']}, {cfg['_path']} has {want}. "
                         "Run `python -m medfig ct extract CT_DIR` again.")


def build(cfg=None, log=print):
    cfg = cfg or load("ct")
    mcfg = load("mri")
    r, v, pal, sc = cfg["raster"], cfg["vector"], cfg["palette"], cfg["screen"]
    data, meta = load_planes(cfg)
    mdata, mmeta = M.load_planes(mcfg)
    check(cfg, meta, mmeta)
    secs = cfg["slices"]
    M.check_ys(cfg, meta, planes_path(cfg))
    M.check_ys(cfg, mmeta, planes_path(mcfg))
    g = scan.grid(meta, cfg)
    vs = g[2]
    scale = vs / v["unit_mm"]
    planes = [plane(data, meta, s["y_mm"], cfg) for s in secs]
    ms = [rounded(classes(p, cfg), r) for p in planes]
    rows = planes[0]["hu"].shape[0]
    centres = [scan.slice_centre(s, structure(mdata, mmeta, s["y_mm"], cfg, rows), r) for s in secs]
    ox, oy, vw, vh = scan.frame([m["ring"] for m in ms], v, scale)
    vh = min(vh, int(round((cut_row(meta) + 0.5 - oy) * scale)))      # the bottom edge is the cut if the rings reach it
    CT_OUT.mkdir(parents=True, exist_ok=True)
    lic = cfg["source"]["license"]
    out = {"viewBox": [0, 0, vw, vh], "mm_per_unit": v["unit_mm"], "frame_mm": scan.frame_mm(ox, oy, g), "orientation": cfg["orientation"],
           "cut_z_mm": meta["ct"]["z_min_mm"], "license": lic, "marker_credit": cfg["source"]["marker_credit"], "config": cfg["_path"],
           "slices": []}
    screen = (f'<rect x="{sc["inset_units"]}" y="{sc["inset_units"]}" width="{vw - 2 * sc["inset_units"]}" height="{vh - sc["inset_units"]}" '
              f'rx="{sc["radius_units"]}" fill="{pal["screen"]}"/>')
    for s, m, centre in zip(secs, ms, centres):
        dm = path_d(m["ring"], ox, oy, v, scale)
        body = "".join(scan.fill(dm if k == "ring" else path_d(m[k], ox, oy, v, scale), pal[c]) for k, c in LAYERS)
        svg = scan.svg(vw, vh, f'Coronal CT slice of the head with the skull, y = {s["y_mm"]} mm, {s["label"]}', lic,
                       screen + body + scan.stroke(dm, pal["outline"], v["outline_width"]))
        f = CT_OUT / f"slice_{s['k']}.svg"
        f.write_text(svg, encoding="utf-8")
        it = scan.item(s, centre, ox, oy, g, scale, v["roi_units"], f.name, {k: round(float(m[k].sum()) * vs * vs) for k, _ in LAYERS})
        out["slices"].append(it)
        log(f"{f.name}: y {s['y_mm']:+d} mm, {len(svg) // 1024} KB" + (f", {s['roi']} at {it['roi_mm']} mm" if it["roi_mm"] else ""))
    (CT_OUT / "slices.json").write_text(json.dumps(out, indent=1))
    log(f"slices.json: viewBox {vw} x {vh}, {v['unit_mm']:.2f} mm per unit, {cfg['orientation']}, cut at z = {meta['ct']['z_min_mm']} mm")
    return out


if __name__ == "__main__":
    build()
    sys.exit(0)

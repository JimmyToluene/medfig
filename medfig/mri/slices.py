"""Steps M2 to M6: coronal planes -> flat SVG slices of the skull-stripped brain in one frame, with markers.

Usage: python -m medfig mri slices
Input:  the planes .npz (config source.planes_npz) and config/mri.json.
Output: out/mri/slice_{k}.svg   brain mask (CSF), grey matter, white matter, outline
        out/mri/slices.json     shared viewBox, mm per unit, frame position in mm, orientation, marker rectangle of each slice

M2  plane(): rows superior -> inferior; columns in the configured orientation (scan.py). Radiological: the left of the
    patient is on the right of the image.
M3  skull stripping and classes from the labels of the template, not from thresholds:
    grey and white matter = the label lists in config tissue_labels;
    brain mask = grey and white matter, closed (so sulci and cisterns are inside), holes filled, limited to the
    filled intracranial label mask of the template (label > 0, enclosed pockets filled).
    Scalp and skull are outside the mask and are not drawn.
M4  round the masks (blur, cut at 0.5), trace, smooth and simplify the contours (vector.py, shared with WSI).
M5  one frame for all slices: 1 unit = vector.unit_mm (the same as WSI), union bounding box of the masks plus padding.
    Marker centre = centroid of the roi_labels of the slice in structure_atlas. A slice without roi_labels has no marker.
M6  write SVG: mask fill (CSF: sulci and ventricles), grey matter fill, white matter fill, outline stroke.

build() checks every y and computes every marker before it writes the first file.
A y that is not in the planes file, or roi_labels with too few pixels, stops the step and leaves out/mri/ as it was.
CT and PET (medfig/ct, medfig/pet) read the same planes file through load_planes() and plane(): the structure labels
for their markers, and PET the brain mask of classes().
"""
import json
import sys

import numpy as np
from scipy import ndimage as ndi

from .. import scan
from ..config import OUT, load, planes_path
from ..scan import roi_centre  # noqa: F401  (the marker centre of a slice; CT and PET use it too)
from ..vector import path_d, round_mask

MRI_OUT = OUT / "mri"
CHANNELS = ("t1w", "tissue", "structure")
LAYERS = (("mask", "csf"), ("brain", "grey_matter"), ("wm", "white_matter"))   # mask name, palette key; back to front


def load_planes(cfg):
    z = np.load(planes_path(cfg))
    return {k: z[k] for k in CHANNELS}, json.loads(str(z["meta"]))


def plane(data, meta, y, cfg):
    """M2. Channels of the coronal plane at y mm: rows superior -> inferior, columns in the orientation of cfg."""
    i = meta["ys"].index(y)
    return {k: scan.orient(a[i], cfg) for k, a in data.items()}


def classes(p, cfg):
    """M3. Returns the masks mask (skull-stripped brain with its CSF), brain (grey and white matter) and wm."""
    r, tl = cfg["raster"], cfg["tissue_labels"]
    tis = p["tissue"]
    wm = np.isin(tis, tl["white_matter"])
    brain = wm | np.isin(tis, tl["grey_matter"])
    icv = ndi.binary_fill_holes(tis > 0)
    mask = ndi.binary_fill_holes(ndi.binary_closing(brain, iterations=r["mask_close_px"])) & icv
    return {"mask": mask | brain, "brain": brain, "wm": wm}


def rounded(masks, r):
    """M4, first half. Every layer lies inside the layer below it."""
    out, below = {}, None
    for k, _ in LAYERS:
        m = round_mask(masks[k], r["mask_round_sigma"], r["min_object_px"])
        out[k] = below = m if below is None else m & below
    return out


def check_ys(cfg, meta, path):
    """Every configured y must be a whole number in the planes file. Raises ValueError that names both files."""
    missing = [s["y_mm"] for s in cfg["slices"] if type(s["y_mm"]) is not int or s["y_mm"] not in meta["ys"]]
    if missing:
        raise ValueError(f"{cfg['_path']}: y_mm {missing} is not in {path}, which holds y = {meta['ys']}")


def build(cfg=None, log=print):
    cfg = cfg or load("mri")
    r, v, pal = cfg["raster"], cfg["vector"], cfg["palette"]
    data, meta = load_planes(cfg)
    g = scan.grid(meta, cfg)
    vs = g[2]
    scale = vs / v["unit_mm"]                                        # frame units per pixel
    secs = cfg["slices"]
    check_ys(cfg, meta, planes_path(cfg))
    planes = [plane(data, meta, s["y_mm"], cfg) for s in secs]
    ms = [rounded(classes(p, cfg), r) for p in planes]
    centres = [scan.slice_centre(s, p["structure"], r) for s, p in zip(secs, planes)]
    try:
        ox, oy, vw, vh = scan.frame([m["mask"] for m in ms], v, scale)
    except ValueError:
        raise ValueError(f"{cfg['_path']}: the lists in tissue_labels match no pixel of the configured planes") from None
    MRI_OUT.mkdir(parents=True, exist_ok=True)
    lic = cfg["source"]["license"]
    out = {"viewBox": [0, 0, vw, vh], "mm_per_unit": v["unit_mm"], "frame_mm": scan.frame_mm(ox, oy, g), "orientation": cfg["orientation"],
           "license": lic, "config": cfg["_path"], "slices": []}
    for s, m, centre in zip(secs, ms, centres):
        dm = path_d(m["mask"], ox, oy, v, scale)
        body = "".join(scan.fill(dm if k == "mask" else path_d(m[k], ox, oy, v, scale), pal[c]) for k, c in LAYERS)
        svg = scan.svg(vw, vh, f'Coronal MRI slice of the skull-stripped brain, y = {s["y_mm"]} mm, {s["label"]}', lic,
                       body + scan.stroke(dm, pal["outline"], v["outline_width"]))
        f = MRI_OUT / f"slice_{s['k']}.svg"
        f.write_text(svg, encoding="utf-8")
        it = scan.item(s, centre, ox, oy, g, scale, v["roi_units"], f.name, {k: round(float(m[k].sum()) * vs * vs) for k, _ in LAYERS})
        out["slices"].append(it)
        log(f"{f.name}: y {s['y_mm']:+d} mm, {len(svg) // 1024} KB" + (f", {s['roi']} at {it['roi_mm']} mm" if it["roi_mm"] else ""))
    (MRI_OUT / "slices.json").write_text(json.dumps(out, indent=1))
    log(f"slices.json: viewBox {vw} x {vh}, {v['unit_mm']:.2f} mm per unit, {cfg['orientation']}")
    return out


if __name__ == "__main__":
    build()
    sys.exit(0)

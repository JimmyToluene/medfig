"""Steps P2 to P6: PET planes -> flat SVG slices of the brain with stepped uptake bands, one series per template.

Usage: python -m medfig pet slices
Input:  the PET planes .npz (config source.planes_npz), the MRI planes .npz (brain mask and structure labels), config/pet.json
        and config/mri.json (the brain mask is made with its parameters).
Output: out/pet/<series>/slice_{k}.svg  brain mask (first band colour), one fill per band, outline
        out/pet/<series>/slices.json    viewBox, mm per unit, frame position in mm, orientation, bands, markers
        out/pet/<series>/key.svg        colour key of the bands

P2  suvr(): the native PET plane at y is sampled linearly at the pixel centres of the MRI grid (0.5 mm), divided by the
    reference mean of its template (planes file meta) and put in the configured orientation (scan.py).
P3  the brain mask of the MRI slice of the same y (mri.slices.classes, mri.slices.rounded with config/mri.json), so the
    PET outline is the MRI outline. The SUVR plane is smoothed (bands.smooth_sigma_mm); band k = brain mask and
    smoothed SUVR >= edges[k], rounded, inside band k - 1.
P4  trace, smooth and simplify (vector.py).
P5  one frame for both series: the bounding box of the union of the brain masks plus padding (the same frame as MRI).
    Markers: centroid of the roi_labels of holiAtlas, as in MRI.
P6  write SVG: the brain mask in the colour of the lowest band, one fill per band, the outline of the mask.

build() checks every y and computes every marker before it writes the first file.
"""
import json
import sys

import numpy as np
from scipy import ndimage as ndi

from .. import scan
from ..config import OUT, load, planes_path
from ..mri import slices as M
from ..vector import path_d, round_mask

PET_OUT = OUT / "pet"
FONT = 'font-family="Lora, Georgia, serif"'


def load_planes(cfg):
    z = np.load(planes_path(cfg))
    return {k: z[k] for k in cfg["source"]["volumes"]}, json.loads(str(z["meta"]))


def suvr(data, meta, key, y, mmeta, cfg):
    """P2. SUVR plane of one template at y on the MRI grid: rows superior -> inferior, columns in the orientation of cfg."""
    a = data[key][meta["ys"].index(y)]                                          # (z, x), native grid
    vs, o = meta["vs"], meta["origin_mm"]
    xs = mmeta["origin_mm"][0] + mmeta["vs"] * np.arange(mmeta["dims"][0])
    zs = mmeta["origin_mm"][2] + mmeta["vs"] * np.arange(mmeta["dims"][2])
    zz, xx = np.meshgrid((zs - o[2]) / vs, (xs - o[0]) / vs, indexing="ij")
    s = ndi.map_coordinates(a, [zz, xx], order=1, mode="constant", cval=0.0) / meta["reference"]["mean"][key]
    return scan.orient(s, cfg)


def brain(mp, mcfg):
    """P3. The rounded MRI brain mask of an MRI plane (the MRI outline)."""
    return M.rounded(M.classes(mp, mcfg), mcfg["raster"])["mask"]


def bands(s, mask, cfg, px_mm):
    """P3. Nested band masks, lowest first: [mask, band 1, ..., band n]. Every band lies inside the one below it.
    px_mm is the pixel size of the plane."""
    b, r = cfg["bands"], cfg["raster"]
    sm = ndi.gaussian_filter(s, b["smooth_sigma_mm"] / px_mm)
    out, below = [mask], mask
    for e in b["edges"]:
        below = round_mask((sm >= e) & mask, r["mask_round_sigma"], r["min_object_px"]) & below
        out.append(below)
    return out


def key_svg(cfg):
    """The colour key: one swatch per colour, the band edges between them, the unit at the right."""
    k, pal, b = cfg["key"], cfg["palette"], cfg["bands"]
    w, h, fs, n = k["swatch_w"], k["swatch_h"], k["font_px"], len(pal["bands"])
    W, H = n * w + 4 + 6 * fs, h + 2 * fs + 4
    sw = "".join(f'<rect x="{2 + i * w}" y="2" width="{w}" height="{h}" fill="{c}"/>' for i, c in enumerate(pal["bands"]))
    tx = "".join(f'<text x="{2 + (i + 1) * w}" y="{h + 3 + fs}" text-anchor="middle" font-size="{fs}" fill="{pal["key_text"]}" {FONT}>{e:.1f}</text>'
                 for i, e in enumerate(b["edges"]))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
            f'aria-label="Colour key: {b["unit"]} bands, edges {", ".join(f"{e:.1f}" for e in b["edges"])}">'
            f'{sw}'
            f'<rect x="2" y="2" width="{n * w}" height="{h}" fill="none" stroke="{pal["outline"]}" stroke-width="1"/>{tx}'
            f'<text x="{n * w + 8}" y="{h / 2 + 2 + fs / 3:g}" font-size="{fs}" fill="{pal["key_text"]}" {FONT}>{b["unit"]}</text></svg>')


def check(cfg, meta):
    if len(cfg["palette"]["bands"]) != len(cfg["bands"]["edges"]) + 1:
        raise ValueError(f"{cfg['_path']}: palette.bands needs one colour more than bands.edges")
    if list(cfg["bands"]["edges"]) != sorted(cfg["bands"]["edges"]):
        raise ValueError(f"{cfg['_path']}: bands.edges must ascend")
    missing = [s["channel"] for s in cfg["series"] if s["channel"] not in meta["reference"]["mean"]]
    if missing:
        raise ValueError(f"{cfg['_path']}: series channel {missing} is not in {planes_path(cfg)}")


def build(cfg=None, log=print):
    cfg = cfg or load("pet")
    mcfg = load("mri")
    r, v, pal = cfg["raster"], cfg["vector"], cfg["palette"]
    data, meta = load_planes(cfg)
    mdata, mmeta = M.load_planes(mcfg)
    check(cfg, meta)
    secs = cfg["slices"]
    M.check_ys(cfg, meta, planes_path(cfg))
    M.check_ys(cfg, mmeta, planes_path(mcfg))
    g = scan.grid(mmeta, cfg)
    vs = g[2]
    scale = vs / v["unit_mm"]
    mps = [M.plane(mdata, mmeta, s["y_mm"], cfg) for s in secs]
    masks = [brain(mp, mcfg) for mp in mps]
    centres = [scan.slice_centre(s, mp["structure"], r) for s, mp in zip(secs, mps)]
    layers = {se["key"]: [bands(suvr(data, meta, se["channel"], s["y_mm"], mmeta, cfg), m, cfg, vs) for s, m in zip(secs, masks)] for se in cfg["series"]}
    ox, oy, vw, vh = scan.frame(masks, v, scale)
    lic = cfg["source"]["license"]
    key = key_svg(cfg)
    outs = {}
    for se in cfg["series"]:
        d = PET_OUT / se["key"]
        d.mkdir(parents=True, exist_ok=True)
        out = {"viewBox": [0, 0, vw, vh], "mm_per_unit": v["unit_mm"], "frame_mm": scan.frame_mm(ox, oy, g), "orientation": cfg["orientation"],
               "series": se, "bands": cfg["bands"]["edges"], "colours": pal["bands"], "reference": meta["reference"]["name"],
               "reference_mean": meta["reference"]["mean"][se["channel"]], "license": lic, "outline_credit": cfg["source"]["outline_credit"],
               "config": cfg["_path"], "slices": []}
        for s, ls, centre in zip(secs, layers[se["key"]], centres):
            dm = path_d(ls[0], ox, oy, v, scale)
            body = "".join(scan.fill(dm if i == 0 else path_d(m, ox, oy, v, scale), c) for i, (m, c) in enumerate(zip(ls, pal["bands"])))
            svg = scan.svg(vw, vh, f'Coronal amyloid PET slice, {se["name"]}, y = {s["y_mm"]} mm, {s["label"]}', lic,
                           body + scan.stroke(dm, pal["outline"], v["outline_width"]))
            f = d / f"slice_{s['k']}.svg"
            f.write_text(svg, encoding="utf-8")
            areas = {"mask": round(float(ls[0].sum()) * vs * vs), "bands": [round(float(m.sum()) * vs * vs) for m in ls[1:]]}
            it = scan.item(s, centre, ox, oy, g, scale, v["roi_units"], f.name, areas)
            out["slices"].append(it)
            log(f"{se['key']}/{f.name}: y {s['y_mm']:+d} mm, {len(svg) // 1024} KB, band areas {areas['bands']} mm2 of {areas['mask']}")
        (d / "slices.json").write_text(json.dumps(out, indent=1))
        (d / "key.svg").write_text(key, encoding="utf-8")
        outs[se["key"]] = out
    log(f"slices.json: viewBox {vw} x {vh}, {v['unit_mm']:.2f} mm per unit, {cfg['orientation']}, series {', '.join(outs)}")
    return outs


if __name__ == "__main__":
    build()
    sys.exit(0)

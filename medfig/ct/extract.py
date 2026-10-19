"""Step C1: the intracranial mask of the CT template in 3D, and coronal planes on the grid of the MRI planes file.

Usage:
    python -m medfig ct extract CT_DIR [OUT.npz] [--y 64,56,...]

CT_DIR holds the CT template of the Clinical Toolbox (config source.url); the script reads source.volume
(scct_unsmooth.nii: 1 mm, MNI, Hounsfield units). `--y` takes whole millimetres on the 1 mm grid of the CT.

The intracranial mask (config hu): voxels inside soft_window are eroded by erode_vox, the largest part is kept and grown
back by erode_vox + 1 voxels inside grow_from <= HU < bone (so it stops at the bone), then the holes are filled in every
plane along each of the three axes. It is made in 3D because a single plane can cut the intracranial space in two.

The planes are sampled on the grid of the MRI planes file (config/mri.json source.planes_npz: 0.5 mm, the holiAtlas
grid), so a CT plane and an MRI plane of the same y share every pixel: HU linearly (value hu.outside outside the CT),
the mask linearly and cut at 0.5.
Output .npz: `hu` int16 and `icv` uint8, each (plane, z, x) with z and x ascending, and `meta` JSON: ys, origin_mm, dims,
vs (the MRI grid), axes, source, volume, ct (origin_mm, dims, vs of the CT volume, z_min_mm = lowest CT plane), hu
(the thresholds used), icv_ml (volume of the 3D mask). Needs numpy and scipy; about 10 s.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ..config import load, planes_path
from ..nifti import load_volume

MASK_KEYS = ("soft_window", "erode_vox", "grow_from", "bone", "outside")     # the hu values that the planes file depends on


def largest(m):
    """The largest connected part of a mask (face neighbours in 3D, edge neighbours in 2D)."""
    lab, n = ndi.label(m)
    return m if n < 2 else lab == (1 + np.argmax(ndi.sum(m, lab, range(1, n + 1))))


def intracranial(hu, h):
    """The 3D intracranial mask of a CT volume (z, y, x). h is the config block hu."""
    lo, hi = h["soft_window"]
    ball = ndi.generate_binary_structure(3, 1)
    core = largest(ndi.binary_erosion((hu >= lo) & (hu <= hi), ball, iterations=h["erode_vox"]))
    m = ndi.binary_dilation(core, ball, iterations=h["erode_vox"] + 1, mask=(hu >= h["grow_from"]) & (hu < h["bone"]))
    for ax in range(3):
        m = np.moveaxis(np.array([ndi.binary_fill_holes(s) for s in np.moveaxis(m, ax, 0)]), 0, ax)
    return m


def sample(vol, origin, vs, xs, y, zs, cval):
    """Linear sample of a (z, y, x) volume at the points (x, y, z) of the plane y (x and z ascending). Returns (z, x)."""
    zz, xx = np.meshgrid((np.asarray(zs) - origin[2]) / vs, (np.asarray(xs) - origin[0]) / vs, indexing="ij")
    return ndi.map_coordinates(vol, [zz, np.full_like(zz, (y - origin[1]) / vs), xx], order=1, mode="constant", cval=cval)


def extract(src_dir, out, ys, cfg, mri_meta, log=print):
    t0 = time.time()
    h = cfg["hu"]
    hu, origin, vs = load_volume(Path(src_dir) / cfg["source"]["volume"])
    hu = hu.astype(np.float32)
    nz, ny, nx = hu.shape
    off = [y for y in ys if not 0 <= (y - origin[1]) / vs <= ny - 1]
    if off:
        raise ValueError(f"y {off} mm is outside the CT volume (y = {origin[1]} to {origin[1] + vs * (ny - 1)} mm)")
    off = [y for y in ys if abs((y - origin[1]) / vs - round((y - origin[1]) / vs)) > 1e-6]
    if off:
        raise ValueError(f"y {off} mm is not on the voxel grid of the CT ({vs} mm steps from y = {origin[1]} mm)")
    icv3 = intracranial(hu, h)
    log(f"{cfg['source']['volume']}: {nx} x {ny} x {nz}, {vs} mm; intracranial mask {icv3.sum() * vs ** 3 / 1000:.0f} ml")
    g = mri_meta
    xs = g["origin_mm"][0] + g["vs"] * np.arange(g["dims"][0])
    zs = g["origin_mm"][2] + g["vs"] * np.arange(g["dims"][2])
    out_hu = np.stack([np.round(sample(hu, origin, vs, xs, y, zs, h["outside"])) for y in ys]).astype(np.int16)
    out_icv = np.stack([sample(icv3.astype(np.float32), origin, vs, xs, y, zs, 0.0) > 0.5 for y in ys]).astype(np.uint8)
    meta = {"ys": list(ys), "origin_mm": g["origin_mm"], "dims": g["dims"], "vs": g["vs"], "axes": "(plane, z, x)",
            "source": cfg["source"]["name"], "volume": cfg["source"]["volume"],
            "ct": {"origin_mm": [float(v) for v in origin], "dims": [nx, ny, nz], "vs": vs, "z_min_mm": float(origin[2])},
            "hu": {k: h[k] for k in MASK_KEYS}, "icv_ml": round(float(icv3.sum()) * vs ** 3 / 1000, 1)}
    np.savez_compressed(out, meta=json.dumps(meta), hu=out_hu, icv=out_icv)
    log(f"wrote {out}  {out_hu.shape}  {time.time() - t0:.0f} s")
    return out


def main(argv=None):
    cfg = load("ct")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ct_dir")
    ap.add_argument("out", nargs="?", default=str(planes_path(cfg)))
    ap.add_argument("--y", default=",".join(str(v) for v in cfg["source"]["candidate_y_mm"]),
                    help="coronal positions in mm, comma separated")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--y" in argv[:-1]:                       # argparse reads `--y -22,8` as an option, but takes `--y=-22,8`
        i = argv.index("--y")
        argv[i:i + 2] = ["--y=" + argv[i + 1]]
    a = ap.parse_args(argv)
    mcfg = load("mri")
    with np.load(planes_path(mcfg)) as z:
        mri_meta = json.loads(str(z["meta"]))
    extract(a.ct_dir, a.out, [int(v) for v in a.y.split(",")], cfg, mri_meta)


if __name__ == "__main__":
    sys.exit(main())

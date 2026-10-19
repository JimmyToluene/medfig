"""Step P1: native coronal planes of the PET templates, and the mean of each template over the reference region.

Usage:
    python -m medfig pet extract PET_DIR HOLIATLAS_DIR [OUT.npz] [--y 64,56,...]

PET_DIR holds the rPOP templates (config source.url): the volumes of config source.volumes, 2 mm, MNI. HOLIATLAS_DIR holds
the unzipped holiAtlas template: the tissue atlas (config reference.tissue_volume) gives the reference region.
`--y` takes whole millimetres on the grid of the templates.

Reference mean of a template: PET sampled linearly at the centre of every holiAtlas voxel whose tissue label is in
reference.tissue_labels (the whole cerebellum), points outside the PET field of view (value 0) left out, then averaged.
The slices step divides by it, so the planes show SUVR. It is made in 3D because the cerebellum is not in every plane.
Output .npz: one float32 array per volume key, (plane, z, x) with z and x ascending, as distributed (2 mm), and `meta`
JSON: ys, js, origin_mm, dims, vs, axes, source, volumes, reference (labels, mean of each key, points used).
Needs numpy and scipy; about 5 s.
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


def reference_mean(vol, origin, vs, tissue, t_origin, t_vs, labels):
    """Mean of a (z, y, x) volume, sampled linearly at the centres of the labelled voxels of a (z, y, x) label volume.
    Returns (mean, number of points inside the field of view)."""
    kk, jj, ii = np.nonzero(np.isin(tissue, labels))
    pts = [(t_origin[2] + kk * t_vs - origin[2]) / vs, (t_origin[1] + jj * t_vs - origin[1]) / vs, (t_origin[0] + ii * t_vs - origin[0]) / vs]
    v = ndi.map_coordinates(vol, pts, order=1, mode="constant", cval=0.0)
    v = v[v != 0]
    if not v.size:
        raise ValueError(f"the reference labels {labels} are outside the PET field of view")
    return float(v.mean()), int(v.size)


def extract(pet_dir, holi_dir, out, ys, cfg, log=print):
    t0 = time.time()
    ref = cfg["reference"]
    tissue, t_origin, t_vs = load_volume(Path(holi_dir) / ref["tissue_volume"])
    arrs, grid, means, n = {}, None, {}, {}
    for key, name in cfg["source"]["volumes"].items():
        a, origin, vs = load_volume(Path(pet_dir) / name)
        g = (a.shape, tuple(origin), vs)
        if grid is not None and g != grid:
            raise ValueError(f"{name}: the grid is not the grid of the other volumes")
        grid = g
        js = [(y - origin[1]) / vs for y in ys]
        off = [y for y, j in zip(ys, js) if not 0 <= j <= a.shape[1] - 1]
        if off:
            raise ValueError(f"{name}: y {off} mm is outside the volume")
        off = [y for y, j in zip(ys, js) if abs(j - round(j)) > 1e-6]
        if off:
            raise ValueError(f"{name}: y {off} mm is not on the voxel grid ({vs} mm steps from y = {origin[1]} mm)")
        js = [int(round(j)) for j in js]
        arrs[key] = a[:, js, :].transpose(1, 0, 2).astype(np.float32)              # (plane, z, x)
        means[key], n[key] = reference_mean(a, origin, vs, tissue, t_origin, t_vs, ref["tissue_labels"])
        log(f"{name}: {a.shape[2]} x {a.shape[1]} x {a.shape[0]}, {vs} mm; {ref['name']} mean {means[key]:.4f} over {n[key]} points")
    (nz, ny, nx), origin, vs = grid
    meta = {"ys": list(ys), "js": js, "origin_mm": list(origin), "dims": [nx, ny, nz], "vs": vs, "axes": "(plane, z, x)",
            "source": cfg["source"]["name"], "volumes": cfg["source"]["volumes"],
            "reference": {"name": ref["name"], "tissue_labels": ref["tissue_labels"], "mean": means, "points": n}}
    np.savez_compressed(out, meta=json.dumps(meta), **arrs)
    log(f"wrote {out}  {next(iter(arrs.values())).shape}  {time.time() - t0:.0f} s")
    return out


def main(argv=None):
    cfg = load("pet")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pet_dir")
    ap.add_argument("holiatlas_dir")
    ap.add_argument("out", nargs="?", default=str(planes_path(cfg)))
    ap.add_argument("--y", default=",".join(str(v) for v in cfg["source"]["candidate_y_mm"]),
                    help="coronal positions in mm, comma separated")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--y" in argv[:-1]:                       # argparse reads `--y -22,8` as an option, but takes `--y=-22,8`
        i = argv.index("--y")
        argv[i:i + 2] = ["--y=" + argv[i + 1]]
    a = ap.parse_args(argv)
    extract(a.pet_dir, a.holiatlas_dir, a.out, [int(v) for v in a.y.split(",")], cfg)


if __name__ == "__main__":
    sys.exit(main())

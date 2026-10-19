"""Step M1: cut whole coronal planes from the holiAtlas 0.5 mm template.

Usage:
    python -m medfig mri extract HOLIATLAS_DIR [OUT.npz] [--y 64,56,...]

`--y` takes whole millimetres. The list may start with a negative value (`--y -22,8`).
Each y must be on the voxel grid of the template. A y between two planes stops the script.

HOLIATLAS_DIR holds the unzipped holiatlas_v1.0.zip (config source.url). The script reads the three volumes named in
config source.volumes: the T1w average, the tissue labels and the structure labels. Each volume is 362 x 434 x 362
voxels and fits in memory. It needs numpy only and takes about 1 s.

Output .npz: `t1w` uint16, `tissue` uint8, `structure` uint8, each (plane, z, x) with z ascending as in the file,
and `meta` JSON (ys, js, origin_mm, dims, vs, axes, source, volumes).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..config import load, planes_path
from ..nifti import load_volume

DTYPE = {"t1w": np.uint16, "tissue": np.uint8, "structure": np.uint8}


def extract(src_dir, out, ys, cfg, log=print):
    t0 = time.time()
    arrs, grid = {}, None
    for key, name in cfg["source"]["volumes"].items():
        a, origin, vs = load_volume(Path(src_dir) / name)
        g = (a.shape, tuple(origin), vs)
        if grid is not None and g != grid:
            raise ValueError(f"{name}: the grid is not the grid of the other volumes")
        grid = g
        js = [int(round((y - origin[1]) / vs)) for y in ys]
        if min(js) < 0 or max(js) >= a.shape[1]:
            raise ValueError(f"y outside the volume: {ys}")
        off = [y for y, j in zip(ys, js) if abs(origin[1] + j * vs - y) > 1e-6]
        if off:
            raise ValueError(f"{name}: y {off} mm is not on the voxel grid ({vs} mm steps from y = {origin[1]} mm)")
        p = a[:, js, :].transpose(1, 0, 2)                       # (plane, z, x)
        if p.min() < 0 or p.max() > np.iinfo(DTYPE[key]).max or not np.array_equal(p, np.round(p)):
            raise ValueError(f"{name}: values do not fit {DTYPE[key].__name__}")
        arrs[key] = p.astype(DTYPE[key])
        log(f"{name}: {a.shape[2]} x {a.shape[1]} x {a.shape[0]}, {vs} mm")
    (nz, ny, nx), origin, vs = grid
    meta = {"ys": list(ys), "js": js, "origin_mm": list(origin), "dims": [nx, ny, nz], "vs": vs, "axes": "(plane, z, x)",
            "source": cfg["source"]["name"], "volumes": cfg["source"]["volumes"]}
    np.savez_compressed(out, meta=json.dumps(meta), **arrs)
    log(f"wrote {out}  {arrs['t1w'].shape}  {time.time() - t0:.0f} s")
    return out


def main(argv=None):
    cfg = load("mri")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("holiatlas_dir")
    ap.add_argument("out", nargs="?", default=str(planes_path(cfg)))
    ap.add_argument("--y", default=",".join(str(v) for v in cfg["source"]["candidate_y_mm"]),
                    help="coronal positions in mm, comma separated")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--y" in argv[:-1]:                       # argparse reads `--y -22,8` as an option, but takes `--y=-22,8`
        i = argv.index("--y")
        argv[i:i + 2] = ["--y=" + argv[i + 1]]
    a = ap.parse_args(argv)
    extract(a.holiatlas_dir, a.out, [int(v) for v in a.y.split(",")], cfg)


if __name__ == "__main__":
    sys.exit(main())

"""Step S1: cut whole coronal planes from the BigBrain 100 um volume without loading the volume.

Usage:
    python -m medfig.extract VOLUME.nii.gz [OUT.npz] [--y 24,16,8,...]

The volume is a gzip NIfTI-1 file, uint8, 1970 x 2330 x 1890 voxels (about 8.7 GB raw, 1.1 GB compressed).
NIfTI stores x fastest, then y, then z. A coronal plane (fixed y) is one row in each z-layer.
The script reads one z-layer at a time and keeps the rows of the requested y values.
Memory stays below 100 MB. It needs numpy only. On a laptop CPU it takes about 22 s.

Output .npz: `data` uint8 (plane, z, x) and `meta` JSON (ys, js, srow, dims, axes, source, vs).
"""
import argparse
import gzip
import json
import struct
import sys
import time

import numpy as np

from .config import load, planes_path


def read_header(f):
    h = f.read(352)
    if struct.unpack("<i", h[0:4])[0] != 348:
        raise ValueError("not a little-endian NIfTI-1 file")
    dim = struct.unpack("<8h", h[40:56])
    datatype = struct.unpack("<h", h[70:72])[0]
    vox_offset = struct.unpack("<f", h[108:112])[0]
    srow = [struct.unpack("<4f", h[o:o + 16]) for o in (280, 296, 312)]
    return dim, datatype, vox_offset, srow


def extract(src, out, ys, log=print):
    t0 = time.time()
    with gzip.open(src, "rb") as f:
        dim, datatype, vox_offset, srow = read_header(f)
        nx, ny, nz = dim[1:4]
        if datatype != 2:
            raise ValueError(f"expects uint8 (datatype 2), got {datatype}")
        if abs(srow[1][1]) < 1e-9 or srow[1][0] or srow[1][2]:
            raise ValueError("expects an axis-aligned y axis")
        f.read(int(vox_offset) - 352)
        js = [int(round((y - srow[1][3]) / srow[1][1])) for y in ys]      # y = srow_y[1] * j + srow_y[3]
        if min(js) < 0 or max(js) >= ny:
            raise ValueError(f"y outside the volume: {ys}")
        planes = np.zeros((len(ys), nz, nx), np.uint8)
        for k in range(nz):
            layer = np.frombuffer(f.read(nx * ny), np.uint8).reshape(ny, nx)
            planes[:, k, :] = layer[js, :]
            if k % 400 == 0:
                log(f"z {k}/{nz}  {time.time() - t0:.0f} s")
    meta = {"ys": list(ys), "js": js, "srow": srow, "dims": [nx, ny, nz], "axes": "(plane, z, x)",
            "source": str(src).split("/")[-1].replace(".nii.gz", ""), "vs": abs(srow[0][0])}
    np.savez_compressed(out, data=planes, meta=json.dumps(meta))
    log(f"wrote {out}  {planes.shape}  {time.time() - t0:.0f} s")
    return out


def main(argv=None):
    cfg = load()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("volume")
    ap.add_argument("out", nargs="?", default=str(planes_path(cfg)))
    ap.add_argument("--y", default=",".join(str(v) for v in cfg["source"]["candidate_y_mm"]),
                    help="coronal positions in mm, comma separated")
    a = ap.parse_args(argv)
    extract(a.volume, a.out, [int(v) for v in a.y.split(",")])


if __name__ == "__main__":
    sys.exit(main())

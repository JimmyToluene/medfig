"""Read a small NIfTI-1 volume with numpy only (for volumes that fit in memory, such as a 0.5 mm MRI template).

load_volume() returns the voxels as (z, y, x) with x, y and z ascending in mm, and the mm position of voxel (0, 0, 0).
The BigBrain 100 um volume does not fit in memory: extract.py streams it one z-layer at a time.
"""
import gzip
import struct

import numpy as np

DTYPES = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4"}


def load_volume(path):
    """Returns (data[z, y, x], origin_mm (x, y, z), voxel size mm). Expects little-endian, axis-aligned NIfTI-1."""
    with (gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")) as f:
        raw = f.read()
    h = raw[:352]
    if struct.unpack("<i", h[0:4])[0] != 348:
        raise ValueError(f"{path}: not a little-endian NIfTI-1 file")
    dim = struct.unpack("<8h", h[40:56])
    datatype = struct.unpack("<h", h[70:72])[0]
    vox_offset = int(struct.unpack("<f", h[108:112])[0])
    slope, inter = struct.unpack("<2f", h[112:120])
    srow = np.array([struct.unpack("<4f", h[o:o + 16]) for o in (280, 296, 312)], float)
    if datatype not in DTYPES:
        raise ValueError(f"{path}: NIfTI datatype {datatype} is not supported")
    step = np.diag(srow[:, :3])
    if np.abs(step).min() < 1e-6:
        raise ValueError(f"{path}: no sform (the voxel-to-mm rows are empty)")
    if np.abs(srow[:, :3] - np.diag(step)).max() > 1e-6 or np.ptp(np.abs(step)) > 1e-6:
        raise ValueError(f"{path}: expects axis-aligned, isotropic voxels")
    nx, ny, nz = dim[1:4]
    a = np.frombuffer(raw, "<" + DTYPES[datatype], nx * ny * nz, vox_offset).reshape(nz, ny, nx)
    origin = srow[:, 3].copy()
    for axis, i in ((2, 0), (1, 1), (0, 2)):                 # make x, y and z ascend
        if step[i] < 0:
            a = np.flip(a, axis)
            origin[i] += step[i] * (a.shape[axis] - 1)
    if np.isfinite(slope) and slope != 0.0 and (slope != 1.0 or inter != 0.0):       # a slope of 0 (NIfTI-1) or a slope that is not finite (NaN) means "no scaling"
        a = a * slope + inter
    return a, [float(v) for v in origin], float(abs(step[0]))

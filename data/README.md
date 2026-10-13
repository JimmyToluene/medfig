# data/

| File | What | Size |
|---|---|---|
| `bigbrain_coronal_100um_sel.npz` | 12 whole coronal planes, y = +24 ... -72 mm, cut from BigBrain `full8_100um_2009b_sym` | 10 MB |

Keys: `data` uint8 (plane, z, x) = (12, 1890, 1970), 100 µm. `meta` JSON: `ys` (mm), `js` (voxel index), `srow` (NIfTI sform rows),
`dims`, `axes`, `source`, `vs` (voxel size, mm). Rows are z ascending as in the file; `sections.plane()` flips them.

Made with `python -m medfig extract full8_100um_2009b_sym.nii.gz` (22 s on the lab PC, 2026-10-13). Check: `sha256sum -c SHA256SUMS`.

## Licence

BigBrain: Amunts K. et al., "BigBrain: An Ultrahigh-Resolution 3D Human Brain Model", Science 340(6139):1472–1475, 2013.
The data are released under CC BY-NC-SA 4.0. This file and every SVG made from it are adapted material under the same licence:
give credit, non-commercial use only, share under the same licence.

## The full volume

The full volume is not in this repository (1.1 GB). The BigBrain release lists it under
`BigBrainRelease.2015/3D_Volumes/MNI-ICBM152_Space/` on ftp.bigbrainproject.org (`.mnc`; the `.nii.gz` copy used here came from the
same release). `medfig extract` reads the `.nii.gz` (gzip NIfTI-1, uint8) only.

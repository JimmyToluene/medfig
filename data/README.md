# data/

| File | What | Size |
|---|---|---|
| `bigbrain_coronal_100um_sel.npz` | 12 whole coronal planes, y = +24 ... -72 mm, cut from BigBrain `full8_100um_2009b_sym` | 10 MB |
| `holiatlas_coronal_500um_sel.npz` | 22 whole coronal planes, y = +64 ... -96 mm, cut from the holiAtlas v1.0 template: T1w, tissue labels, structure labels | 2.6 MB |
| `clinical_ct_coronal_500um_sel.npz` | the same 22 planes of the Clinical Toolbox CT template on the grid of the holiAtlas planes: HU and the intracranial mask | 1.9 MB |
| `rpop_fbb_coronal_2mm_sel.npz` | the same 22 planes of the rPOP florbetaben templates (amyloid-negative, amyloid-positive), native 2 mm | 1.0 MB |
| `LICENSE-clinical-ct.txt` | the BSD 2-Clause licence of the Clinical Toolbox repository (its condition 2 asks for this copy) | 1 KB |

Check the planes files from the repository root: `sha256sum -c data/SHA256SUMS`.

## BigBrain planes (WSI)

Keys: `data` uint8 (plane, z, x) = (12, 1890, 1970), 100 µm. `meta` JSON: `ys` (mm), `js` (voxel index), `srow` (NIfTI sform rows),
`dims`, `axes`, `source`, `vs` (voxel size, mm). Rows are z ascending as in the file; `sections.plane()` flips them.

Made with `python -m medfig extract full8_100um_2009b_sym.nii.gz` (22 s on the lab PC, 2026-10-13).

## holiAtlas planes (MRI)

Keys: `t1w` uint16, `tissue` uint8, `structure` uint8, each (plane, z, x) = (22, 362, 362), 0.5 mm. `meta` JSON: `ys` (mm),
`js` (voxel index), `origin_mm` (x, y, z of voxel 0), `dims`, `vs` (voxel size, mm), `axes`, `source`, `volumes` (file names).
Rows are z ascending; `mri.slices.plane()` flips them.

- `t1w`: the T1-weighted average of the template. Only the report figures show it. The SVG slices do not use it:
  the classes come from the labels.
- `tissue`: 0 no tissue label (outside the intracranial volume, and a few enclosed pockets), 1 CSF, 2 cortical GM,
  3 cerebral WM, 4 deep GM, 5 cerebellar GM, 6 cerebellar WM, 7 brainstem, 8 WM lesion, 9 vessels and connective tissue.
  `config/mri.json` (`tissue_labels`) says which labels are grey matter and which are white matter.
- `structure`: the labels of `structure_atlas.nii.gz`. `HoliBrain_Protocol_v1.0.xlsx` in the zip gives the names.
  `config/mri.json` (`roi_labels` of each slice) gives the labels that set the marker of the slice.

Made with `python -m medfig mri extract /path/holiatlas` (about 1 s on the lab DGX Spark, 2026-10-13). The directory holds the
unzipped template. The MRI stack uses 3 of the 22 planes. The other planes are candidates for other choices.

## Clinical Toolbox CT planes (CT)

Keys: `hu` int16 (Hounsfield units) and `icv` uint8 (0/1), each (plane, z, x) = (22, 362, 362), 0.5 mm, on the grid of the
holiAtlas planes file (same `origin_mm`, `dims`, `vs`), so a CT plane and an MRI plane of the same y share every pixel.
`meta` JSON: `ys`, `origin_mm`, `dims`, `vs`, `axes`, `source`, `volume`, `ct` (origin, dims, voxel size of the CT volume and
`z_min_mm`, its lowest plane), `hu` (the thresholds that made `icv`), `icv_ml`. Rows are z ascending; `ct.slices.plane()` flips
them and cuts them at `z_min_mm`.

- `hu`: the CT template sampled linearly at the pixel centres; -1024 (`hu.outside` in `config/ct.json`) outside the CT volume.
- `icv`: the intracranial mask, made in 3D on the 1 mm CT volume (see `medfig/ct/extract.py`), sampled linearly and cut at 0.5.

Made with `python -m medfig ct extract ~/data/clinical_ct` (about 1 s on the lab DGX Spark, 2026-10-14) from
`scct_unsmooth.nii` (181 x 217 x 181 voxels, 1 mm, float, 14 218 626 bytes, SHA-256
`0fbeedea171bcd61db6dea967a3d5cbaebce8f0b225274e074bce3547724ac9c`), file `Tutorial/high_res/scct_unsmooth.nii` of
https://github.com/neurolabusc/Clinical (commit `1b6bc35`, 2021-05-19).
Its header puts it in MNI space as it is; its intracranial mask matches the holiAtlas intracranial volume with a Dice
coefficient of 0.95 to 0.97 at the three planes of the figure (no shift applied). The scans of the template were
contrast-enhanced, and the volume ends at z = -71 mm, so the figure is cut flat there.

## rPOP florbetaben planes (PET)

Keys: `fbb_neg`, `fbb_pos` float32, each (plane, z, x) = (22, 86, 91), 2 mm, the values of the templates as distributed.
`meta` JSON: `ys`, `js`, `origin_mm`, `dims`, `vs`, `axes`, `source`, `volumes`, `reference` (name, holiAtlas tissue labels,
the mean of each template over the reference region, the number of points). `pet.slices` samples a plane onto the 0.5 mm
grid of the holiAtlas planes and divides it by the reference mean (SUVR).

The reference region is the whole cerebellum of holiAtlas (tissue labels 5 and 6): each template is sampled linearly at the
centre of every such voxel, values 0 (outside the field of view) are left out, and the rest is averaged. This is done on the
whole volume, because the cerebellum is not in every plane.

Made with `python -m medfig pet extract ~/data/rpop ~/data/holiatlas` (under 1 s on the lab DGX Spark, 2026-10-14) from
`templates/Template_FBB_neg.nii` and `templates/Template_FBB_pos.nii` of https://github.com/LeoIacca/rPOP (commit `0f6af1c`,
2021-12-13; 91 x 111 x 86 voxels, 2 mm, 1 737 724 bytes each; SHA-256 `292caa6c...a101195` and `6eb2f0e6...224a7ee2`).

## Licence

BigBrain: Amunts K. et al., "BigBrain: An Ultrahigh-Resolution 3D Human Brain Model", Science 340(6139):1472–1475, 2013.
The data are released under CC BY-NC-SA 4.0. The planes file and every SVG made from it are adapted material under the same licence:
give credit, non-commercial use only, share under the same licence.

holiAtlas v1.0: Manjón J. V. et al., "Ultra-high resolution multimodal MRI densely labelled holistic structural brain atlas",
Scientific Reports 16:9457, 2026, https://doi.org/10.1038/s41598-026-40186-2. Preprint: arXiv:2501.16879.
The sources do not agree on the licence:

- The zip has no licence file.
- The journal article has no usage notes. It says only that the data are "under a Creative Commons license".
- The first version of the preprint, arXiv:2501.16879v1 (Usage Notes), states CC BY-NC-SA 4.0.
  The later versions of the preprint do not name this licence.
- The Zenodo record https://doi.org/10.5281/zenodo.15690524 holds the same zip under CC BY 4.0
  (`atlas_v1.0.zip`, MD5 `4869902cb6e6082820109b0aae3dd16c`).

medfig keeps CC BY-NC-SA 4.0, the stricter of the two statements, for the planes file and every figure made from it:
give credit, non-commercial use only, share under the same licence.
Credit line: "holiAtlas (Manjón et al., Sci Rep 2026), CC BY-NC-SA 4.0".
The CT and PET figures also use holiAtlas: the marker positions (both), the brain outline and the reference region (PET).

Clinical Toolbox CT template: Rorden C. et al., "Age-specific CT and MRI templates for spatial normalization",
NeuroImage 61(4):957–965, 2012, https://doi.org/10.1016/j.neuroimage.2012.03.020. The sources state two licences:

- The repository https://github.com/neurolabusc/Clinical has a BSD 2-Clause licence (copyright 2021 Chris Rorden).
  `LICENSE-clinical-ct.txt` is a copy of it; its second condition asks redistributions to reproduce the notice.
- The 2012 article says that the scripts and images are released under the GNU General Public License, version 2 or later.

Nobody was asked. Both licences allow redistribution. The user chose (2026-10-14) to use the template and to note both
statements. medfig shares the CT planes file under the BSD 2-Clause licence of the repository it came from, and the CT
figures, which also carry holiAtlas marker positions, under CC BY-NC-SA 4.0.
Credit line: "Clinical Toolbox CT template (Rorden et al., NeuroImage 2012), BSD 2-Clause".

rPOP florbetaben templates: Iaccarino L. et al., "rPOP: Robust PET-only processing of community acquired heterogeneous
amyloid-PET data", NeuroImage 246:118775, 2022, https://doi.org/10.1016/j.neuroimage.2021.118775.
The repository https://github.com/LeoIacca/rPOP states CC BY-NC-SA 4.0 and "academic/research purposes" with no warranty.
The templates are means of PET scans of the ADNI study (Alzheimer's Disease Neuroimaging Initiative). No document was found
that says whether ADNI agreed to this redistribution of means of its scans; the user accepted this open point (2026-10-14).
Do not use the PET figures commercially; ask the rPOP authors before a use outside research.
medfig shares the PET planes file and the PET figures under CC BY-NC-SA 4.0.
Credit line: "rPOP florbetaben templates (Iaccarino et al., NeuroImage 2022), CC BY-NC-SA 4.0".

Licence text of CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/

## The full volumes

The full volumes are not in this repository.

BigBrain (1.1 GB): the BigBrain release lists it under
`BigBrainRelease.2015/3D_Volumes/MNI-ICBM152_Space/` on ftp.bigbrainproject.org (`.mnc`; the `.nii.gz` copy used here came from the
same release). `medfig extract` reads the `.nii.gz` (gzip NIfTI-1, uint8) only.

CT template (14 MB) and rPOP templates (1.7 MB each): the repository files named above. Keep them in `~/data/clinical_ct/`
and `~/data/rpop/` (the defaults of `make ct-extract` and `make pet-extract`).

holiAtlas (zip 186 MB): https://volbrain.net/public/data/holiatlas_v1.0.zip (`source.url` in `config/mri.json`).
The Zenodo record https://doi.org/10.5281/zenodo.15690524 holds the same file as `atlas_v1.0.zip`. Unzip it into one directory.
`medfig mri extract` reads the three volumes named in `source.volumes`: `T1w_average.nii.gz`, `tissue_atlas.nii.gz` and
`structure_atlas.nii.gz` (gzip NIfTI-1, 362 x 434 x 362 voxels, 0.5 mm).

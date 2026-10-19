# CLAUDE.md: medfig

Read this file before you change medfig. Read `report/report.pdf` for the method.

## What medfig does

medfig makes flat figures of coronal brain sections and an oblique 3D stack of them for a web page.
It does four modalities: WSI (histology) from BigBrain, MRI (the skull-stripped brain) from the holiAtlas template,
CT (bone ring and brain, no scalp) from the Clinical Toolbox CT template, and PET (amyloid, florbetaben, an amyloid-negative
and an amyloid-positive template) from rPOP.
The machine is a DGX Spark (aarch64). The pipeline uses the CPU only.

## Scope

The goal is figures for four modalities: WSI, MRI, PET and CT. Version 0.3.0 does all four.
The product for MRI, CT and PET is the flat SVG slice. The oblique stack page is a WSI idea (glass slides, slide level and
case level). The user said (2026-10-14) that it is "not the same thing" for MRI, PET and CT: do not develop the stack page
for the scanner modalities. They get it only by reuse of the shared code.

## Rules

1. Keep every parameter in the configuration file of its modality: `config/sections.json` (WSI), `config/mri.json` (MRI),
   `config/ct.json` (CT), `config/pet.json` (PET). Do not put numbers in the code.
2. Keep every report number in `report/numbers.tex`. `python -m medfig report` writes it. Do not type numbers into `report.tex`.
   This rule covers the numbers of every modality.
3. Run `make test` before each commit. All checks must pass.
4. If you change a raster or vector parameter that moves an outline, the reference check fails.
   Update `tests/reference/` (WSI) or `tests/reference/<modality>/` (MRI, CT, PET) in the same commit and say why.
   PET draws the MRI brain mask, and CT and PET take their markers from the MRI planes file: a change of an MRI raster
   parameter or of the MRI planes file moves the PET outline too.
   The checks do not see every parameter. For the WSI sections the check compares path data only.
   It does not see a change of `outline_width` or of a fill colour. For the MRI slices the check also compares the rest of the SVG text.
   The reference checks do not see the marker size (`block_units`, `roi_units`) or `roi_min_px`.
   The checks of `tests/test_report.py` see them: they compare a fresh build with `build/fig/*_numbers.json` and `report/numbers.tex`.
   After such a change, run the steps of the modality (`make mri`, `make ct`, `make pet`) and `python -m medfig report`,
   and commit the new files with it.
   A change of `medfig/stack.py` or of the WSI `stack_3d` values needs new copies of `tests/reference/stack.*`.
   After a change that no check sees, look at the figure.
5. Keep the credit line of the source in every SVG and every page: BigBrain for WSI, holiAtlas for MRI, the Clinical
   Toolbox CT template for CT, the rPOP templates for PET. The CT and PET pages also credit holiAtlas (markers; PET outline
   and reference region). medfig shares the figures under CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/).
   BigBrain is released under this licence. The holiAtlas sources do not agree: the first preprint version states
   CC BY-NC-SA 4.0, the Zenodo record states CC BY 4.0, and the journal article names no licence.
   medfig keeps the stricter one. The CT template is BSD 2-Clause in its repository and GPL v2 or later in its 2012 article;
   keep `data/LICENSE-clinical-ct.txt`. The rPOP templates are CC BY-NC-SA 4.0 in their repository, but they are means of
   ADNI scans and no clearance for their redistribution was found; the user accepted this (2026-10-14). Research use only.
   `data/README.md` has the details. Verify a licence at its primary source before you state it.
6. Do not add BU slides, BU stems or patient data to this repository.
7. Do not say that a figure shows the BU cutting plan, BU block locations, a patient or a measurement. The figures are a schematic.
8. Do not commit the full volumes: BigBrain (`*.nii.gz`, `*.mnc`, 1.1 GB), the holiAtlas template
   (`holiatlas_v1.0.zip` and its `*.nii.gz` files, 186 MB), the CT template (`scct_unsmooth.nii`) and the rPOP templates
   (`Template_*.nii`). Keep them outside the repository: `~/data/holiatlas`, `~/data/clinical_ct`, `~/data/rpop`.
9. A new modality gets its own subpackage `medfig/<modality>/`, its own configuration file in `config/`
   and its own folder `out/<modality>/`. Shared code is in `medfig/vector.py` (outlines), `medfig/stack.py` (3D stack page),
   `medfig/nifti.py` (small NIfTI reader) and `medfig/scan.py` (the scanner modalities MRI, CT, PET: orientation, grid,
   frame, markers, SVG text, thumbnails, page, captures). The WSI modules stay at the top level of `medfig/` and write to `out/`.
   Every modality uses `contours()`, `smooth_contour()` and `path_d()` of `vector.py`, and `stack.py`.
   MRI, CT and PET also use `vector.round_mask()`, `nifti.py` and `scan.py`. CT and PET read the MRI planes file
   (`medfig.mri.slices.load_planes`, `plane`), so they share its grid, its markers and (PET) its brain mask.
   PET has two series (`config/pet.json` `series`), each written like one modality to `out/pet/<series>/`.
   A grid is a 4-tuple (x of column 0, z of row 0, pixel size, x step); the x step is negative in the radiological orientation.
10. The MRI figure is the skull-stripped brain, because this view fits a clinical figure. Do not draw the scalp or the skull.
    Classes come from the labels of the template, not from intensity thresholds.
11. The CT figure keeps the skull: a white bone ring and the brain on a dark screen, no scalp (the user's choice, 2026-10-14;
    a skull-stripped CT does not read as CT). Its classes come from Hounsfield-unit thresholds (`hu` in `config/ct.json`).
    Ask the user before you draw the scalp for any modality.
12. The PET figure is the rPOP florbetaben pair (amyloid-negative and amyloid-positive), warm palette, SUVR bands inside the
    MRI brain mask (the user's choice, 2026-10-14). Do not simulate PET from labels: it looked wrong.
13. MRI, CT and PET are drawn in the radiological orientation (`orientation` in their configuration; the user's choice,
    2026-10-14): the left of the patient is on the right of the image. WSI keeps the left on the left. Keep every scanner
    modality on the same planes (y = +8, -22, -64 mm) and at 1 unit = 0.3 mm, so the figures pair.

## Setup

1. Run `scripts/dgx_setup.sh`. It does not use sudo. It works on the lab DGX Spark if the Chromium system libraries are present.
2. If `pip install` fails for a package on aarch64, report the package name and the error. Do not change the package.
3. If the script prints "Chromium does not start", system libraries are missing.
   `python -m playwright install --with-deps chromium` installs them. That command needs sudo.
4. If `python -m playwright install chromium` fails, set `MEDFIG_CHROMIUM` to a Chromium binary.
   Then run `make render`, `make mri-render`, `make ct-render` and `make pet-render` again.
5. The lab DGX Spark has no system LaTeX. For the report, install `latexmk` and TeX Live (see `scripts/dgx_setup.sh`).
   This needs sudo. Without LaTeX, `make report` fails, and `make all` fails at its last step.
   On the lab DGX Spark a TeX Live of another project works read-only: put
   `~/JimmyProj/generative-media/manim-lab/texlive/bin/aarch64-linux` and `~/JimmyProj/generative-media/manim-lab/mamba/envs/manim/bin`
   (latexmk) on PATH and set `TEXMFVAR` and `TEXMFCONFIG` to a temporary directory.

## Run

- `make all` runs every step: WSI, MRI, CT, PET, then the PDF.
- `make mri` runs every MRI step from the planes file: `mri-slices`, `mri-icons`, `mri-web`, `mri-figures`, `mri-render`.
  Each of them is a target.
- `python -m medfig mri` does the same as `make mri`. `python -m medfig mri slices` runs one step.
- `make ct` and `make pet` do the same for CT and PET (`python -m medfig ct`, `python -m medfig pet`). They read the MRI planes file.
- `make gif` writes `build/stack.gif`. `make mri-gif`, `make ct-gif` and `make pet-gif` write the GIFs of the other modalities.
- `make extract VOLUME=/path/full8_100um_2009b_sym.nii.gz` cuts new WSI planes. Use it only for new `y` values.
- `make mri-extract HOLIATLAS=/path/holiatlas` cuts new MRI planes from the unzipped template. Use it only for new `y` values.
- `make ct-extract` (CT_DIR, default `~/data/clinical_ct`) and `make pet-extract` (PET_DIR, default `~/data/rpop`, and HOLIATLAS)
  cut new CT and PET planes. Run them after a change of `hu` (CT) or `reference` (PET): the planes files depend on them,
  and `ct slices` stops if the thresholds of the planes file differ from the configuration.
- WARNING: `make clean` removes `out/` and `build/`. `build/fig/` holds tracked report figures.
  After `make clean`, run `make all` or restore them with `git checkout -- build/fig`.
  Until then, the checks of `tests/test_report.py` fail, because they read `build/fig/*numbers.json`.

## Open items

- Block centres come from general MNI anatomy. Check them against an atlas (for example Harvard–Oxford) before you use a figure that names a structure.
- At 390 px width the three sections are crowded. A phone layout with two sections is one option.
- The hand-set WSI block centres differ from the holiAtlas centroids at the same `y`. The values are (x, z) in mm, WSI first:
  putamen (-24, 4) vs (-22.2, -1.7), hippocampus (-27, -14) vs (-28.2, -16.4), cerebellar hemisphere (-26, -40) vs (-24.3, -37.0).
- The MRI stack uses the same three planes as WSI (`y` = +8, -22, -64 mm).
  The planes file holds 22 candidate planes for other choices (`source.candidate_y_mm` in `config/mri.json`).
- The bounds of `test_classes_are_sane` are set for planes through the middle of the brain.
  The check fails for a slice at 10 of the 22 candidate planes. Check a new slice by eye. The reasons:
  - `y` = +64, +56, +48, +40 and +32 mm, and `y` = -88 and -96 mm: the brain section is small.
    The mask covers less than a quarter of the image. `test_units` also fails there (area bound).
  - `y` = +24 mm: the rounded mask is three objects. At -88 and -96 mm it is also more than one object.
  - `y` = -40 and -48 mm: the brainstem reaches the bottom of the template field of view.
    The mask touches the image border and is cut flat there.
- In the second state of the stack (case level, Volume), only item 1 shows its dashed marker square.
  The back items show their number only. This is the behaviour of the published v0.1.0 page.
  The cause is in `medfig/stack.py`: the rules `.is-case .mk` and `.cs2 .mk, ...` have equal specificity, and the later rule hides the squares.
  The two buttons of the page show no keyboard focus ring (`all:unset` in the page template).
  Offered both fixes on 2026-10-14, the user chose neither, so the published WSI page does not change.
- On the MRI, CT and PET pages the markers (left structures) are on the image right in the radiological orientation. The stack
  fans the back slices out to the left, so in Volume their number badges are hidden behind the front slice. Not to be fixed:
  the stack page is not developed for the scanner modalities (see Scope).
- CT: the template scans were contrast-enhanced, and the volume ends at z = -71 mm, so the CT figure is cut flat there.
  Its intracranial mask is larger than the holiAtlas intracranial volume (1988 ml for the whole mask; Dice 0.95 to 0.97 at
  the three planes).
- The media query of the stack is `max-width:900px`, so a page of exactly 900 px uses the narrow-width values.
  The frame sheets, the end states and the GIFs are captured at 900 px. No step captures the wide layout.

# medfig

Flat coronal brain figures, and an oblique 3D stack of three of them for a web page. medfig does four modalities:

- WSI (histology): sections drawn from BigBrain.
- MRI: slices of the skull-stripped brain drawn from the holiAtlas template.
- CT: slices of the head (white bone ring and brain on a dark screen, no scalp) drawn from the Clinical Toolbox CT template.
- PET: amyloid PET slices (stepped uptake bands inside the MRI brain outline), an amyloid-negative and an amyloid-positive
  florbetaben template from rPOP.

MRI, CT and PET are drawn in the radiological orientation (the left of the patient on the right of the image), on the same
three planes (y = +8, -22, -64 mm) and at the same scale as WSI (1 unit = 0.3 mm). WSI keeps the left on the left.

Method, figures and parameters: `report/report.pdf`.

![WSI pipeline](build/fig/pipeline_overview.png)

## Quick start

```bash
scripts/dgx_setup.sh          # venv, packages, Chromium, data checksums, tests
. .venv/bin/activate
make all                      # WSI, MRI, CT, PET, then report/report.pdf (the PDF needs LaTeX)
open out/stack.html           # WSI: Slide-level / Case-level switch (or serve out/ and open stack.html)
```

One modality only:

```bash
make mri                      # out/mri/ (SVG, HTML) and build/fig/mri_* (figures)
make ct                       # out/ct/ and build/fig/ct_*   (reads the MRI planes file too)
make pet                      # out/pet/pos/, out/pet/neg/ and build/fig/pet_*   (reads the MRI planes file too)
```

The flat SVG slices are the product: `out/mri/slice_k.svg`, `out/ct/slice_k.svg`, `out/pet/<series>/slice_k.svg` with
`out/pet/<series>/key.svg`. The scanner modalities also get the stack page of WSI (`stack.html`), by reuse of its code; it
was designed for glass slides, not for scans.

## WSI targets

| Target | Product |
|---|---|
| `make sections` | `out/section_{1,2,3}.svg`, `out/sections.json` |
| `make icons` | `out/slide_one.svg`, `out/slide_strip.svg` |
| `make web` | `out/stack.html`, `out/stack.css`, `out/stack_snippet.html` |
| `make figures render overview` | `build/fig/*.png`, `build/frames/` |
| `make gif` | `build/stack.gif` |
| `make extract VOLUME=...` | a new planes file from the full BigBrain volume |
| `make site SITE=dir` | copy the SVGs into a host site and print the markup and CSS |

## MRI targets

| Target | Product |
|---|---|
| `make mri` | every MRI step from the planes file: slices, icons, web, figures, render |
| `make mri-slices` | `out/mri/slice_{1,2,3}.svg`, `out/mri/slices.json` |
| `make mri-icons` | `out/mri/scan_one.svg`, `out/mri/scan_strip.svg` |
| `make mri-web` | `out/mri/stack.html`, `out/mri/stack.css`, `out/mri/stack_snippet.html` |
| `make mri-figures mri-render` | `build/fig/mri_*.png`, `build/fig/mri_numbers.json`, `build/frames/mri/` |
| `make mri-gif` | `build/mri_stack.gif` |
| `make mri-extract HOLIATLAS=dir` | a new planes file from the unzipped holiAtlas template |

The MRI figure is the skull-stripped brain. Scalp and skull are not drawn. This view fits a clinical figure better than the whole head.
The tissue classes come from the label maps of the template, not from intensity thresholds.

## CT and PET targets

| Target | Product |
|---|---|
| `make ct` / `make pet` | every step from the planes file: slices, icons, web, figures, render |
| `make ct-slices` | `out/ct/slice_{1,2,3}.svg`, `out/ct/slices.json` |
| `make pet-slices` | `out/pet/{pos,neg}/slice_{1,2,3}.svg`, `slices.json`, `key.svg` |
| `make ct-figures ct-render` | `build/fig/ct_*.png`, `build/fig/ct_numbers.json` |
| `make pet-figures pet-render` | `build/fig/pet_*.png`, `build/fig/pet_numbers.json` |
| `make ct-extract CT_DIR=dir` | a new CT planes file from `scct_unsmooth.nii` (default dir `~/data/clinical_ct`) |
| `make pet-extract PET_DIR=dir HOLIATLAS=dir` | a new PET planes file from the rPOP templates (default `~/data/rpop`) |

The CT figure keeps the skull, because a CT without bone does not read as CT: bone, intracranial space and CSF come from
Hounsfield-unit thresholds; the scalp is not drawn. The PET figure draws the uptake as nested bands of SUVR (reference: the
whole cerebellum of holiAtlas) inside the MRI brain mask, so a PET slice and an MRI slice of the same y share the outline.
The markers of MRI, CT and PET are centroids of holiAtlas structure labels.

## All modalities

| Target | Product |
|---|---|
| `make all` | the WSI steps (sections, icons, web, figures, render, overview), then `make mri ct pet`, then `make report` |
| `make report` | `report/report.pdf` |
| `make test` | the checks, including a match of the WSI sections with the SVGs on the lab landing page |

Each step is also a command: `python -m medfig sections`, `python -m medfig mri slices`, `python -m medfig ct slices`.
`python -m medfig --help` lists them.
`make report` needs the outputs of every modality. If one is missing, it stops and names the file and the command to run.

## Inputs

- `config/sections.json` (WSI): every parameter (planes, blocks, thresholds, smoothing, palette, 3D stack).
- `config/mri.json`, `config/ct.json`, `config/pet.json`: every parameter of MRI, CT and PET (orientation, planes, slices,
  labels or thresholds, bands, smoothing, palette, 3D stack).
- `data/bigbrain_coronal_100um_sel.npz`: 12 coronal planes cut from BigBrain (10 MB).
- `data/holiatlas_coronal_500um_sel.npz`: 22 coronal planes cut from holiAtlas (T1w, tissue labels, structure labels; 2.6 MB).
- `data/clinical_ct_coronal_500um_sel.npz`: the same 22 planes of the CT template (HU, intracranial mask; 1.9 MB).
- `data/rpop_fbb_coronal_2mm_sel.npz`: the same 22 planes of the two rPOP florbetaben templates (1.0 MB).

See `data/README.md` for the keys of the files, the licences and the full volumes.

## Outputs

- `out/`: the WSI products. `out/mri/`, `out/ct/`, `out/pet/pos/`, `out/pet/neg/`: the products of the other modalities.
- `build/fig/`: the figures of the report. The files of the other modalities have the prefix `mri_`, `ct_` or `pet_`.

## Licence

The WSI figures are adapted from BigBrain (Amunts et al., Science 2013).
The MRI figures are adapted from holiAtlas (Manjón et al., Sci Rep 2026).
The CT figures are adapted from the Clinical Toolbox CT template (Rorden et al., NeuroImage 2012).
The PET figures are adapted from the rPOP florbetaben templates (Iaccarino et al., NeuroImage 2022).
The CT and PET figures also use holiAtlas: marker positions, and for PET the brain outline and the reference region.
medfig shares the figures under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
Keep the credit line of the source:

- WSI: "Brain sections drawn from BigBrain (Amunts et al., Science 2013), CC BY-NC-SA 4.0"
- MRI: "Brain slices drawn from holiAtlas (Manjón et al., Sci Rep 2026), CC BY-NC-SA 4.0"
- CT: "Clinical Toolbox CT template (Rorden et al., NeuroImage 2012), BSD 2-Clause"; markers: holiAtlas, as above
- PET: "rPOP florbetaben templates (Iaccarino et al., NeuroImage 2022), CC BY-NC-SA 4.0"; outline, reference region and
  markers: holiAtlas, as above

`build/fig/mri_end_slice.png` and `build/fig/mri_end_volume.png` hold the stack only, without text. Put the credit line beside them.

BigBrain is released under CC BY-NC-SA 4.0.
The holiAtlas sources do not agree on the licence. The zip has no licence file.
The journal article says only "a Creative Commons license".
The first version of the preprint (arXiv:2501.16879v1, Usage Notes) states CC BY-NC-SA 4.0.
The Zenodo record holds the same zip under CC BY 4.0.
medfig keeps CC BY-NC-SA 4.0, the stricter of the two statements. See `data/README.md`.

The CT template is stated two ways: BSD 2-Clause in its repository (copy in `data/LICENSE-clinical-ct.txt`), GPL version 2
or later in its 2012 article. The rPOP templates are CC BY-NC-SA 4.0 in their repository; they are means of ADNI scans, and
no document was found that clears this redistribution. Use the PET figures for research only. See `data/README.md`.

holiAtlas sources:

- Article: https://doi.org/10.1038/s41598-026-40186-2 (Scientific Reports 16:9457, 2026)
- Zenodo record: https://doi.org/10.5281/zenodo.15690524
- Download: https://volbrain.net/public/data/holiatlas_v1.0.zip

The figures are a schematic. The WSI sections and blocks are not the BU cutting plan or BU block locations.
The MRI, CT and PET slices and markers come from templates (group means). They do not show a patient or a measurement.

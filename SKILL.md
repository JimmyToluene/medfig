---
name: medfig
description: Flat, publication-style schematic figures of coronal brain sections from open data in four modalities - histology (BigBrain), skull-stripped MRI (holiAtlas), CT with its bone ring (Clinical Toolbox CT template) and amyloid PET (rPOP florbetaben, negative and positive) - on the same MNI planes (y = +8, -22, -64 mm) and at one scale (1 unit = 0.3 mm), written as SVG with the source's credit line. Use it to make, restyle or extend these brain-section figures, to put modalities side by side, or to rebuild the medfig pipeline, its tests and its methods report.
---

# medfig

Read `CLAUDE.md` before you change anything: it holds the project's rules (every parameter in `config/*.json`, every
report number in `report/numbers.tex`, tests before each commit, credit lines, licences, and the user's decisions on
orientation and style). The method, every step and every parameter: `report/report.pdf`.

## Run

```bash
scripts/dgx_setup.sh            # or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
. .venv/bin/activate
make sections mri ct pet        # flat SVGs in out/ from the planes files in data/
make test                       # all checks must pass
make help                       # every target
```

The product is the flat SVG slice, k = 1, 2, 3 for y = +8, -22, -64 mm:
`out/section_k.svg` (histology), `out/mri/slice_k.svg`, `out/ct/slice_k.svg`, `out/pet/<neg|pos>/slice_k.svg` with
`out/pet/<neg|pos>/key.svg`. Report figures (`build/fig/`) and rendered frames need Playwright with Chromium; the PDF
needs LaTeX (`make report`).

## Layout

- `medfig/`: the pipeline. Shared code: `vector.py` (outlines), `scan.py` (MRI, CT, PET), `nifti.py`, `stack.py`;
  one subpackage per scanner modality (`mri/`, `ct/`, `pet/`); histology modules at the top level.
- `config/`: one JSON file per modality with every parameter.
- `data/`: the extracted planes files (small). The full volumes are not included; `data/README.md` says where to get
  them, with their licences and checksums.
- `out/`: the current SVG outputs. `report/`: the methods report (LaTeX and PDF). `tests/`: checks and reference SVGs.

## Rules that matter most

- The figures are a schematic: never present one as a patient, a measurement, or a cutting plan or block location.
- Keep the source's credit line in every SVG and page. The figures are shared under CC BY-NC-SA 4.0; the rPOP PET
  templates are for research use only (`data/README.md`).
- MRI, CT and PET are drawn in the radiological orientation (the patient's left on the image right); histology keeps
  the left on the left. Keep every modality on the same planes and scale so the figures pair.
- MRI is the skull-stripped brain, CT keeps the bone ring without the scalp, PET is the rPOP florbetaben pair in a warm
  palette. Ask the user before drawing the scalp for any modality.

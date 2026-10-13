# CLAUDE.md: medfig

Read this file before you change medfig. Read `report/report.pdf` for the method.

## What medfig does

medfig makes flat figures of coronal brain sections from BigBrain and an oblique 3D stack of them for a web page.
The machine is a DGX Spark (aarch64). The pipeline uses the CPU only.

## Rules

1. Keep every parameter in `config/sections.json`. Do not put numbers in the code.
2. Keep every report number in `report/numbers.tex`. `python -m medfig report` writes it. Do not type numbers into `report.tex`.
3. Run `make test` before each commit. All checks must pass.
4. If you change a raster or vector parameter on purpose, the reference check fails. Update `tests/reference/` in the same commit and say why.
5. Keep the BigBrain credit line in every SVG and every page. The licence is CC BY-NC-SA 4.0.
6. Do not add BU slides, BU stems or patient data to this repository.
7. Do not say that a figure shows the BU cutting plan or BU block locations. The figures are a schematic.
8. Do not commit the full BigBrain volume (`*.nii.gz`, `*.mnc`). It is 1.1 GB.

## Setup

1. Run `scripts/dgx_setup.sh`.
2. If `pip install` fails for a package on aarch64, report the package name and the error. Do not change the package.
3. If `python -m playwright install chromium` fails, set `MEDFIG_CHROMIUM` to a Chromium binary and run `make render` again.
4. For the report, install `latexmk` and TeX Live (see `scripts/dgx_setup.sh`).

## Run

- `make all` runs every step from the planes file to the PDF.
- `make extract VOLUME=/path/full8_100um_2009b_sym.nii.gz` cuts new planes. Use it only for new `y` values.

## Open items

- Block centres come from general MNI anatomy. Check them against an atlas (for example Harvard–Oxford) before you use a figure that names a structure.
- At 390 px width the three sections are crowded. A phone layout with two sections is one option.

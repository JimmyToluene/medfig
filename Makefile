# medfig. Run `make help`. Every target is a Python module of medfig/ (WSI) or medfig/<modality>/ (MRI, CT, PET); see report/report.pdf for the method.
PY ?= python3
VOLUME ?= $(HOME)/data/bigbrain/full8_100um_2009b_sym.nii.gz
HOLIATLAS ?= $(HOME)/data/holiatlas
CT_DIR ?= $(HOME)/data/clinical_ct
PET_DIR ?= $(HOME)/data/rpop

# The steps of `all`, `mri`, `ct` and `pet` run in the order written. Each step reads the files of the step before it, so -j must not reorder them.
.NOTPARALLEL:
.PHONY: help all extract sections icons web figures render gif overview report test site clean \
        mri mri-extract mri-slices mri-icons mri-web mri-figures mri-render mri-gif \
        ct ct-extract ct-slices ct-icons ct-web ct-figures ct-render ct-gif \
        pet pet-extract pet-slices pet-icons pet-web pet-figures pet-render pet-gif
help:            ## list the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'
all: sections icons web figures render overview mri ct pet report   ## everything: WSI, MRI, CT, PET, then the PDF
extract:         ## S1: planes file from the full volume (VOLUME=path/to/full8_100um_2009b_sym.nii.gz)
	$(PY) -m medfig extract $(VOLUME)
sections:        ## S2-S7: out/section_*.svg, out/sections.json
	$(PY) -m medfig sections
icons:           ## S8: out/slide_one.svg, out/slide_strip.svg
	$(PY) -m medfig icons
web:             ## S9: out/stack.html, stack.css, stack_snippet.html
	$(PY) -m medfig web
figures:         ## report figures S1-S6: build/fig/
	$(PY) -m medfig figures
render:          ## frames, frame sheet, end states, figures S7-S8 (needs Playwright + Chromium)
	$(PY) -m medfig render
gif:             ## build/stack.gif
	$(PY) -m medfig.render --gif
overview:        ## build/fig/pipeline_overview.png
	$(PY) -m medfig overview
mri: mri-slices mri-icons mri-web mri-figures mri-render   ## MRI: every step from the planes file (M2-M8)
mri-extract:     ## M1: MRI planes file from the unzipped template (HOLIATLAS=path/to/holiatlas)
	$(PY) -m medfig mri extract $(HOLIATLAS)
mri-slices:      ## M2-M6: out/mri/slice_*.svg, out/mri/slices.json
	$(PY) -m medfig mri slices
mri-icons:       ## M7: out/mri/scan_one.svg, out/mri/scan_strip.svg
	$(PY) -m medfig mri icons
mri-web:         ## M7: out/mri/stack.html, stack.css, stack_snippet.html
	$(PY) -m medfig mri web
mri-figures:     ## M8: MRI report figures, build/fig/mri_step_*.png, build/fig/mri_numbers.json
	$(PY) -m medfig mri figures
mri-render:      ## M8: MRI frames, frame sheet, end states, figures of the SVGs and icons (needs Playwright + Chromium)
	$(PY) -m medfig mri render
mri-gif:         ## build/mri_stack.gif
	$(PY) -m medfig mri render --gif
ct: ct-slices ct-icons ct-web ct-figures ct-render   ## CT: every step from the planes file (C2-C8); needs the MRI planes file
ct-extract:      ## C1: CT planes file from the CT template (CT_DIR=path/to/dir with scct_unsmooth.nii)
	$(PY) -m medfig ct extract $(CT_DIR)
ct-slices:       ## C2-C6: out/ct/slice_*.svg, out/ct/slices.json
	$(PY) -m medfig ct slices
ct-icons:        ## C7: out/ct/scan_one.svg, out/ct/scan_strip.svg
	$(PY) -m medfig ct icons
ct-web:          ## C7: out/ct/stack.html, stack.css, stack_snippet.html
	$(PY) -m medfig ct web
ct-figures:      ## C8: CT report figures, build/fig/ct_step_*.png, build/fig/ct_numbers.json
	$(PY) -m medfig ct figures
ct-render:       ## C8: CT frames, end states, figures of the SVGs and icons (needs Playwright + Chromium)
	$(PY) -m medfig ct render
ct-gif:          ## build/ct_stack.gif
	$(PY) -m medfig ct render --gif
pet: pet-slices pet-icons pet-web pet-figures pet-render   ## PET: every step from the planes file (P2-P8); needs the MRI planes file
pet-extract:     ## P1: PET planes file (PET_DIR=path/to/rPOP templates, HOLIATLAS=path/to/holiatlas)
	$(PY) -m medfig pet extract $(PET_DIR) $(HOLIATLAS)
pet-slices:      ## P2-P6: out/pet/<series>/slice_*.svg, slices.json, key.svg
	$(PY) -m medfig pet slices
pet-icons:       ## P7: out/pet/<series>/scan_one.svg, scan_strip.svg
	$(PY) -m medfig pet icons
pet-web:         ## P7: out/pet/<series>/stack.html, stack.css, stack_snippet.html
	$(PY) -m medfig pet web
pet-figures:     ## P8: PET report figures, build/fig/pet_step_*.png, build/fig/pet_numbers.json
	$(PY) -m medfig pet figures
pet-render:      ## P8: PET frames, end states, figures of the SVG pair and icons (needs Playwright + Chromium)
	$(PY) -m medfig pet render
pet-gif:         ## build/pet_<series>_stack.gif
	$(PY) -m medfig pet render --gif
report:          ## report/report.pdf (needs latexmk and a TeX Live with the packages in report.tex)
	$(PY) -m medfig report
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex >/dev/null && latexmk -c >/dev/null && echo report/report.pdf
test:            ## checks
	$(PY) -m pytest -q
site:            ## copy SVGs into SITE=path/to/site and print the markup and CSS
	$(PY) -m medfig site $(SITE)
clean:           ## remove out/ and build/
	rm -rf out build

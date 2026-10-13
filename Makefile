# medfig. Run `make help`. Every target is a Python module of medfig/; see report/report.pdf for the method.
PY ?= python3
VOLUME ?= $(HOME)/data/bigbrain/full8_100um_2009b_sym.nii.gz

.PHONY: help all extract sections icons web figures render gif overview report test site clean
help:            ## list the targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'
all: sections icons web figures render overview report   ## everything from the planes file to the PDF
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
report:          ## report/report.pdf (needs latexmk and a TeX Live with the packages in report.tex)
	$(PY) -m medfig report
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex >/dev/null && latexmk -c >/dev/null && echo report/report.pdf
test:            ## checks
	$(PY) -m pytest -q
site:            ## copy SVGs into SITE=path/to/site and print the markup and CSS
	$(PY) -m medfig site $(SITE)
clean:           ## remove out/ and build/
	rm -rf out build

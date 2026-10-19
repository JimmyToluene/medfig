"""medfig: flat medical figures of coronal brain sections, and an oblique 3D stack of them. Four modalities: WSI, MRI, CT, PET.

WSI pipeline, from BigBrain (parameters: config/sections.json; modules: medfig/; see report/report.pdf):
  S1 extract    stream coronal planes out of the 100 um volume
  S2-S7 sections  downsample, tissue mask, grey/white split, round and trace, one frame, SVG
  S8 icons      slide icons cut at each block marker
  S9 web/render the 3D stack page, frames, end states, GIF
  figures, overview: the figures of the report

MRI pipeline, the skull-stripped brain from holiAtlas (parameters: config/mri.json; modules: medfig/mri/):
  M1 extract    cut coronal planes (T1w, tissue labels, structure labels) from the 0.5 mm template
  M2-M6 slices  skull stripping and tissue classes from the labels, round and trace, one frame, markers, SVG
  M7 icons/web  scan thumbnails, the 3D stack page with a Slice / Volume switch
  M8 figures, render: the MRI figures of the report, frames of the page

CT pipeline, head CT with the bone ring from the Clinical Toolbox CT template (parameters: config/ct.json; modules: medfig/ct/):
  C1 extract, C2-C6 slices (HU classes, cut at the field of view, dark screen), C7 icons/web, C8 figures, render
PET pipeline, amyloid PET from the rPOP florbetaben templates, two series (parameters: config/pet.json; modules: medfig/pet/):
  P1 extract, P2-P6 slices (SUVR bands inside the MRI brain mask, colour key), P7 icons/web, P8 figures, render
MRI, CT and PET are drawn in the radiological orientation (config key "orientation").

Shared code: vector.py (outlines), stack.py (3D stack page), nifti.py (small NIfTI reader), scan.py (the scanner modalities:
orientation, grid, frame, markers, SVG, thumbnails, page, captures), config.py (paths).
"""
__version__ = "0.3.0"

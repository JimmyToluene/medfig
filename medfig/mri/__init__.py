"""medfig.mri: flat figures of coronal MRI slices of the skull-stripped brain from holiAtlas, and an oblique 3D stack of them.

Pipeline (parameters: config/mri.json):
  M1 extract    cut coronal planes (T1w, tissue labels, structure labels) from the 0.5 mm template
  M2-M6 slices  skull stripping and tissue classes from the labels, round and trace, one frame, markers, SVG
  M7 icons/web  scan thumbnails, the 3D stack page with a Slice / Volume switch
  M8 figures, render: the MRI figures of the report, frames of the page
"""

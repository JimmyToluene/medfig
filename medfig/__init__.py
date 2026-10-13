"""medfig: flat medical figures of coronal brain sections from BigBrain, and an oblique 3D stack of them.

Pipeline (see report/report.pdf):
  S1 extract    stream coronal planes out of the 100 um volume
  S2-S7 sections  downsample, tissue mask, grey/white split, round and trace, one frame, SVG
  S8 icons      slide icons cut at each block marker
  S9 web/render the 3D stack page, frames, end states, GIF
  figures, overview: the figures of the report
"""
__version__ = "0.1.0"

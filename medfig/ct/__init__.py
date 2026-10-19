"""medfig.ct: flat figures of coronal head CT slices (the bone ring and the brain, no scalp), and an oblique 3D stack of them.

Pipeline (parameters: config/ct.json):
  C1 extract    the 3D intracranial mask of the CT template; planes (HU, intracranial mask) on the grid of the MRI planes
  C2-C6 slices  bone, intracranial space and CSF from HU thresholds; skull ring; cut at the field of view; frame,
                markers (holiAtlas centroids, as in MRI), SVG on a dark screen
  C7 icons/web  scan thumbnails, the 3D stack page with a Slice / Volume switch
  C8 figures, render: the CT figures of the report, frames of the page
"""

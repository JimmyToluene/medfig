"""medfig.pet: flat figures of coronal amyloid PET slices (stepped uptake bands inside the MRI brain outline).

Two series: an amyloid-negative and an amyloid-positive florbetaben template (config series); each series is drawn
like one MRI stack, with the same band edges and colours, so the pair shows where the uptake differs.
Pipeline (parameters: config/pet.json):
  P1 extract    native coronal planes of each template; the reference mean (whole cerebellum of holiAtlas) in 3D
  P2-P6 slices  SUVR on the MRI grid, smoothing, nested bands inside the MRI brain mask, frame and markers (as in MRI),
                SVG; the colour key
  P7 icons/web  scan thumbnails and the stack page of each series
  P8 figures, render: the PET figures of the report
"""

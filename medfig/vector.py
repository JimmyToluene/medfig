"""Raster mask -> smooth vector outline.

Every modality (WSI, MRI, CT, PET) uses contours(), smooth_contour() and path_d().
MRI, CT and PET use round_mask(). WSI rounds its masks in sections.round_masks().

round_mask()      blur a mask and cut it at one half: pixel corners become round, one-pixel gaps close.
contours()        marching squares at 0.5 on the padded mask, so contours at the image edge close.
smooth_contour()  circular Gaussian on the points, then Douglas-Peucker.
path_d()          SVG path data of a mask in frame units.

The `v` argument is the "vector" block of a configuration file (contour_sigma, dp_epsilon, min_contour_points).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology


def round_mask(mask, sigma, min_object_px=0):
    """Blur the mask and cut it at one half. Objects of at most min_object_px pixels are dropped."""
    m = ndi.gaussian_filter(mask.astype(float), sigma) > 0.5
    return morphology.remove_small_objects(m, max_size=min_object_px) if min_object_px else m


def smooth_contour(c, v):
    """Closed contour (row, col) -> smoothed, simplified (x, y) polygon, or None."""
    if len(c) < v["min_contour_points"]:
        return None
    x = ndi.gaussian_filter1d(c[:, 1], v["contour_sigma"], mode="wrap")
    y = ndi.gaussian_filter1d(c[:, 0], v["contour_sigma"], mode="wrap")
    p = measure.approximate_polygon(np.c_[x, y], v["dp_epsilon"])
    return p if len(p) >= 4 else None


def contours(mask):
    """Marching squares at 0.5. Padding closes contours at the image edge."""
    return [c - 1 for c in measure.find_contours(np.pad(mask, 1).astype(float), 0.5)]


def path_d(mask, ox, oy, v, scale=1.0):
    """SVG path data of the mask. (ox, oy) is the frame origin in pixels; scale is frame units per pixel."""
    out = []
    for c in contours(mask):
        p = smooth_contour(c, v)
        if p is None:
            continue
        out.append("M" + " L".join(f"{(x - ox) * scale:.1f} {(y - oy) * scale:.1f}" for x, y in p) + "Z")
    return "".join(out)

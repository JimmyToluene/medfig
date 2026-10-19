"""Checks of medfig/scan.py, the code that the scanner modalities (MRI, CT, PET) share: orientation, grid, frame, markers.

Run: make test (or python -m pytest -q). Every check runs on small synthetic arrays. A plane is stored (z, x) with z and
x ascending, as in the planes files. In the radiological orientation the left of the patient (x < 0) is on the right of
the image; in the neurological orientation it is on the left. In both, rows run superior to inferior.
"""
import json
from xml.etree import ElementTree

import numpy as np
import pytest

from medfig import config, scan

RADIO, NEURO = {"orientation": "radiological"}, {"orientation": "neurological"}
META = {"origin_mm": [-1.0, -2.0, -1.5], "dims": [6, 9, 4], "vs": 0.5}         # x from -1 to 1.5 mm, z from -1.5 to 0 mm
BIG = {"origin_mm": [-10.0, 0.0, -5.0], "dims": [40, 3, 30], "vs": 0.5}        # x from -10 to 9.5 mm, z from -5 to 9.5 mm
UNIT = 0.3                                                                      # mm per frame unit, as in every configuration


def mm_planes(meta):
    """The x and the z in mm of each pixel of a plane stored (z, x) with z and x ascending, from the file layout."""
    (nx, _, nz), o, vs = meta["dims"], meta["origin_mm"], meta["vs"]
    z, x = np.mgrid[:nz, :nx]
    return o[0] + vs * x, o[2] + vs * z


def test_x_sign():
    assert scan.x_sign(RADIO) == -1 and scan.x_sign(NEURO) == 1
    assert set(scan.ORIENTATIONS) == {"radiological", "neurological"}


@pytest.mark.parametrize("value", ["Radiological", "sideways", "", None], ids=["capital letter", "unknown", "empty", "null"])
def test_x_sign_rejects_an_unknown_orientation(value):
    """The message names the configuration file and the two orientations."""
    with pytest.raises(ValueError, match=r"x\.json: orientation .*'neurological', 'radiological'"):
        scan.x_sign({"_path": "x.json", "orientation": value})


def test_x_sign_needs_the_key():
    with pytest.raises(ValueError, match="configuration: orientation None"):
        scan.x_sign({})


def test_every_scanner_configuration_has_one_orientation():
    """The figures of MRI, CT and PET pair only if they share the orientation. Each configuration names a known one."""
    sides = {m: scan.x_sign(config.load(m)) for m in ("mri", "ct", "pet")}
    assert len(set(sides.values())) == 1, f"the scanner modalities do not share one orientation: {sides}"


def test_orient():
    """Rows: highest z first. Columns: highest x first in the radiological orientation, lowest x first in the neurological."""
    a = np.arange(12).reshape(3, 4)
    keep = a.copy()
    assert np.array_equal(scan.orient(a, NEURO), a[::-1]) and np.array_equal(scan.orient(a, RADIO), a[::-1, ::-1])
    assert scan.orient(a, RADIO)[0, 0] == a[-1, -1] and scan.orient(a, NEURO)[0, 0] == a[-1, 0]
    assert np.array_equal(a, keep), "orient() must not change its input"
    m = a > 5
    assert scan.orient(m, RADIO).dtype == bool and np.array_equal(scan.orient(m, RADIO), m[::-1, ::-1])


def test_grid():
    """(x of column 0, z of row 0, pixel size, x step per column) of the tiny grid: x from -1 to 1.5 mm, z from -1.5 to 0 mm."""
    assert scan.grid(META, NEURO) == (-1.0, 0.0, 0.5, 0.5)
    assert scan.grid(META, RADIO) == (1.5, 0.0, 0.5, -0.5)
    with pytest.raises(ValueError, match="orientation"):
        scan.grid(META, {"orientation": "sideways"})


@pytest.mark.parametrize("cfg", [RADIO, NEURO], ids=["radiological", "neurological"])
def test_to_mm_follows_orient(cfg):
    """to_mm(row, column) of the grid is the mm of that pixel after orient(), for every pixel."""
    g = scan.grid(META, cfg)
    xs, zs = (scan.orient(a, cfg) for a in mm_planes(META))
    for r in range(xs.shape[0]):
        for c in range(xs.shape[1]):
            assert scan.to_mm(r, c, g) == pytest.approx((xs[r, c], zs[r, c])), f"pixel ({r}, {c}) of the {cfg['orientation']} plane"


def test_frame():
    """The union bounding box of the masks, padded by pad_units / scale pixels (rounded), in pixels and units."""
    masks = np.zeros((2, 20, 30), bool)
    masks[0, 5:9, 10:13] = True                                       # rows 5 to 8, columns 10 to 12
    masks[1, 7:13, 11:21] = True                                      # rows 7 to 12, columns 11 to 20
    v = {"pad_units": 6}
    assert scan.frame(masks, v, 2.0) == (10 - 3, 5 - 3, (20 - 10 + 6) * 2, (12 - 5 + 6) * 2)
    assert scan.frame(masks, v, 1.0) == (10 - 6, 5 - 6, 20 - 10 + 12, 12 - 5 + 12)
    with pytest.raises(ValueError, match="empty"):
        scan.frame(np.zeros((2, 20, 30), bool), v, 2.0)


@pytest.mark.parametrize("cfg", [RADIO, NEURO], ids=["radiological", "neurological"])
def test_frame_mm_round_trip(cfg):
    """A mask pixel maps to its mm and back to frame units: u = s (x - x_left) / unit, v = (z_top - z) / unit.

    Two pixels: one left of the midline (x = -6.5 mm, z = 1 mm) and one right of it (x = 5 mm, z = 5 mm).
    The left pixel is on the image right in the radiological orientation and on the image left in the neurological.
    """
    s, g = scan.x_sign(cfg), scan.grid(BIG, cfg)
    scale = g[2] / UNIT
    stored = np.zeros((BIG["dims"][2], BIG["dims"][0]), bool)
    want = {(12, 7): (-6.5, 1.0), (20, 30): (5.0, 5.0)}             # (z index, x index) in the file: (x, z) in mm
    for k, i in want:
        stored[k, i] = True
    m = scan.orient(stored, cfg)
    ox, oy, vw, vh = scan.frame([m], {"pad_units": 3}, scale)
    x_left, z_top = scan.frame_mm(ox, oy, g)
    xs, zs = (scan.orient(a, cfg) for a in mm_planes(BIG))
    assert (x_left, z_top) == pytest.approx((xs[oy, ox], zs[oy, ox])), "frame_mm is the mm of the top left frame pixel"
    units = {}
    for r, c in zip(*np.nonzero(m)):
        x, z = scan.to_mm(r, c, g)
        assert (x, z) in [pytest.approx(w) for w in want.values()], f"pixel ({r}, {c}) is at ({x}, {z}) mm, not at a mask pixel"
        u, v = (c - ox) * scale, (r - oy) * scale                       # frame units, as path_d() draws them
        assert u == pytest.approx(s * (x - x_left) / UNIT, abs=1e-6) and v == pytest.approx((z_top - z) / UNIT, abs=1e-6)
        assert 0 < u < vw and 0 < v < vh, "the pixel is outside the frame"
        units[round(x, 1)] = u
    assert s * (units[-6.5] - units[5.0]) < 0, "the left pixel is on the wrong side of the image"
    assert (units[-6.5] > units[5.0]) == (cfg["orientation"] == "radiological")


def test_roi_centre_and_slice_centre():
    """slice_centre() gives None for a slice without roi_labels and names the slice when the labels have too few pixels."""
    st = np.zeros((10, 10), np.uint8)
    st[2:4, 3:8] = 33                                                 # 10 pixels, centroid (2.5, 5)
    r = {"roi_min_px": 10}
    s = {"k": 2, "y_mm": -22, "roi_labels": [33]}
    assert scan.roi_centre(st, [33], r) == (2.5, 5.0) == scan.slice_centre(s, st, r)
    assert scan.slice_centre({"k": 1, "y_mm": 8}, st, r) is None and scan.slice_centre({**s, "roi_labels": []}, st, r) is None
    with pytest.raises(ValueError, match=r"slice 2 \(y = -22 mm\): structure labels \[33\]: 10 px .*roi_min_px"):
        scan.slice_centre(s, st, {"roi_min_px": 11})


@pytest.mark.parametrize("cfg", [RADIO, NEURO], ids=["radiological", "neurological"])
def test_item(cfg):
    """The marker of a slice: roi_mm is the mm of the centre, rect is a square of the given side around it in frame units."""
    g = scan.grid(BIG, cfg)
    scale = g[2] / UNIT
    s = {"k": 1, "y_mm": 8, "label": "a slice", "roi": "left putamen", "roi_labels": [33]}
    it = scan.item(s, (12.0, 7.0), 2, 3, g, scale, 20, "slice_1.svg", {"mask": 1})
    assert it["roi_mm"] == [round(v, 1) for v in scan.to_mm(12.0, 7.0, g)]
    assert it["rect"] == [round((7 - 2) * scale - 10, 1), round((12 - 3) * scale - 10, 1), 20, 20]
    assert {k: it[k] for k in ("k", "y_mm", "label", "roi", "file", "area_mm2")} == {
        "k": 1, "y_mm": 8, "label": "a slice", "roi": "left putamen", "file": "slice_1.svg", "area_mm2": {"mask": 1}}
    none = scan.item(s, None, 2, 3, g, scale, 20, "slice_1.svg", {})
    assert (none["roi"], none["roi_mm"], none["rect"]) == (None, None, None), "a slice without a centre has no marker and no marker name"


def test_svg_is_well_formed_and_carries_the_credit_line():
    body = scan.fill("M0 0 L1 0 L1 1Z", "#123456") + scan.stroke("M0 0 L1 0 L1 1Z", "#654321", 4.4)
    text = scan.svg(10, 20, "A slice, y = 8 mm", "Source (Author 2026), CC BY-NC-SA 4.0", body)
    root = ElementTree.fromstring(text)
    assert root.get("viewBox") == "0 0 10 20" and root.get("aria-label") == "A slice, y = 8 mm" and root.get("role") == "img"
    assert "<!-- Drawn from Source (Author 2026), CC BY-NC-SA 4.0 -->" in text
    fill, stroke = root.findall("{http://www.w3.org/2000/svg}path")
    assert (fill.get("fill"), fill.get("fill-rule")) == ("#123456", "evenodd")
    assert (stroke.get("fill"), stroke.get("stroke"), stroke.get("stroke-width")) == ("none", "#654321", "4.4")


@pytest.mark.parametrize("orientation, side", [("radiological", "the left of the patient is on the right of the image"),
                                               ("neurological", "the left of the patient is on the left of the image")])
def test_web_note_names_the_orientation(tmp_path, orientation, side):
    """The note of the stack page says where the left of the patient is, and credits the source."""
    meta = {"viewBox": [0, 0, 100, 80], "license": "Source (Author 2026), CC BY-NC-SA 4.0",
            "slices": [{"k": 1, "y_mm": 8, "file": "slice_1.svg", "rect": [10, 20, 30, 30]}]}
    (tmp_path / "slices.json").write_text(json.dumps(meta))
    cfg = {**config.load("mri"), "orientation": orientation}
    scan.web(cfg, tmp_path, "A stack", "MRI", "of a test", log=lambda *_: None)
    page = (tmp_path / "stack.html").read_text(encoding="utf-8")
    assert side in page and meta["license"] in page and "y = +8 mm of a test" in page


@pytest.mark.xfail(strict=True, reason=(
    "medfig/scan.py web(), lines 197-198: the orientation in the note comes from the configuration (cfg['orientation']), "
    "while the y values, the credit line and the slices come from slices.json. After a change of `orientation` in the "
    "configuration and a `web` step without a new `slices` step, the page names the new orientation under slices drawn "
    "in the old one, and nothing refuses the stale slices.json."))
def test_web_note_follows_the_slices(tmp_path):
    """The note names the orientation of slices.json (the one the slices were drawn in), or the step refuses a slices.json
    that another orientation wrote."""
    meta = {"viewBox": [0, 0, 100, 80], "orientation": "radiological", "license": "Source (Author 2026), CC BY-NC-SA 4.0",
            "slices": [{"k": 1, "y_mm": 8, "file": "slice_1.svg", "rect": None}]}
    (tmp_path / "slices.json").write_text(json.dumps(meta))
    try:
        scan.web({**config.load("mri"), "orientation": "neurological"}, tmp_path, "A stack", "MRI", "of a test", log=lambda *_: None)
    except ValueError:
        return
    page = (tmp_path / "stack.html").read_text(encoding="utf-8")
    assert "the left of the patient is on the right of the image" in page, "the note names an orientation the slices were not drawn in"

"""Checks of the medfig MRI pipeline (holiAtlas, skull-stripped brain). Run: make test (or python -m pytest -q)."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from xml.etree import ElementTree

import numpy as np
import pytest
from scipy import ndimage as ndi

from medfig import __main__ as cli, config, scan, stack
from medfig.mri import icons as I, slices as M, web as W

CFG = config.load("mri")
NEURO = {**CFG, "orientation": "neurological"}                 # the same configuration, left of the patient on the image left
REF = config.ROOT / "tests" / "reference" / "mri"
SVG = "{http://www.w3.org/2000/svg}"
SVG_PATH = SVG + "path"
# The child process of test_slices_step_in_an_ascii_locale: it prints its default text encoding, sends the output
# of the step to the directory in argv[1], then runs `python -m medfig mri slices` through the command-line function.
ASCII_RUN = ("import locale, pathlib, sys\n"
             "import medfig.mri.slices as M\n"
             "from medfig.__main__ import mri\n"
             "print(locale.getpreferredencoding(False))\n"
             "M.MRI_OUT = pathlib.Path(sys.argv[1])\n"
             "sys.exit(mri(['slices']))\n")


def quiet(*_):
    pass


def path_data(svg):
    return re.findall(r' d="([^"]*)"', svg)


def without_path_data(svg):
    """The SVG text with every d attribute emptied: the viewBox, the label, the credit line, the colours and the stroke."""
    return re.sub(r' d="[^"]*"', ' d=""', svg)


def label_image():
    """A 60 x 60 tissue-label image for classes() with mask_close_px = 2. Each part tests one rule of the brain mask.

    A square of cortex (label 2) with white matter (3) inside it. A ventricle (CSF, label 1), 20 px wide, closed in
    by white matter, with a pocket without a label (0) in it. A sulcus (CSF) and a gap without a label, each 4 px wide
    and open to the outside. Two rows of cortex at the top border of the image.
    """
    t = np.zeros((60, 60), np.uint8)
    t[10:50, 10:50] = 2
    t[14:46, 14:46] = 3
    t[22:42, 16:36] = 1         # ventricle
    t[30:34, 24:28] = 0         # pocket without a label in the ventricle
    t[10:20, 40:44] = 1         # sulcus, open at the top of the square
    t[40:50, 40:44] = 0         # gap without a label, open at the bottom of the square
    t[0:2, 10:50] = 2           # cortex at the image border
    return t


def points(d):
    """The coordinates of SVG path data as an (n, 2) array of (x, y)."""
    return np.array(re.findall(r"-?\d+\.\d", d), float).reshape(-1, 2)


def border(a):
    return np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])


@pytest.fixture(scope="module")
def planes():
    return M.load_planes(CFG)


@pytest.fixture(scope="module")
def cut(planes):
    """For each configured slice: (slice, channels of the plane, masks from the labels, rounded masks)."""
    data, meta = planes
    ps = [M.plane(data, meta, s["y_mm"], CFG) for s in CFG["slices"]]
    cs = [M.classes(p, CFG) for p in ps]
    return [(s, p, c, M.rounded(c, CFG["raster"])) for s, p, c in zip(CFG["slices"], ps, cs)]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One build into a temporary directory: (directory, content of slices.json)."""
    d = tmp_path_factory.mktemp("mri")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(M, "MRI_OUT", d)
        M.build(CFG, log=quiet)
    return d, json.loads((d / "slices.json").read_text())


def test_planes_file(planes):
    data, meta = planes
    with np.load(config.planes_path(CFG)) as z:
        assert {"t1w", "tissue", "structure", "meta"} <= set(z.files)
    assert [data[k].dtype for k in M.CHANNELS] == [np.uint16, np.uint8, np.uint8]
    assert len({a.shape for a in data.values()}) == 1 and data["t1w"].ndim == 3
    n, nz, nx = data["t1w"].shape
    assert n == len(meta["ys"]) == len(meta["js"]) and [nx, nz] == [meta["dims"][0], meta["dims"][2]]
    assert all(meta["origin_mm"][1] + j * meta["vs"] == y for y, j in zip(meta["ys"], meta["js"])), "js are not the voxel rows of ys"
    for s in CFG["slices"]:
        assert s["y_mm"] in meta["ys"], "every configured slice must be in the planes file"
        p = M.plane(data, meta, s["y_mm"], CFG)
        for lab in s.get("roi_labels") or []:
            assert (p["structure"] == lab).any(), f"structure label {lab} is not in the plane y {s['y_mm']}"


def test_plane_rows_run_superior_to_inferior(cut):
    """The top row is air above the head. The head leaves the image at the bottom row (the neck)."""
    for s, p, _, _ in cut:
        t1 = p["t1w"]
        assert t1[0].max() < 0.1 * t1.max() < t1[-1].max(), f"the plane is upside down at y {s['y_mm']}"


def test_plane_columns_follow_the_orientation(planes):
    """holiAtlas labels a left structure with an odd number and its right partner with the next lower even number
    (33 left putamen, 32 right putamen). In the planes file (x ascends) the odd label lies at x < 0, the even one at x > 0.
    In the radiological orientation the left structure is on the image right of the midline column (x = 0 mm), the right
    one on the image left. The neurological plane is the mirror image of the radiological one.
    """
    data, meta = planes
    assert CFG["orientation"] == "radiological"
    nx, vs, x0 = meta["dims"][0], meta["vs"], meta["origin_mm"][0]
    mid = -x0 / vs                                                     # column of x = 0 mm in the file
    for s in CFG["slices"]:
        stored = data["structure"][meta["ys"].index(s["y_mm"])]
        radio, neuro = M.plane(data, meta, s["y_mm"], CFG), M.plane(data, meta, s["y_mm"], NEURO)
        assert all(np.array_equal(neuro[k], radio[k][:, ::-1]) for k in M.CHANNELS), f"the neurological plane is not the mirror at y {s['y_mm']}"
        for left in s["roi_labels"]:
            at = f"label {left} ({s['roi']}) and {left - 1} at y {s['y_mm']}"
            assert left % 2 == 1 and s["roi"].startswith("left"), f"{at}: a left structure has an odd label"
            assert x0 + vs * np.nonzero(stored == left)[1].mean() < 0 < x0 + vs * np.nonzero(stored == left - 1)[1].mean(), (
                f"{at}: the odd label is not at x < 0 and the even label at x > 0 in the planes file")
            for p, centre, name in ((radio, nx - 1 - mid, "right"), (neuro, mid, "left")):
                cl, cr = (np.nonzero(p["structure"] == v)[1].mean() for v in (left, left - 1))
                assert (cl > centre > cr) == (name == "right") and (cl < centre < cr) == (name == "left"), (
                    f"{at}: the left structure is not on the image {name} in the {'radiological' if name == 'right' else 'neurological'} plane")


def test_classes_are_sane(cut):
    for s, _, c, m in cut:
        at = f"at y {s['y_mm']}"
        for d in (c, m):
            assert not (d["wm"] & ~d["brain"]).any() and not (d["brain"] & ~d["mask"]).any(), f"the layers are not nested {at}"
        frac_m, frac_b, frac_w = m["mask"].mean(), m["brain"].sum() / m["mask"].sum(), m["wm"].sum() / m["brain"].sum()
        assert 0.25 < frac_m < 0.55, f"mask fraction {frac_m:.2f} {at}"
        assert 0.85 < frac_b < 1.0, f"brain share of the mask {frac_b:.2f} {at}: the mask must hold CSF (sulci, ventricles)"
        assert 0.2 < frac_w < 0.65, f"white-matter share of the brain {frac_w:.2f} {at}"
        assert ndi.label(m["mask"])[1] == 1, f"the mask is {ndi.label(m['mask'])[1]} objects {at}"
        assert not border(c["mask"]).any() and not border(m["mask"]).any(), f"the mask touches the image border {at}"


def test_mask_leaves_the_head_out(cut):
    """Skull stripping. Scalp fat is brighter than white matter in T1w. These pixels are in the image, outside the mask."""
    for s, p, c, m in cut:
        bright = p["t1w"] > np.median(p["t1w"][c["wm"]])
        share = (bright & ~m["mask"]).sum() / m["mask"].sum()
        assert share > 0.03, f"bright pixels outside the mask are {share:.3f} of the mask area at y {s['y_mm']}"
        assert not (bright & m["mask"] & (p["tissue"] == 0)).any(), f"bright pixels without a tissue label are in the mask at y {s['y_mm']}"


def test_tissue_label_lists():
    tl = CFG["tissue_labels"]
    gm, wm = tl["grey_matter"], tl["white_matter"]
    names = {int(k): v for k, v in tl["names"].items()}
    assert sorted(names) == list(range(1, 10)), "the configuration names the labels 1 to 9"
    assert set(gm + wm) <= set(names), "a label is not in the atlas"
    assert not set(gm) & set(wm), "a label is in both lists"
    assert all("GM" in names[v] for v in gm), "a grey-matter label has no GM in its name"
    assert not any("GM" in names[v] or "CSF" in names[v] for v in wm), "a white-matter label is grey matter or CSF in the atlas"


def test_skull_stripping_is_from_labels(cut):
    """The mask comes from the tissue labels and stays inside the intracranial volume. T1w intensities play no part."""
    tl = CFG["tissue_labels"]
    for s, p, c, _ in cut:
        at = f"at y {s['y_mm']}"
        icv = ndi.binary_fill_holes(p["tissue"] > 0)
        assert np.array_equal(c["mask"], c["mask"] & icv), f"the mask leaves the intracranial volume {at}"
        assert np.array_equal(c["brain"], np.isin(p["tissue"], tl["grey_matter"] + tl["white_matter"])), at
        assert np.array_equal(c["wm"], np.isin(p["tissue"], tl["white_matter"])), at
        only = M.classes({"tissue": p["tissue"]}, CFG)
        assert all(np.array_equal(c[k], only[k]) for k in c), f"the classes use more than the tissue labels {at}"


def test_classes_on_a_synthetic_label_image():
    """The rules of the brain mask on a small image. The three configured slices do not reach every rule."""
    t = label_image()
    c = M.classes({"tissue": t}, {**CFG, "raster": {**CFG["raster"], "mask_close_px": 2}})
    mask = c["mask"]
    assert np.array_equal(c["brain"], np.isin(t, [2, 3])) and np.array_equal(c["wm"], t == 3)
    assert not (c["brain"] & ~mask).any(), "grey and white matter must be inside the mask"
    assert mask[0:2, 10:50].all(), "brain pixels at the image border must stay in the mask (the closing alone drops them)"
    assert mask[22:42, 16:36].all(), "CSF that the brain closes in must be inside the mask, with a pocket without a label in it"
    assert mask[13:20, 40:44].all(), "a sulcus of 4 px must be inside the mask: the closing by 2 px bridges it"
    assert not mask[40:50, 40:44].any(), "a gap without a label that opens to the outside must stay outside the mask"
    assert not (mask & ~ndi.binary_fill_holes(t > 0)).any(), "the mask leaves the filled label mask"


def test_markers(planes, cut, built):
    """Each marker sits on its structure and inside the frame. A left structure has x < 0 mm, and in the radiological
    orientation its marker square lies on the image right of the midline (x = 0 mm). A frame unit u is
    s (x - x_left) / mm_per_unit with s = -1 (radiological) or +1 (neurological)."""
    data, meta = planes
    out = built[1]
    vs, vb, (fx, fz), u = meta["vs"], out["viewBox"], out["frame_mm"], out["mm_per_unit"]
    assert out["orientation"] == CFG["orientation"] == "radiological"
    sign = scan.x_sign(CFG)
    midline = sign * (0 - fx) / u                                     # the x of the midline in frame units
    for (s, p, _, m), it in zip(cut, out["slices"]):
        if not s.get("roi_labels"):
            assert it["rect"] is None and it["roi_mm"] is None
            continue
        kk, ii = np.where(np.isin(data["structure"][meta["ys"].index(s["y_mm"])], s["roi_labels"]))     # as in the file: z ascends
        centroid = [meta["origin_mm"][0] + ii.mean() * vs, meta["origin_mm"][2] + kk.mean() * vs]
        assert it["roi_mm"] == pytest.approx(centroid, abs=0.051), f"{s['roi']}: roi_mm is not the centroid of the labels in MNI mm"
        st = np.isin(p["structure"], s["roi_labels"])
        cr, cc = M.roi_centre(p["structure"], s["roi_labels"], CFG["raster"])
        i, j = int(round(cr)), int(round(cc))
        assert ndi.distance_transform_edt(~st)[i, j] * vs <= 3, f"{s['roi']}: the centroid is more than 3 mm from the structure"
        assert m["mask"][i, j], f"{s['roi']}: the centroid is outside the brain mask"
        x_mm, z_mm = it["roi_mm"]
        assert 0 < sign * (x_mm - fx) / u < vb[2] and 0 < (fz - z_mm) / u < vb[3], f"{s['roi']}: the centre is outside the frame"
        x, y, w, h = it["rect"]
        assert 0 <= x and x + w <= vb[2] and 0 <= y and y + h <= vb[3], f"{s['roi']}: the marker is outside the frame"
        assert 0 < midline < vb[2], "the midline is outside the frame"
        if s["roi"].startswith("left"):
            assert x_mm < 0, f"{s['roi']}: the centre is at x {x_mm} mm, not on the left of the midline"
            assert x + w / 2 > midline, f"{s['roi']}: the marker centre {x + w / 2} is not on the image right of the midline {midline:.1f}"


def test_neurological_build_is_the_mirror(built, tmp_path, monkeypatch):
    """The same configuration with "orientation": "neurological" gives the mirror image of the radiological slices.

    The viewBox and the markers in mm are the same. The frame starts at the other side of the head: its left edge is
    the right edge of the radiological frame, so x_left(neuro) = x_left(radio) - width * mm_per_unit (to the rounding of
    the width to whole units). Each outline and each marker square is mirrored about the middle of the frame.
    """
    d, radio = built
    monkeypatch.setattr(M, "MRI_OUT", tmp_path)
    neuro = M.build(NEURO, log=quiet)
    u, vw = radio["mm_per_unit"], radio["viewBox"][2]
    assert neuro["orientation"] == "neurological" and neuro["viewBox"] == radio["viewBox"]
    assert neuro["frame_mm"][1] == radio["frame_mm"][1] and neuro["frame_mm"][0] == pytest.approx(radio["frame_mm"][0] - vw * u, abs=u)
    assert neuro["frame_mm"][0] < 0 < radio["frame_mm"][0], "the left edge of the frame is on the left of the patient only in the neurological orientation"
    for a, b in zip(radio["slices"], neuro["slices"]):
        assert (a["roi_mm"], a["area_mm2"]) == (b["roi_mm"], b["area_mm2"]), f"slice {a['k']}: the markers in mm or the areas differ"
        x, y, w, h = a["rect"]
        assert b["rect"][1:] == [y, w, h] and b["rect"][0] + w / 2 == pytest.approx(vw - (x + w / 2), abs=1), f"slice {a['k']}: the marker is not mirrored"
        assert b["rect"][0] + w / 2 < -neuro["frame_mm"][0] / u and x + w / 2 > radio["frame_mm"][0] / u, (
            f"slice {a['k']}: the left structure must be on the image left of the midline only in the neurological orientation")
        for pa, pb in zip(path_data((d / a["file"]).read_text(encoding="utf-8")), path_data((tmp_path / b["file"]).read_text(encoding="utf-8"))):
            qa, qb = points(pa), points(pb)
            assert [qb[:, 0].min(), qb[:, 0].max()] == pytest.approx([vw - qa[:, 0].max(), vw - qa[:, 0].min()], abs=1), f"slice {a['k']}: an outline is not mirrored"
            assert [qb[:, 1].min(), qb[:, 1].max()] == pytest.approx([qa[:, 1].min(), qa[:, 1].max()], abs=0.25), f"slice {a['k']}: an outline moved up or down"


def test_roi_centre_needs_the_structure():
    with pytest.raises(ValueError):
        M.roi_centre(np.zeros((20, 20), np.uint8), [33], CFG["raster"])


def test_roi_centre_needs_roi_min_px_pixels():
    """One pixel fewer than roi_min_px stops the step. With roi_min_px pixels the centre is their centroid (row, column)."""
    n = CFG["raster"]["roi_min_px"]
    st = np.zeros((8, n + 8), np.uint8)
    st[3, 2:n + 1] = 33                                               # n - 1 pixels
    st[6, 4] = 34                                                     # another structure
    with pytest.raises(ValueError, match="roi_min_px"):
        M.roi_centre(st, [33], CFG["raster"])
    st[3, n + 1] = 33                                                 # n pixels: columns 2 to n + 1 of row 3
    assert M.roi_centre(st, [33], CFG["raster"]) == (3.0, 2 + (n - 1) / 2)


def test_build_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "MRI_OUT", tmp_path)
    a = M.build(CFG, log=quiet)
    h1 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("slice_*.svg")}
    M.build(CFG, log=quiet)
    h2 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("slice_*.svg")}
    assert h1 == h2 and len(h1) == len(CFG["slices"])
    assert json.loads((tmp_path / "slices.json").read_text()) == a
    assert {"viewBox", "mm_per_unit", "frame_mm", "orientation", "license", "slices"} <= set(a) and a["license"] == CFG["source"]["license"]
    assert a["orientation"] == CFG["orientation"]
    pal = CFG["palette"]
    for s in a["slices"]:
        assert {"k", "y_mm", "label", "roi", "roi_mm", "rect", "file", "area_mm2"} <= set(s)
        svg = (tmp_path / s["file"]).read_text(encoding="utf-8")
        assert svg.count("<path") == 4 and re.search(r'viewBox="0 0 \d+ \d+"', svg)
        assert "holiAtlas" in svg and "CC BY-NC-SA 4.0" in svg and a["license"] in svg
        paths = ElementTree.fromstring(svg).findall(SVG_PATH)
        assert [e.get("fill") for e in paths] == [pal["csf"], pal["grey_matter"], pal["white_matter"], "none"]
        assert paths[3].get("stroke") == pal["outline"] and paths[3].get("d") == paths[0].get("d"), "the outline is the mask"


def test_build_checks_the_configuration_before_it_writes(planes, cut, tmp_path, monkeypatch):
    """A y that is not in the planes file or is not a whole number, roi_labels with too few pixels, label lists that match nothing or an unknown orientation stop the step before the first file. The output directory stays as it was."""
    out = tmp_path / "mri"
    monkeypatch.setattr(M, "MRI_OUT", out)
    first, p = CFG["slices"][0], cut[0][1]
    y = max(planes[1]["ys"]) + 1
    absent = next(v for v in range(255, 0, -1) if not (p["structure"] == v).any())
    with pytest.raises(ValueError, match=rf"y_mm \[{y}\] is not in .*\.npz, which holds y = "):
        M.build({**CFG, "slices": [first, {**first, "k": 2, "y_mm": y}]}, log=quiet)
    with pytest.raises(ValueError, match=r"slice 2 \(y = .* mm\): .*roi_min_px"):
        M.build({**CFG, "slices": [first, {**first, "k": 2, "roi_labels": [absent]}]}, log=quiet)
    with pytest.raises(ValueError, match=r"y_mm \[8\.0\] is not in "):
        M.build({**CFG, "slices": [{**first, "y_mm": 8.0}]}, log=quiet)
    with pytest.raises(ValueError, match="tissue_labels match no pixel"):
        M.build({**CFG, "tissue_labels": {**CFG["tissue_labels"], "grey_matter": [200], "white_matter": [201]},
                 "slices": [{k: v for k, v in first.items() if k != "roi_labels"}]}, log=quiet)
    with pytest.raises(ValueError, match=r"orientation 'left on the left': use one of \['neurological', 'radiological'\]"):
        M.build({**CFG, "orientation": "left on the left"}, log=quiet)
    assert not out.exists(), "a build that failed wrote files"
    meta = M.build({**CFG, "slices": [{k: v for k, v in first.items() if k != "roi_labels"}]}, log=quiet)
    assert [(s["roi"], s["roi_mm"], s["rect"]) for s in meta["slices"]] == [(None, None, None)], "a slice without roi_labels has no marker and no marker name"
    assert sorted(f.name for f in out.iterdir()) == [f"slice_{first['k']}.svg", "slices.json"]


def test_units(planes, cut, built):
    """One unit is 0.3 mm, as in WSI, so the two modalities share a scale. The frame maps MNI mm to units.

    Column c of a radiological plane is the column nx - 1 - c of the planes file (x descends from left to right)."""
    _, meta = planes
    d, out = built
    sign = scan.x_sign(CFG)
    wsi = config.load()
    with np.load(config.planes_path(wsi)) as z:
        wsi_vs = json.loads(str(z["meta"])).get("vs", 0.1)
    u, vs, (fx, fz), vb = out["mm_per_unit"], meta["vs"], out["frame_mm"], out["viewBox"]
    assert u == pytest.approx(wsi["raster"]["downsample"] * wsi_vs) and u == pytest.approx(0.3)
    for (s, _, _, m), it in zip(cut, out["slices"]):
        at = f"at y {s['y_mm']}"
        if it["rect"]:
            x, y, w, h = it["rect"]
            centre = [sign * (it["roi_mm"][0] - fx) / u, (fz - it["roi_mm"][1]) / u]
            assert centre == pytest.approx([x + w / 2, y + h / 2], abs=0.25), f"roi_mm and rect do not agree {at}"
        pts = points(path_data((d / it["file"]).read_text(encoding="utf-8"))[0])
        half = CFG["vector"]["outline_width"] / 2
        assert pts.min() >= half and (pts.max(0) <= np.array(vb[2:]) - half).all(), f"the outline leaves the viewBox {at}"
        rr, cc = np.where(m["mask"])
        z_top = meta["origin_mm"][2] + vs * (m["mask"].shape[0] - 1)
        xs = meta["origin_mm"][0] + vs * (cc if sign > 0 else meta["dims"][0] - 1 - cc)   # x of each mask pixel, from the file layout
        xs_svg = fx + sign * pts[:, 0] * u
        box_svg = [xs_svg.min(), xs_svg.max(), fz - pts[:, 1].max() * u, fz - pts[:, 1].min() * u]
        box_px = [xs.min(), xs.max(), z_top - rr.max() * vs, z_top - rr.min() * vs]
        assert box_svg == pytest.approx(box_px, abs=vs), f"the outline is not at the place of the mask in mm {at}"
        area = it["area_mm2"]
        assert area == {k: round(float(m[k].sum()) * vs * vs) for k in m} and 8000 < area["mask"] < 18000, f"areas {area} {at}"


def test_reference_paths():
    """The SVG path data equal the reference copies in tests/reference/mri.

    The first reference copies were made on 2026-10-13 (left of the patient on the image left). The copies were
    regenerated on 2026-10-14 for the radiological orientation (left of the patient on the image right), on a DGX Spark,
    aarch64. Nothing is published yet.
    """
    refs = sorted(REF.glob("slice_*.svg"))
    if not refs:
        pytest.skip("no reference copies")
    for p in refs:
        out = M.MRI_OUT / p.name
        if not out.exists():
            pytest.skip("run `python -m medfig.mri.slices` first")
        assert path_data(out.read_text(encoding="utf-8")) == path_data(p.read_text(encoding="utf-8")), (
            f"{p.name}: out/mri is not the reference. If test_build_matches_reference passes, out/mri is old: run `make mri-slices`")


def test_build_matches_reference(built):
    """The same check on a fresh build. It does not need out/mri.

    The text around the path data must equal the reference too: the viewBox, the label, the credit line, the colours
    and the stroke. So a change of vector.outline_width, of a palette colour or of a slice label fails here.
    """
    d, meta = built
    refs = sorted(REF.glob("slice_*.svg"))
    assert [p.name for p in refs] == sorted(s["file"] for s in meta["slices"]), "one reference copy for each slice"
    vb = " ".join(str(v) for v in meta["viewBox"])
    for p in refs:
        new, ref = (d / p.name).read_text(encoding="utf-8"), p.read_text(encoding="utf-8")
        assert path_data(new) == path_data(ref), p.name
        assert without_path_data(new) == without_path_data(ref), f"{p.name}: the text around the path data is not the reference"
        assert ElementTree.fromstring(new).get("viewBox") == vb, f"{p.name}: the viewBox is not the viewBox of slices.json"


def test_web_snippet_and_css(built):
    """The snippet has one .csec for each slice and one marker for each slice with a rect."""
    meta = built[1]
    n, vb = len(meta["slices"]), meta["viewBox"]
    marked = [s for s in meta["slices"] if s["rect"]]
    css = W.css(CFG)
    assert css == stack.css(CFG, n) and css.count("{") == css.count("}") and all(f".is-case .cs{k}{{" in css for k in range(2, n + 1))
    html = W.snippet(meta)
    assert html.count('class="csec ') == n and html.count('class="mk"') == html.count("<rect ") == len(marked)
    assert html.count(f'<svg viewBox="0 0 {vb[2]} {vb[3]}"') == len(marked)
    for s in meta["slices"]:
        assert f'<div class="csec cs{s["k"]}"><img src="{s["file"]}" alt="">' in html
    for s in marked:
        x, y, w, h = s["rect"]
        assert f'<rect x="{x}" y="{y}" width="{w}" height="{h}"' in html
    badges = re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="([\d.]+)"', html)
    assert len(badges) == len(marked)
    for cx, cy, r in badges:
        assert float(cx) >= float(r) and float(cy) >= float(r), "a number badge leaves the frame"
    assert W.snippet(meta, prefix="img/").count('<img src="img/slice_') == n


def test_snippet_without_marker(built):
    """A slice with rect None has no marker markup."""
    meta = json.loads((built[0] / "slices.json").read_text())
    s = meta["slices"][-1]
    s["rect"] = None
    marked = sum(1 for t in meta["slices"] if t["rect"])
    html = stack.snippet(meta, key="slices")
    assert html.count('class="csec ') == len(meta["slices"]) and html.count('class="mk"') == html.count('class="bd"') == marked
    assert f'<div class="csec cs{s["k"]}"><img src="{s["file"]}" alt=""></div>' in html


def test_web_build(built, tmp_path, monkeypatch):
    """web.build() writes the page, the snippet and the CSS from slices.json. The page carries the credit line."""
    meta = built[1]
    (tmp_path / "slices.json").write_text((built[0] / "slices.json").read_text())
    monkeypatch.setattr(W, "MRI_OUT", tmp_path)
    W.build(CFG, log=quiet)
    assert (tmp_path / "stack.css").read_text() == W.css(CFG)
    assert (tmp_path / "stack_snippet.html").read_text() == W.snippet(meta) + "\n"
    page = (tmp_path / "stack.html").read_text(encoding="utf-8")
    assert CFG["source"]["license"] in page and "schematic" in page
    assert "the left of the patient is on the right of the image" in page, "the note must name the radiological orientation"
    assert W.snippet(meta) in page and W.css(CFG) in page
    assert 'src="scan_one.svg"' in page and 'src="scan_strip.svg"' in page
    assert ">Slice</button>" in page and ">Volume</button>" in page


def test_icons(built, tmp_path, monkeypatch):
    """icons.build() writes the two scan icons. Each holds the credit line, is well-formed XML and draws scan k from slice k."""
    for p in built[0].iterdir():
        shutil.copy(p, tmp_path)
    monkeypatch.setattr(I, "MRI_OUT", tmp_path)
    I.build(CFG, log=quiet)
    meta = built[1]
    ks = [str(s["k"]) for s in meta["slices"]]
    n, vb = len(ks), " ".join(str(v) for v in meta["viewBox"])
    slices = {str(s["k"]): path_data((tmp_path / s["file"]).read_text(encoding="utf-8")) for s in meta["slices"]}
    for name, (w, h), want in (("scan_one.svg", (126, 48), ks[:1]), ("scan_strip.svg", (n * 74 + (n - 1) * 6 + 10, 36), ks)):
        svg = (tmp_path / name).read_text(encoding="utf-8")
        root = ElementTree.fromstring(svg)
        assert CFG["source"]["license"] in svg, f"{name}: no credit line"
        assert [root.get(k) for k in ("viewBox", "width", "height")] == [f"0 0 {w} {h}", str(w), str(h)], name
        assert re.findall(r'<g id="s(\d+)">', svg) == want, f"{name}: one drawing for each scan, in order"
        assert re.findall(r'<use href="#s(\d+)"/>', svg) == want, f"{name}: scan k must show slice k"
        for g in root.iter(SVG + "g"):
            assert [e.get("d") for e in g.findall(SVG_PATH)] == slices[g.get("id")[1:]], f"{name}: {g.get('id')} is not the slice"
        assert [e.get("viewBox") for e in root.findall(SVG + "svg")] == [vb] * len(want), f"{name}: each screen has the frame of the slices"


def test_command_line_help_and_unknown_step(capsys):
    """`python -m medfig mri -h` prints the usage and returns 0. An unknown step returns 2 and names the steps."""
    for flag in ("-h", "--help", "help"):
        assert cli.mri([flag]) == 0
        out, err = capsys.readouterr()
        assert "python -m medfig mri slices|icons|web|figures|render" in out and err == "", flag
    assert cli.mri(["nosuch"]) == 2
    out, err = capsys.readouterr()
    assert out == "" and "unknown step 'nosuch'" in err and all(s in err for s in cli.MRI + ["extract"])


def test_slices_step_in_an_ascii_locale(tmp_path):
    """`python -m medfig mri slices` in a process whose default text encoding is ASCII. The credit line holds a letter outside ASCII.

    The configuration and the SVG files are UTF-8 on every machine, so the step must not read or write them with the
    default encoding. The command line has no option for the output directory: the child process sets slices.MRI_OUT.
    The three variables give an ASCII default on Linux (glibc) and on macOS. Where the default stays UTF-8, the check passes and shows nothing.
    """
    env = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LC_ALL": "C",
           "PYTHONPATH": os.pathsep.join(p for p in (str(config.ROOT), os.environ.get("PYTHONPATH")) if p)}
    r = subprocess.run([sys.executable, "-c", ASCII_RUN, str(tmp_path)], env=env, cwd=config.ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    encoding = r.stdout.split()[0]
    lic = CFG["source"]["license"]
    assert json.loads((tmp_path / "slices.json").read_text(encoding="utf-8"))["license"] == lic
    for s in CFG["slices"]:
        svg = (tmp_path / f"slice_{s['k']}.svg").read_text(encoding="utf-8")
        assert lic in svg, f"slice_{s['k']}.svg: the credit line is not UTF-8 (default encoding of the process: {encoding})"

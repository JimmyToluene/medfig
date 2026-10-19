"""Checks of the medfig CT pipeline (Clinical Toolbox CT template: the bone ring and the brain, no scalp).

Run: make test (or python -m pytest -q). The checks read the CT planes file and the MRI planes file in data/. They do
not need the CT volume: the 3D intracranial mask and the plane cutter of `ct extract` run on a small synthetic head.
"""
import hashlib
import json
import re
import shutil
import sys
from xml.etree import ElementTree

import numpy as np
import pytest
from scipy import ndimage as ndi

from medfig import __main__ as cli, config, scan, stack
from medfig.ct import extract as X, icons as I, slices as C, web as W
from medfig.mri import slices as M
from test_shared import write_nifti

CFG, MCFG = config.load("ct"), config.load("mri")
NEURO = {**CFG, "orientation": "neurological"}                 # the same configuration, left of the patient on the image left
REF = config.ROOT / "tests" / "reference" / "ct"
SVG = "{http://www.w3.org/2000/svg}"


def quiet(*_):
    pass


def path_data(svg):
    return re.findall(r' d="([^"]*)"', svg)


def without_path_data(svg):
    """The SVG text with every d attribute emptied: the viewBox, the label, the credit line, the screen, the colours and the stroke."""
    return re.sub(r' d="[^"]*"', ' d=""', svg)


def points(d):
    """The coordinates of SVG path data as an (n, 2) array of (x, y)."""
    return np.array(re.findall(r"-?\d+\.\d", d), float).reshape(-1, 2)


def head(n=64):
    """A synthetic head (z, y, x) of n^3 voxels, 1 mm: air, scalp (HU 30) to r = 30, a skull shell (HU 1000) from r = 20
    to r = 23, brain (HU 35) inside r = 20 with a calcification (HU 1200) and an air pocket in it. Returns (HU, brain ball)."""
    c = (n - 1) / 2
    z, y, x = np.mgrid[:n, :n, :n]
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
    hu = np.full((n, n, n), -1000, np.int16)
    hu[r < 30] = 30
    hu[r < 23] = 1000
    hu[r < 20] = 35
    hu[28:32, 28:32, 24:28] = 1200
    hu[28:31, 40:43, 30:33] = -1000
    return hu, r < 20


@pytest.fixture(scope="module")
def planes():
    return C.load_planes(CFG)


@pytest.fixture(scope="module")
def mplanes():
    return M.load_planes(MCFG)


@pytest.fixture(scope="module")
def cut(planes):
    """For each configured slice: (slice, channels of the plane, classes from HU, rounded masks)."""
    data, meta = planes
    ps = [C.plane(data, meta, s["y_mm"], CFG) for s in CFG["slices"]]
    cs = [C.classes(p, CFG) for p in ps]
    return [(s, p, c, C.rounded(c, CFG["raster"])) for s, p, c in zip(CFG["slices"], ps, cs)]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One build into a temporary directory: (directory, content of slices.json)."""
    d = tmp_path_factory.mktemp("ct")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "CT_OUT", d)
        C.build(CFG, log=quiet)
    return d, json.loads((d / "slices.json").read_text())


@pytest.fixture(scope="module")
def mri_built(tmp_path_factory):
    """One MRI build into a temporary directory: the content of its slices.json (the markers to compare with)."""
    d = tmp_path_factory.mktemp("mri")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(M, "MRI_OUT", d)
        return M.build(MCFG, log=quiet)


def test_planes_file(planes, mplanes):
    """HU int16 and the intracranial mask uint8, on the grid of the MRI planes file, made with the thresholds of config/ct.json."""
    data, meta = planes
    mdata, mmeta = mplanes
    h = CFG["hu"]
    with np.load(config.planes_path(CFG)) as z:
        assert set(z.files) == {"hu", "icv", "meta"}
    assert data["hu"].dtype == np.int16 and data["icv"].dtype == np.uint8
    assert data["hu"].shape == data["icv"].shape == (len(meta["ys"]), meta["dims"][2], meta["dims"][0])
    assert [meta[k] for k in ("origin_mm", "dims", "vs")] == [mmeta[k] for k in ("origin_mm", "dims", "vs")], "the CT grid is not the MRI grid"
    assert data["hu"].shape[1:] == mdata["structure"].shape[1:], "a CT plane and an MRI plane must share every pixel"
    assert meta["hu"] == {k: h[k] for k in X.MASK_KEYS}, "the planes file was made with other thresholds than config/ct.json"
    assert all(s["y_mm"] in meta["ys"] and s["y_mm"] in mmeta["ys"] for s in CFG["slices"]), "every configured slice must be in both planes files"
    assert set(np.unique(data["icv"]).tolist()) <= {0, 1} and data["hu"].min() >= h["outside"]
    assert meta["ct"]["z_min_mm"] == meta["ct"]["origin_mm"][2] and meta["icv_ml"] > 0
    icv = data["icv"] > 0
    inside = (data["hu"][icv] >= h["grow_from"]) & (data["hu"][icv] < h["bone"])
    assert inside.mean() > 0.999, f"{1 - inside.mean():.4f} of the intracranial mask is bone or air: the mask must stop at the bone"


def test_plane_is_cut_at_the_field_of_view(planes):
    """Rows below the lowest CT plane (z_min_mm) are outside the CT. plane() cuts there and repeats the cut row cut_pad_px times."""
    data, meta = planes
    r, n, outside = C.cut_row(meta), CFG["raster"]["cut_pad_px"], CFG["hu"]["outside"]
    assert scan.to_mm(r, 0, scan.grid(meta, CFG))[1] == pytest.approx(meta["ct"]["z_min_mm"]), "the cut row is not the lowest CT plane"
    for s in CFG["slices"]:
        at = f"at y {s['y_mm']}"
        full = {k: scan.orient(a[meta["ys"].index(s["y_mm"])], CFG) for k, a in data.items()}
        assert (full["hu"][r + 1:] == outside).all() and not full["icv"][r + 1:].any(), f"rows below the cut must be outside the CT {at}"
        assert (full["hu"][r] > outside).any(), f"the cut row must be inside the CT {at}"
        p = C.plane(data, meta, s["y_mm"], CFG)
        for k in C.CHANNELS:
            assert p[k].shape == (r + 1 + n, full[k].shape[1]) and p[k].dtype == full[k].dtype, f"{k} {at}"
            assert np.array_equal(p[k][:r + 1], full[k][:r + 1]), f"{k}: the rows above the cut changed {at}"
            assert (p[k][r + 1:] == p[k][r]).all(), f"{k}: the padding must repeat the cut row {at}"


def test_orientation(planes, mplanes):
    """Radiological: the left structure of each slice (holiAtlas label, odd) is on the image right of the midline column
    (x = 0 mm), its right partner (the even label below it) on the image left. The neurological plane is the mirror image."""
    data, meta = planes
    mdata, mmeta = mplanes
    assert CFG["orientation"] == "radiological"
    nx, mid = meta["dims"][0], -meta["origin_mm"][0] / meta["vs"]          # column of x = 0 mm in the file
    for s in CFG["slices"]:
        y, at = s["y_mm"], f"at y {s['y_mm']}"
        radio, neuro = C.plane(data, meta, y, CFG), C.plane(data, meta, y, NEURO)
        assert all(np.array_equal(neuro[k], radio[k][:, ::-1]) for k in C.CHANNELS), f"the neurological plane is not the mirror {at}"
        rows = radio["hu"].shape[0]
        for cfg, centre, side in ((CFG, nx - 1 - mid, "right"), (NEURO, mid, "left")):
            st = C.structure(mdata, mmeta, y, cfg, rows)
            for left in s["roi_labels"]:
                cl, cr = (np.nonzero(st == v)[1].mean() for v in (left, left - 1))
                assert (cl > centre > cr) if side == "right" else (cl < centre < cr), (
                    f"label {left} ({s['roi']}) is not on the image {side} in the {cfg['orientation']} plane {at}")


def test_ct_and_mri_planes_pair(cut, mplanes):
    """The CT plane and the MRI plane of the same y share every pixel: the CT brain covers the MRI brain mask (Dice > 0.85).
    structure() puts the MRI labels on the CT rows: the same rows, then zeros where the CT plane is longer."""
    mdata, mmeta = mplanes
    for s, p, _, m in cut:
        mp = M.plane(mdata, mmeta, s["y_mm"], CFG)
        n = mp["structure"].shape[0]
        st = C.structure(mdata, mmeta, s["y_mm"], CFG, p["hu"].shape[0])
        assert st.shape == p["hu"].shape and np.array_equal(st[:n], mp["structure"]) and not st[n:].any(), f"structure() at y {s['y_mm']}"
        assert np.array_equal(C.structure(mdata, mmeta, s["y_mm"], CFG, 100), mp["structure"][:100]), "structure() must cut a longer MRI plane"
        a, b = m["brain"][:n], M.rounded(M.classes(mp, MCFG), MCFG["raster"])["mask"]
        dice = 2 * (a & b).sum() / (a.sum() + b.sum())
        assert dice > 0.85, f"Dice of the CT brain and the MRI brain mask {dice:.2f} at y {s['y_mm']}"


def test_classes_are_sane(cut):
    """ring contains brain contains CSF; the ring is one object; bone is HU above hu.bone, CSF below hu.csf; the scalp stays out."""
    h = CFG["hu"]
    for s, p, c, m in cut:
        at = f"at y {s['y_mm']}"
        for d in (c, m):
            assert not (d["brain"] & ~d["ring"]).any() and not (d["csf"] & ~d["brain"]).any(), f"the layers are not nested {at}"
        assert ndi.label(c["ring"])[1] == 1 and ndi.label(m["ring"])[1] == 1, f"the ring is not one object {at}"
        assert np.array_equal(c["bone"], p["hu"] > h["bone"]) and (p["hu"][c["bone"]] > h["bone"]).all(), f"bone {at}"
        assert not (c["brain"] & c["bone"]).any(), f"bone pixels are in the brain {at}"
        assert np.array_equal(c["brain"], (p["icv"] > 0) & ~c["bone"] & c["ring"]), f"brain is not the intracranial mask in the ring {at}"
        assert (p["hu"][c["csf"]] < h["csf"]).all(), f"CSF pixels have HU >= hu.csf {at}"
        assert (c["bone"] & c["ring"]).sum() > 0.8 * c["bone"].sum(), f"most bone of the plane must be the skull {at}"
        frac_r, frac_b, frac_c = m["ring"].mean(), m["brain"].sum() / m["ring"].sum(), m["csf"].sum() / m["brain"].sum()
        assert 0.35 < frac_r < 0.65, f"ring fraction {frac_r:.2f} {at}"
        assert 0.7 < frac_b < 0.95, f"brain share of the ring {frac_b:.2f} {at}: the ring must hold the skull"
        assert 0 < frac_c < 0.15, f"CSF share of the brain {frac_c:.3f} {at}"
        soft = (p["hu"] >= h["soft_window"][0]) & (p["hu"] <= h["soft_window"][1])
        assert (soft & ~m["ring"]).sum() > 0.1 * m["ring"].sum(), f"the scalp (soft tissue outside the ring) is missing {at}"
        assert not (m["ring"][0].any() or m["ring"][:, 0].any() or m["ring"][:, -1].any()), f"the ring touches the top or a side {at}"


def test_classes_on_a_synthetic_plane():
    """The rules of classes() on an 80 x 80 plane: scalp, a skull ring with a suture (1 px of soft tissue through the bone),
    brain with a CSF pocket, and a separate piece of bone below (the spine). The piece and the scalp are not in the ring;
    the suture is closed (all but the pixel at its outer mouth, which the rounding fills)."""
    yy, xx = np.mgrid[:80, :80]
    r = np.hypot(yy - 36, xx - 40)
    hu, icv = np.full((80, 80), -1024, np.int16), np.zeros((80, 80), np.uint8)
    skull, inside = (r < 28) & (r >= 24), r < 24
    hu[r < 32], hu[skull], hu[inside], icv[inside] = 40, 1000, 35, 1
    suture = skull & (xx == 40) & (yy < 36)
    hu[suture] = 40
    hu[36:42, 30:38] = 0                                              # CSF pocket
    hu[72:78, 36:44] = 1000                                           # bone outside the skull
    c = C.classes({"hu": hu, "icv": icv}, CFG)
    assert np.array_equal(c["bone"], hu > CFG["hu"]["bone"]) and c["bone"][72:78, 36:44].all()
    assert np.array_equal(c["ring"] & ~suture, (skull | inside) & ~suture), "the ring must be the skull and its content, without the scalp and the spine"
    assert c["ring"][suture].sum() == suture.sum() - 1 and not c["ring"][suture & (yy == yy[suture].min())].any(), "the suture is not closed"
    assert np.array_equal(c["brain"], inside) and np.array_equal(c["csf"], inside & (hu < CFG["hu"]["csf"])) and c["csf"].sum() == 48
    m = C.rounded(c, CFG["raster"])
    assert np.array_equal(m["ring"], skull | inside), "the rounded ring must be the skull and its content, the suture filled"


def test_intracranial_on_a_synthetic_head():
    """A skull shell around a soft-tissue ball, with scalp outside: the mask is the ball (its pockets filled), not the scalp."""
    hu, ball = head()
    m = X.intracranial(hu.astype(np.float32), CFG["hu"])
    assert m.dtype == bool and m.shape == hu.shape
    assert not (m & ~ball).any(), "the mask leaks out of the skull"
    assert np.array_equal(m, ball), f"the mask misses {int((ball & ~m).sum())} voxels of the ball (the calcification and the air pocket are inside)"


def test_largest():
    a = np.zeros((10, 10, 10), bool)
    a[1:3, 1:3, 1:3] = a[5:9, 5:9, 5:9] = True
    assert np.array_equal(X.largest(a), np.pad(np.ones((4, 4, 4), bool), ((5, 1), (5, 1), (5, 1))))
    b = np.zeros((6, 6), bool)
    b[1, 1] = b[3, 3] = b[4, 3] = True                                # edge neighbours only in 2D: (1, 1) is apart
    assert X.largest(b).sum() == 2 and not X.largest(b)[1, 1]


@pytest.fixture
def ct_dir(tmp_path):
    """A synthetic CT template directory: head() at 1 mm, x and y from -32 to 31 mm, z from -20 to 43 mm (the lowest plane)."""
    hu, _ = head()
    write_nifti(tmp_path / CFG["source"]["volume"], hu, [(1.0, 0, 0, -32.0), (0, 1.0, 0, -32.0), (0, 0, 1.0, -20.0)], 4)
    return tmp_path, hu


# The grid of a synthetic MRI planes file: 0.5 mm, x from -32 to 31 mm, z from -24 to 44 mm (below and above the CT).
GRID = {"origin_mm": [-32.0, -40.0, -24.0], "dims": [127, 161, 137], "vs": 0.5}


def test_ct_extract(ct_dir):
    """`ct extract` on the synthetic head: HU and the 3D intracranial mask sampled on the MRI grid, z ascending.

    MRI pixel (row 2k + 8, column 2i) is CT voxel (k, i): there the values are the CT values. Between them HU is the mean
    of the two neighbours. Rows below the lowest CT plane (z < -20 mm) and above the top one are outside: hu.outside, no mask.
    """
    src, hu = ct_dir
    out = X.extract(src, src / "planes.npz", [0, 3], CFG, GRID, log=quiet)
    with np.load(out) as z:
        p, icv, meta = z["hu"], z["icv"], json.loads(str(z["meta"]))
    assert p.dtype == np.int16 and icv.dtype == np.uint8 and p.shape == icv.shape == (2, 137, 127)
    assert meta["ys"] == [0, 3] and [meta[k] for k in ("origin_mm", "dims", "vs")] == [GRID[k] for k in ("origin_mm", "dims", "vs")]
    assert meta["ct"] == {"origin_mm": [-32.0, -32.0, -20.0], "dims": [64, 64, 64], "vs": 1.0, "z_min_mm": -20.0}
    assert meta["hu"] == {k: CFG["hu"][k] for k in X.MASK_KEYS}
    m3 = X.intracranial(hu.astype(np.float32), CFG["hu"])
    assert meta["icv_ml"] == round(m3.sum() / 1000, 1)
    for i, j in enumerate((32, 35)):                                  # y = 0 and 3 mm
        assert np.array_equal(p[i, 8:136:2, ::2], hu[:, j, :]), f"plane {i}: HU at the CT voxel centres"
        assert np.array_equal(icv[i, 8:136:2, ::2], m3[:, j, :]), f"plane {i}: the mask at the CT voxel centres"
        assert np.array_equal(p[i, 8:136:2, 1::2], np.round((hu[:, j, :-1].astype(float) + hu[:, j, 1:]) / 2)), f"plane {i}: HU between voxels"
        assert (p[i, :8] == CFG["hu"]["outside"]).all() and not icv[i, :8].any(), f"plane {i}: rows below the CT"
        assert (p[i, -2:] == CFG["hu"]["outside"]).all(), f"plane {i}: rows above the CT"


@pytest.mark.parametrize("y, match", [(0.5, "voxel grid"), (40, "outside"), (-33, "outside")])
def test_ct_extract_rejects_a_y_off_the_ct(ct_dir, y, match):
    src, _ = ct_dir
    with pytest.raises(ValueError, match=match):
        X.extract(src, src / "planes.npz", [0, y], CFG, GRID, log=quiet)
    assert not (src / "planes.npz").exists()


def test_frame_bottom_edge_is_the_cut(cut, planes, built):
    """At y = -64 mm the ring runs out of the CT field of view. The bottom edge of the frame is then the lower edge of the
    cut row (z_min_mm - pixel / 2), and every outline that reaches the cut runs past the bottom of the viewBox."""
    _, meta = planes
    d, out = built
    u, (fx, fz), vb = out["mm_per_unit"], out["frame_mm"], out["viewBox"]
    assert out["cut_z_mm"] == meta["ct"]["z_min_mm"]
    reach = [m["ring"][-1].any() for _, _, _, m in cut]
    assert any(reach), "no ring reaches the cut: the check sees nothing"
    assert fz - vb[3] * u == pytest.approx(meta["ct"]["z_min_mm"] - meta["vs"] / 2, abs=u), "the bottom edge of the frame is not the cut"
    for it, r in zip(out["slices"], reach):
        pts = points(path_data((d / it["file"]).read_text(encoding="utf-8"))[0])
        assert (pts[:, 1].max() > vb[3]) == r, f"slice {it['k']}: the ring outline and the cut do not agree"
        if not r:
            assert pts[:, 1].max() < vb[3] - CFG["vector"]["outline_width"] / 2, f"slice {it['k']}: the outline leaves the viewBox"


def test_markers_equal_the_mri_markers(built, mri_built):
    """The markers are the holiAtlas centroids, as in MRI: the same roi_mm. Each square lies in the frame, at roi_mm, and
    in the radiological orientation a left structure is on the image right of the midline."""
    out = built[1]
    u, (fx, fz), vb, sign = out["mm_per_unit"], out["frame_mm"], out["viewBox"], scan.x_sign(CFG)
    midline = sign * (0 - fx) / u
    assert out["orientation"] == "radiological" and out["marker_credit"] == CFG["source"]["marker_credit"]
    for it, mit in zip(out["slices"], mri_built["slices"]):
        assert (it["k"], it["roi"], it["roi_mm"]) == (mit["k"], mit["roi"], mit["roi_mm"]), f"slice {it['k']}: the marker is not the MRI marker"
        x, y, w, h = it["rect"]
        assert 0 <= x and x + w <= vb[2] and 0 <= y and y + h <= vb[3], f"slice {it['k']}: the marker is outside the frame"
        centre = [sign * (it["roi_mm"][0] - fx) / u, (fz - it["roi_mm"][1]) / u]
        assert centre == pytest.approx([x + w / 2, y + h / 2], abs=0.25), f"slice {it['k']}: roi_mm and rect do not agree"
        if it["roi"].startswith("left"):
            assert it["roi_mm"][0] < 0 and x + w / 2 > midline, f"slice {it['k']}: the left structure is not on the image right"


def test_build_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CT_OUT", tmp_path)
    a = C.build(CFG, log=quiet)
    h1 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("slice_*.svg")}
    C.build(CFG, log=quiet)
    h2 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("slice_*.svg")}
    assert h1 == h2 and len(h1) == len(CFG["slices"])
    assert json.loads((tmp_path / "slices.json").read_text()) == a
    assert {"viewBox", "mm_per_unit", "frame_mm", "orientation", "cut_z_mm", "license", "marker_credit", "slices"} <= set(a)
    assert a["license"] == CFG["source"]["license"] and a["orientation"] == CFG["orientation"]
    pal, sc = CFG["palette"], CFG["screen"]
    for s in a["slices"]:
        assert {"k", "y_mm", "label", "roi", "roi_mm", "rect", "file", "area_mm2"} <= set(s)
        svg = (tmp_path / s["file"]).read_text(encoding="utf-8")
        assert a["license"] in svg and "BSD 2-Clause" in svg, f"{s['file']}: no credit line"
        root = ElementTree.fromstring(svg)
        screen, *paths = list(root)
        assert screen.tag == SVG + "rect" and screen.get("fill") == pal["screen"], "the first element is the dark screen"
        assert [screen.get(k) for k in ("x", "y", "rx")] == [str(sc["inset_units"])] * 2 + [str(sc["radius_units"])]
        assert int(screen.get("y")) + int(screen.get("height")) == a["viewBox"][3], "the screen must reach the bottom edge (the cut)"
        assert [e.tag for e in paths] == [SVG + "path"] * 4
        assert [e.get("fill") for e in paths] == [pal["bone"], pal["brain"], pal["csf"], "none"]
        assert paths[3].get("stroke") == pal["outline"] and paths[3].get("d") == paths[0].get("d"), "the outline is the ring"
        assert s["area_mm2"]["ring"] > s["area_mm2"]["brain"] > s["area_mm2"]["csf"] > 0


def test_build_checks_the_configuration_before_it_writes(planes, mplanes, tmp_path, monkeypatch):
    """Thresholds that differ from those of the planes file, a planes file on another grid, a y that is not in the planes
    file or an unknown orientation stop the step before the first file. hu.csf is applied by the slices step: a change
    of it needs no new planes file."""
    _, meta = planes
    _, mmeta = mplanes
    C.check(CFG, meta, mmeta)
    for k, v in (("bone", 500), ("erode_vox", 2), ("soft_window", [-10, 100]), ("grow_from", -50), ("outside", -1000)):
        with pytest.raises(ValueError, match=r"was made with hu .*Run `python -m medfig ct extract CT_DIR` again"):
            C.check({**CFG, "hu": {**CFG["hu"], k: v}}, meta, mmeta)
    C.check({**CFG, "hu": {**CFG["hu"], "csf": 12}}, meta, mmeta)
    with pytest.raises(ValueError, match=r"the grid is not the grid of the MRI planes file.*ct extract"):
        C.check(CFG, {**meta, "vs": 1.0}, mmeta)
    out = tmp_path / "ct"
    monkeypatch.setattr(C, "CT_OUT", out)
    with pytest.raises(ValueError, match="ct extract"):
        C.build({**CFG, "hu": {**CFG["hu"], "bone": 500}}, log=quiet)
    first = CFG["slices"][0]
    with pytest.raises(ValueError, match=r"y_mm \[9\] is not in "):
        C.build({**CFG, "slices": [first, {**first, "k": 2, "y_mm": 9}]}, log=quiet)
    with pytest.raises(ValueError, match="orientation 'sideways'"):
        C.build({**CFG, "orientation": "sideways"}, log=quiet)
    assert not out.exists(), "a build that failed wrote files"


def test_build_matches_reference(built):
    """The path data and the text around them equal the reference copies in tests/reference/ct (made 2026-10-14 on a
    DGX Spark, aarch64, radiological orientation): the viewBox, the label, the credit line, the screen, the colours and the stroke."""
    d, meta = built
    refs = sorted(REF.glob("slice_*.svg"))
    assert [p.name for p in refs] == sorted(s["file"] for s in meta["slices"]), "one reference copy for each slice"
    vb = " ".join(str(v) for v in meta["viewBox"])
    for p in refs:
        new, ref = (d / p.name).read_text(encoding="utf-8"), p.read_text(encoding="utf-8")
        assert path_data(new) == path_data(ref), p.name
        assert without_path_data(new) == without_path_data(ref), f"{p.name}: the text around the path data is not the reference"
        assert ElementTree.fromstring(new).get("viewBox") == vb, f"{p.name}: the viewBox is not the viewBox of slices.json"


def test_web_build(built, tmp_path, monkeypatch):
    """web.build() writes the page, the snippet and the CSS. The page credits the CT template and, for the markers, holiAtlas."""
    meta = built[1]
    (tmp_path / "slices.json").write_text((built[0] / "slices.json").read_text())
    monkeypatch.setattr(W, "CT_OUT", tmp_path)
    W.build(CFG, log=quiet)
    page = (tmp_path / "stack.html").read_text(encoding="utf-8")
    assert CFG["source"]["license"] in page and f"Marker positions from {CFG['source']['marker_credit']}" in page and "holiAtlas" in page
    assert "schematic" in page and "the left of the patient is on the right of the image" in page
    assert (tmp_path / "stack.css").read_text() == stack.css(CFG, len(CFG["slices"])) and stack.css(CFG, len(CFG["slices"])) in page
    assert (tmp_path / "stack_snippet.html").read_text() == stack.snippet(meta, key="slices") + "\n"
    assert 'src="scan_one.svg"' in page and 'src="scan_strip.svg"' in page and "<title>Coronal CT stack</title>" in page


def test_icons(built, tmp_path, monkeypatch):
    """icons.build() writes the two scan icons: well-formed XML with the credit line, scan k drawn from slice k."""
    for p in built[0].iterdir():
        shutil.copy(p, tmp_path)
    monkeypatch.setattr(I, "CT_OUT", tmp_path)
    I.build(CFG, log=quiet)
    ks = [str(s["k"]) for s in built[1]["slices"]]
    for name, want in (("scan_one.svg", ks[:1]), ("scan_strip.svg", ks)):
        svg = (tmp_path / name).read_text(encoding="utf-8")
        ElementTree.fromstring(svg)
        assert CFG["source"]["license"] in svg, f"{name}: no credit line"
        assert re.findall(r'<use href="#s(\d+)"/>', svg) == want, f"{name}: scan k must show slice k"
        assert "One attached CT slice" in svg or "Attached CT slices" in svg, f"{name}: the text alternative must name CT"


def test_command_line(monkeypatch, capsys):
    """`python -m medfig ct -h` prints the usage and returns 0. An unknown step returns 2 and names the steps."""
    monkeypatch.setattr(sys, "argv", ["medfig", "ct", "-h"])
    assert cli.main() == 0
    out, err = capsys.readouterr()
    assert "python -m medfig ct extract CT_DIR" in out and err == ""
    monkeypatch.setattr(sys, "argv", ["medfig", "ct", "nosuch"])
    assert cli.main() == 2
    out, err = capsys.readouterr()
    assert out == "" and "medfig ct: unknown step 'nosuch'" in err and all(s in err for s in cli.MRI + ["extract"])

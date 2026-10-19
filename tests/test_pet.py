"""Checks of the medfig PET pipeline (rPOP florbetaben templates, amyloid-negative and amyloid-positive, SUVR bands).

Run: make test (or python -m pytest -q). The checks read the PET planes file and the MRI planes file in data/. They do
not need the PET templates or holiAtlas: the reference mean and the plane cutter of `pet extract` run on small synthetic
volumes. Each series (config series) is written like one modality to out/pet/<series>/.
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

from medfig import __main__ as cli, config, stack
from medfig.mri import slices as M
from medfig.pet import extract as X, icons as I, slices as P, web as W
from test_shared import write_nifti

CFG, MCFG = config.load("pet"), config.load("mri")
NEURO = {**CFG, "orientation": "neurological"}                 # the same configuration, left of the patient on the image left
REF = config.ROOT / "tests" / "reference" / "pet"
SVG = "{http://www.w3.org/2000/svg}"
KEYS = [se["key"] for se in CFG["series"]]


def quiet(*_):
    pass


def path_data(svg):
    return re.findall(r' d="([^"]*)"', svg)


def without_path_data(svg):
    """The SVG text with every d attribute emptied: the viewBox, the label, the credit line, the colours and the stroke."""
    return re.sub(r' d="[^"]*"', ' d=""', svg)


@pytest.fixture(scope="module")
def planes():
    return P.load_planes(CFG)


@pytest.fixture(scope="module")
def mplanes():
    return M.load_planes(MCFG)


@pytest.fixture(scope="module")
def layers(planes, mplanes):
    """For each series key: for each configured slice, (slice, SUVR plane, band masks [mask, band 1, ...])."""
    data, meta = planes
    mdata, mmeta = mplanes
    out = {}
    for se in CFG["series"]:
        rows = []
        for s in CFG["slices"]:
            sv = P.suvr(data, meta, se["channel"], s["y_mm"], mmeta, CFG)
            mask = P.brain(M.plane(mdata, mmeta, s["y_mm"], CFG), MCFG)
            rows.append((s, sv, P.bands(sv, mask, CFG, mmeta["vs"])))
        out[se["key"]] = rows
    return out


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One build into a temporary directory: (directory, {series key: content of slices.json})."""
    d = tmp_path_factory.mktemp("pet")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(P, "PET_OUT", d)
        P.build(CFG, log=quiet)
    return d, {k: json.loads((d / k / "slices.json").read_text()) for k in KEYS}


@pytest.fixture(scope="module")
def mri_built(tmp_path_factory):
    """One MRI build into a temporary directory: (directory, content of slices.json)."""
    d = tmp_path_factory.mktemp("mri")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(M, "MRI_OUT", d)
        return d, M.build(MCFG, log=quiet)


def test_planes_file(planes, mplanes):
    """float32 native planes (2 mm) of every template, z and x ascending, and the reference mean of each template."""
    data, meta = planes
    _, mmeta = mplanes
    vols = CFG["source"]["volumes"]
    with np.load(config.planes_path(CFG)) as z:
        assert set(z.files) == set(vols) | {"meta"}
    assert set(data) == set(vols) and all(a.dtype == np.float32 for a in data.values())
    assert all(a.shape == (len(meta["ys"]), meta["dims"][2], meta["dims"][0]) for a in data.values())
    assert meta["vs"] == 2.0 and meta["volumes"] == vols
    assert all(meta["origin_mm"][1] + j * meta["vs"] == y for y, j in zip(meta["ys"], meta["js"])), "js are not the voxel rows of ys"
    assert all(s["y_mm"] in meta["ys"] and s["y_mm"] in mmeta["ys"] for s in CFG["slices"]), "every configured slice must be in both planes files"
    ref = meta["reference"]
    assert ref["name"] == CFG["reference"]["name"] and ref["tissue_labels"] == CFG["reference"]["tissue_labels"]
    for se in CFG["series"]:
        mean = ref["mean"][se["channel"]]
        assert np.isfinite(mean) and mean > 0 and ref["points"][se["channel"]] > 0, f"{se['channel']}: no reference mean"
        assert data[se["channel"]].min() >= 0, f"{se['channel']}: negative uptake"


def test_suvr_is_the_native_plane_over_the_reference_mean(planes, mplanes):
    """suvr() samples the native plane at the MRI pixel centres. MRI pixel (z index 4l - 16, x index 4i) is native voxel
    (l, i) (x = -90 + 2i mm, z = -80 + 2l mm): there the SUVR is the native value over the reference mean. In the
    radiological orientation that pixel is at row nz - 1 - z index, column nx - 1 - x index; the neurological plane is the mirror."""
    data, meta = planes
    _, mmeta = mplanes
    nx, _, nz = mmeta["dims"]
    for se in CFG["series"]:
        for s in CFG["slices"]:
            sv = P.suvr(data, meta, se["channel"], s["y_mm"], mmeta, CFG)
            assert sv.shape == (nz, nx)
            assert np.array_equal(P.suvr(data, meta, se["channel"], s["y_mm"], mmeta, NEURO), sv[:, ::-1]), "the neurological plane is not the mirror"
            nat = data[se["channel"]][meta["ys"].index(s["y_mm"])]
            ll, ii = np.arange(4, nat.shape[0]), np.arange(nat.shape[1])
            k, c = 4 * ll - 16, 4 * ii
            ll, k, ii, c = ll[k < nz], k[k < nz], ii[c < nx], c[c < nx]
            assert len(k) > 20 and len(c) > 40
            want = nat[np.ix_(ll, ii)] / meta["reference"]["mean"][se["channel"]]
            assert np.allclose(sv[np.ix_(nz - 1 - k, nx - 1 - c)], want, atol=1e-6), f"{se['key']} at y {s['y_mm']}: SUVR is not native / reference mean"


def test_orientation(mplanes):
    """Radiological: the left structure of each slice (holiAtlas label, odd) is on the image right of the midline column
    (x = 0 mm), its right partner (the even label below it) on the image left. PET reads the MRI planes in its own orientation."""
    mdata, mmeta = mplanes
    assert CFG["orientation"] == "radiological"
    nx, mid = mmeta["dims"][0], -mmeta["origin_mm"][0] / mmeta["vs"]
    for s in CFG["slices"]:
        st = M.plane(mdata, mmeta, s["y_mm"], CFG)["structure"]
        for left in s["roi_labels"]:
            cl, cr = (np.nonzero(st == v)[1].mean() for v in (left, left - 1))
            assert cl > nx - 1 - mid > cr, f"label {left} ({s['roi']}) is not on the image right at y {s['y_mm']}"


def test_bands_are_nested(layers):
    """Band k lies inside band k - 1 and the first layer is the brain mask. A band pixel has smoothed SUVR >= its edge
    (but at the rounded border). There is one colour per layer: len(edges) + 1."""
    b, pal = CFG["bands"], CFG["palette"]
    assert len(pal["bands"]) == len(b["edges"]) + 1 and b["edges"] == sorted(b["edges"])
    for key, rows in layers.items():
        for s, sv, ls in rows:
            at = f"{key} at y {s['y_mm']}"
            assert len(ls) == len(pal["bands"]), at
            assert all(not (hi & ~lo).any() for lo, hi in zip(ls, ls[1:])), f"the bands are not nested {at}"
            sm = ndi.gaussian_filter(sv, b["smooth_sigma_mm"] / 0.5)
            for e, m in zip(b["edges"], ls[1:]):
                if m.any():
                    assert (sm[m] >= e).mean() > 0.9, f"band {e} holds pixels below its edge {at}"
                    assert (sm[ls[0] & ~m] < e).mean() > 0.9, f"pixels above {e} are outside the band {at}"


def test_positive_template_has_more_uptake(built):
    """The pair shows where the uptake differs: every band of the amyloid-positive slice is larger than that of the negative one."""
    out = built[1]
    for p, n in zip(out["pos"]["slices"], out["neg"]["slices"]):
        assert p["area_mm2"]["mask"] == n["area_mm2"]["mask"], f"slice {p['k']}: the two series have other brain masks"
        assert all(a > b for a, b in zip(p["area_mm2"]["bands"], n["area_mm2"]["bands"])), f"slice {p['k']}: {p['area_mm2']} vs {n['area_mm2']}"
        assert p["area_mm2"]["bands"] == sorted(p["area_mm2"]["bands"], reverse=True), "a band larger than the one below it"


def test_frame_and_outline_equal_mri(built, mri_built):
    """The PET outline is the MRI brain mask: the viewBox, the frame, the mask path and the outline path equal those of
    the MRI slice of the same y. The markers (holiAtlas centroids) equal the MRI markers, so a left structure is on the image right."""
    d, out = built
    md, mout = mri_built
    for key in KEYS:
        o = out[key]
        assert [o[k] for k in ("viewBox", "mm_per_unit", "frame_mm", "orientation")] == [mout[k] for k in ("viewBox", "mm_per_unit", "frame_mm", "orientation")], key
        for it, mit in zip(o["slices"], mout["slices"]):
            pet, mri = path_data((d / key / it["file"]).read_text(encoding="utf-8")), path_data((md / mit["file"]).read_text(encoding="utf-8"))
            assert pet[0] == mri[0] and pet[-1] == mri[-1], f"{key} slice {it['k']}: the mask or the outline is not the MRI one"
            assert (it["roi"], it["roi_mm"], it["rect"]) == (mit["roi"], mit["roi_mm"], mit["rect"]), f"{key} slice {it['k']}: the marker is not the MRI marker"


def test_build_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PET_OUT", tmp_path)
    a = P.build(CFG, log=quiet)
    h1 = {str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*.svg")}
    P.build(CFG, log=quiet)
    h2 = {str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*.svg")}
    assert h1 == h2 and len(h1) == len(KEYS) * (len(CFG["slices"]) + 1), "slices and key.svg of every series"
    assert set(a) == set(KEYS) and sorted(p.name for p in tmp_path.iterdir()) == sorted(KEYS)
    pal = CFG["palette"]
    for se in CFG["series"]:
        o = a[se["key"]]
        assert json.loads((tmp_path / se["key"] / "slices.json").read_text()) == o
        assert o["series"] == se and o["bands"] == CFG["bands"]["edges"] and o["colours"] == pal["bands"] and o["orientation"] == CFG["orientation"]
        assert o["license"] == CFG["source"]["license"] and o["outline_credit"] == CFG["source"]["outline_credit"]
        assert o["reference"] == CFG["reference"]["name"] and o["reference_mean"] > 0
        for s in o["slices"]:
            svg = (tmp_path / se["key"] / s["file"]).read_text(encoding="utf-8")
            assert o["license"] in svg and "CC BY-NC-SA 4.0" in svg and se["name"] in svg, f"{se['key']}/{s['file']}: no credit line or series name"
            paths = ElementTree.fromstring(svg).findall(SVG + "path")
            assert [e.get("fill") for e in paths] == pal["bands"] + ["none"], "one fill per colour, lowest first, then the outline"
            assert paths[-1].get("stroke") == pal["outline"] and paths[-1].get("d") == paths[0].get("d"), "the outline is the mask"
            assert len(s["area_mm2"]["bands"]) == len(CFG["bands"]["edges"])


@pytest.mark.parametrize("key", KEYS)
def test_build_matches_reference(built, key):
    """The path data and the text around them equal the reference copies in tests/reference/pet/<series> (made 2026-10-14
    on a DGX Spark, aarch64, radiological orientation): the viewBox, the label, the credit line, the colours and the stroke."""
    d, out = built
    refs = sorted((REF / key).glob("slice_*.svg"))
    assert [p.name for p in refs] == sorted(s["file"] for s in out[key]["slices"]), "one reference copy for each slice"
    vb = " ".join(str(v) for v in out[key]["viewBox"])
    for p in refs:
        new, ref = (d / key / p.name).read_text(encoding="utf-8"), p.read_text(encoding="utf-8")
        assert path_data(new) == path_data(ref), f"{key}/{p.name}"
        assert without_path_data(new) == without_path_data(ref), f"{key}/{p.name}: the text around the path data is not the reference"
        assert ElementTree.fromstring(new).get("viewBox") == vb, f"{key}/{p.name}: the viewBox is not the viewBox of slices.json"


def test_check_errors(planes, tmp_path, monkeypatch):
    """A palette without one colour more than the edges, edges that do not ascend, a series channel that is not in the
    planes file, a y that is not in it or an unknown orientation stop the step before the first file."""
    _, meta = planes
    P.check(CFG, meta)
    pal, b = CFG["palette"], CFG["bands"]
    for bad in ({**CFG, "palette": {**pal, "bands": pal["bands"][:-1]}}, {**CFG, "bands": {**b, "edges": b["edges"] + [2.2]}}):
        with pytest.raises(ValueError, match="palette.bands needs one colour more than bands.edges"):
            P.check(bad, meta)
    with pytest.raises(ValueError, match="bands.edges must ascend"):
        P.check({**CFG, "bands": {**b, "edges": b["edges"][::-1]}}, meta)
    with pytest.raises(ValueError, match=r"series channel \['fbb_mid'\] is not in"):
        P.check({**CFG, "series": CFG["series"] + [{**CFG["series"][0], "key": "mid", "channel": "fbb_mid"}]}, meta)
    out = tmp_path / "pet"
    monkeypatch.setattr(P, "PET_OUT", out)
    with pytest.raises(ValueError, match="palette.bands"):
        P.build({**CFG, "palette": {**pal, "bands": pal["bands"] + ["#000000"]}}, log=quiet)
    first = CFG["slices"][0]
    with pytest.raises(ValueError, match=r"y_mm \[9\] is not in "):
        P.build({**CFG, "slices": [first, {**first, "k": 2, "y_mm": 9}]}, log=quiet)
    with pytest.raises(ValueError, match="orientation 'sideways'"):
        P.build({**CFG, "orientation": "sideways"}, log=quiet)
    assert not out.exists(), "a build that failed wrote files"


def test_key_svg(built):
    """key.svg of each series: well-formed XML, one swatch per colour in order, the band edges as text, the unit."""
    d, _ = built
    pal, b = CFG["palette"], CFG["bands"]
    for key in KEYS:
        svg = (d / key / "key.svg").read_text(encoding="utf-8")
        assert svg == P.key_svg(CFG)
        root = ElementTree.fromstring(svg)
        rects = root.findall(SVG + "rect")
        assert [r.get("fill") for r in rects] == pal["bands"] + ["none"], "one swatch per colour, then the frame"
        assert [t.text for t in root.findall(SVG + "text")] == [f"{e:.1f}" for e in b["edges"]] + [b["unit"]]
        assert b["unit"] in root.get("aria-label") and root.get("width") == root.get("viewBox").split()[2]


def test_web_build(built, tmp_path, monkeypatch):
    """web.build() writes the page of each series. It credits the templates and, for the outline, the reference region and
    the markers, holiAtlas; it shows key.svg and names the radiological orientation."""
    d, out = built
    for key in KEYS:
        (tmp_path / key).mkdir()
        shutil.copy(d / key / "slices.json", tmp_path / key)
    monkeypatch.setattr(W, "PET_OUT", tmp_path)
    W.build(CFG, log=quiet)
    for se in CFG["series"]:
        page = (tmp_path / se["key"] / "stack.html").read_text(encoding="utf-8")
        assert CFG["source"]["license"] in page and CFG["source"]["outline_credit"] in page and "holiAtlas" in page, se["key"]
        assert f"reference region ({CFG['reference']['name']})" in page and '<img src="key.svg"' in page, se["key"]
        assert f"<title>Coronal amyloid PET stack, {se['name']}</title>" in page and se["label"] in page
        assert "schematic" in page and "the left of the patient is on the right of the image" in page
        assert stack.snippet(out[se["key"]], key="slices") in page
        assert (tmp_path / se["key"] / "stack.css").read_text() == stack.css(CFG, len(CFG["slices"]))


def test_icons(built, tmp_path, monkeypatch):
    """icons.build() writes the two scan icons of each series: well-formed XML with the credit line, scan k drawn from slice k."""
    d, out = built
    shutil.copytree(d, tmp_path, dirs_exist_ok=True)
    monkeypatch.setattr(I, "PET_OUT", tmp_path)
    I.build(CFG, log=quiet)
    for key in KEYS:
        ks = [str(s["k"]) for s in out[key]["slices"]]
        for name, want in (("scan_one.svg", ks[:1]), ("scan_strip.svg", ks)):
            svg = (tmp_path / key / name).read_text(encoding="utf-8")
            ElementTree.fromstring(svg)
            assert CFG["source"]["license"] in svg, f"{key}/{name}: no credit line"
            assert re.findall(r'<use href="#s(\d+)"/>', svg) == want, f"{key}/{name}: scan k must show slice k"


def test_command_line(monkeypatch, capsys):
    """`python -m medfig pet -h` prints the usage and returns 0. An unknown step returns 2 and names the steps."""
    monkeypatch.setattr(sys, "argv", ["medfig", "pet", "-h"])
    assert cli.main() == 0
    out, err = capsys.readouterr()
    assert "python -m medfig pet extract PET_DIR HOLIATLAS_DIR" in out and err == ""
    monkeypatch.setattr(sys, "argv", ["medfig", "pet", "nosuch"])
    assert cli.main() == 2
    out, err = capsys.readouterr()
    assert out == "" and "medfig pet: unknown step 'nosuch'" in err and all(s in err for s in cli.MRI + ["extract"])


def linear(x, y, z):
    """A linear uptake: linear sampling gives it back exactly, so the mean over points is the value at their mean."""
    return 10 + 0.5 * x + 0.25 * y + 0.125 * z


def pet_volume(shape=(8, 8, 8), origin=(-4.0, -6.0, -8.0), vs=2.0):
    """linear() at the voxel centres of a (z, y, x) grid."""
    k, j, i = np.mgrid[:shape[0], :shape[1], :shape[2]]
    return linear(origin[0] + vs * i, origin[1] + vs * j, origin[2] + vs * k).astype(np.float32)


def test_reference_mean_of_a_constant_volume():
    """A constant volume gives that constant. Labelled voxels outside the PET grid sample 0 and are left out."""
    vol = np.full((6, 6, 6), 2.5, np.float32)                          # 2 mm, x, y and z from 0 to 10 mm
    tissue = np.zeros((14, 14, 14), np.uint8)                          # 1 mm, from 0 to 13 mm
    tissue[2:5, 2:5, 2:5] = 5
    assert X.reference_mean(vol, [0, 0, 0], 2.0, tissue, [0, 0, 0], 1.0, [5, 6]) == (2.5, 27)
    tissue[:] = 6
    mean, n = X.reference_mean(vol, [0, 0, 0], 2.0, tissue, [0, 0, 0], 1.0, [5, 6])
    assert mean == pytest.approx(2.5) and n == 11 ** 3, "only the voxels from 0 to 10 mm are inside the PET field of view"


def test_reference_mean_samples_at_the_voxel_centres():
    """A linear uptake on another grid (other origin, 2 mm against 1 mm): the mean is the uptake at the labelled voxel
    centres, so the x, y and z axes and the origins are used the right way round. Other labels are left out."""
    t_origin = [-3.0, -2.0, -1.0]
    tissue = np.full((6, 5, 4), 2, np.uint8)                           # (z, y, x), 1 mm: x -3 to 0, y -2 to 2, z -1 to 4 mm
    tissue[0, 1, 2] = tissue[5, 4, 0] = 5
    tissue[3, 0, 3] = 6
    kk, jj, ii = np.nonzero(np.isin(tissue, [5, 6]))
    want = linear(t_origin[0] + ii, t_origin[1] + jj, t_origin[2] + kk).mean()
    mean, n = X.reference_mean(pet_volume(), [-4.0, -6.0, -8.0], 2.0, tissue, t_origin, 1.0, [5, 6])
    assert n == 3 and mean == pytest.approx(want, abs=1e-5)


def test_reference_mean_outside_the_field_of_view():
    tissue = np.zeros((4, 4, 4), np.uint8)
    tissue[1, 1, 1] = 5
    with pytest.raises(ValueError, match=r"reference labels \[5, 6\] are outside the PET field of view"):
        X.reference_mean(pet_volume(), [-4.0, -6.0, -8.0], 2.0, tissue, [100.0, 0.0, 0.0], 1.0, [5, 6])


@pytest.fixture
def pet_dirs(tmp_path):
    """Synthetic rPOP and holiAtlas directories: two PET templates (linear() and twice it) on the grid of pet_volume(),
    and a tissue atlas at 1 mm with a cerebellum (labels 5 and 6) of 2 x 2 x 2 voxels."""
    pet, holi = tmp_path / "rpop", tmp_path / "holi"
    pet.mkdir(), holi.mkdir()
    srow = [(2.0, 0, 0, -4.0), (0, 2.0, 0, -6.0), (0, 0, 2.0, -8.0)]
    vols = {"fbb_neg": pet_volume(), "fbb_pos": 2 * pet_volume()}
    for key, name in CFG["source"]["volumes"].items():
        write_nifti(pet / name, vols[key], srow, 16)
    tissue = np.full((6, 5, 4), 2, np.uint8)
    tissue[1:3, 1:3, 1:3] = 5
    tissue[2, 2, 2] = 6
    write_nifti(holi / CFG["reference"]["tissue_volume"], tissue, [(1.0, 0, 0, -3.0), (0, 1.0, 0, -2.0), (0, 0, 1.0, -1.0)], 2)
    return pet, holi, vols, tissue


def test_pet_extract(pet_dirs):
    """`pet extract`: the native planes of each template (z and x ascending, float32) and the reference mean of each."""
    pet, holi, vols, tissue = pet_dirs
    out = X.extract(pet, holi, pet / "planes.npz", [0, -6], CFG, log=quiet)
    with np.load(out) as z:
        data, meta = {k: z[k] for k in vols}, json.loads(str(z["meta"]))
    assert meta["ys"] == [0, -6] and meta["js"] == [3, 0] and meta["dims"] == [8, 8, 8] and meta["vs"] == 2.0
    assert meta["origin_mm"] == [-4.0, -6.0, -8.0]
    kk, jj, ii = np.nonzero(np.isin(tissue, CFG["reference"]["tissue_labels"]))
    base = linear(-3.0 + ii, -2.0 + jj, -1.0 + kk).mean()
    for key, f in (("fbb_neg", 1), ("fbb_pos", 2)):
        assert data[key].dtype == np.float32 and data[key].shape == (2, 8, 8)
        assert all(np.array_equal(data[key][i], vols[key][:, j, :]) for i, j in enumerate(meta["js"])), f"{key}: the planes are not the native planes"
        assert meta["reference"]["mean"][key] == pytest.approx(f * base, abs=1e-5) and meta["reference"]["points"][key] == len(kk)
    with pytest.raises(ValueError, match="voxel grid"):
        X.extract(pet, holi, pet / "other.npz", [1], CFG, log=quiet)
    with pytest.raises(ValueError, match="outside the volume"):
        X.extract(pet, holi, pet / "other.npz", [10], CFG, log=quiet)
    assert not (pet / "other.npz").exists()

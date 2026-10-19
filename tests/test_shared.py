"""Checks of the code that every modality uses: config.py, nifti.py, vector.py, stack.py, render.py and the command line.
The code of the scanner modalities (scan.py: orientation, grid, frame, markers) has its own file, test_scan.py.

Run: make test (or python -m pytest -q).
The NIfTI files are synthetic. Each test writes them into tmp_path with numpy and struct. The MRI plane cutter
(mri/extract.py) runs on a synthetic template directory. The last check builds the WSI sections and the WSI stack page
again and compares them with tests/reference, so a change of the shared code that changes a WSI product fails here.
"""
import gzip
import json
import re
import struct
import sys

import numpy as np
import pytest
from scipy import ndimage as ndi

from medfig import __main__ as cli, config, nifti, render, scan, sections as S, stack, vector as V, web as WSI_WEB
from medfig.mri import extract as X, slices as M

WSI, MRI = config.load(), config.load("mri")
SROW = [(0.5, 0, 0, -1.0), (0, 0.5, 0, -2.0), (0, 0, 0.5, -1.5)]      # x, y and z ascend, 0.5 mm, voxel (0, 0, 0) at (-1, -2, -1.5) mm
VOL = np.arange(4 * 9 * 6).reshape(4, 9, 6)                           # (z, y, x), values 0 to 215
NUMBER = r"-?\d+\.\d"
STACK_FILES = ("stack.css", "stack_snippet.html", "stack.html")       # WSI copies in tests/reference (the published v0.1.0 page)


def quiet(*_):
    pass


def path_data(svg):
    return re.findall(r' d="([^"]*)"', svg)


def points(d):
    """The coordinates of SVG path data as an (n, 2) array of (x, y)."""
    return np.array(re.findall(NUMBER, d), float).reshape(-1, 2)


def write_nifti(path, a, srow, datatype, slope=1.0, inter=0.0, sizeof_hdr=348, sform_code=1, vox_offset=352):
    """Write the (z, y, x) array as a little-endian NIfTI-1 file: 348-byte header, 4 empty bytes, voxels with x fastest.

    A vox_offset above 352 puts more empty bytes between the header and the voxels, as a header extension does.
    """
    nz, ny, nx = a.shape
    h = bytearray(vox_offset)
    struct.pack_into("<i", h, 0, sizeof_hdr)
    struct.pack_into("<8h", h, 40, 3, nx, ny, nz, 1, 1, 1, 1)                               # dim
    struct.pack_into("<2h", h, 70, datatype, a.dtype.itemsize * 8)                          # datatype, bitpix
    struct.pack_into("<8f", h, 76, 1, *(abs(srow[i][i]) for i in range(3)), 0, 0, 0, 0)     # pixdim
    struct.pack_into("<3f", h, 108, vox_offset, slope, inter)                               # vox_offset, scl_slope, scl_inter
    struct.pack_into("<2h", h, 252, 0, sform_code)                                          # qform_code, sform_code
    struct.pack_into("<12f", h, 280, *(v for row in srow for v in row))                     # srow_x, srow_y, srow_z
    h[344:348] = b"n+1\0"
    with (gzip.open if str(path).endswith(".gz") else open)(path, "wb") as f:
        f.write(bytes(h) + a.astype(a.dtype.newbyteorder("<")).tobytes())
    return path


@pytest.mark.parametrize("name", ["v.nii", "v.nii.gz"])
@pytest.mark.parametrize("code, dtype", [(2, np.uint8), (4, np.int16), (16, np.float32)])
def test_load_volume(tmp_path, name, code, dtype):
    a, origin, vs = nifti.load_volume(write_nifti(tmp_path / name, VOL.astype(dtype), SROW, code))
    assert a.dtype == dtype and a.shape == VOL.shape and np.array_equal(a, VOL)
    assert origin == [-1.0, -2.0, -1.5] and vs == 0.5


@pytest.mark.parametrize("code, dtype, factor", [(8, np.int32, -100_000), (64, np.float64, 1 / 7), (512, np.uint16, 300), (768, np.uint32, 19_000_000)],
                         ids=["int32", "float64", "uint16", "uint32"])
def test_load_volume_wide_types(tmp_path, code, dtype, factor):
    """Values that int16 and float32 cannot hold come back the same. The template files do not use these types."""
    want = (VOL * factor).astype(dtype)
    assert np.abs(want).max() > 32767 or not np.array_equal(want, want.astype(np.float32)), "the values must need the wide type"
    a = nifti.load_volume(write_nifti(tmp_path / "v.nii.gz", want, SROW, code))[0]
    assert a.dtype == dtype and np.array_equal(a, want)


def test_load_volume_reads_the_voxels_at_vox_offset(tmp_path):
    """16 bytes between the header and the voxels: vox_offset is 368, not 352."""
    a = nifti.load_volume(write_nifti(tmp_path / "v.nii", VOL.astype(np.int16), SROW, 4, vox_offset=368))[0]
    assert a.dtype == np.int16 and np.array_equal(a, VOL)


@pytest.mark.parametrize("i", [0, 1, 2], ids=["x", "y", "z"])
def test_load_volume_makes_each_axis_ascend(tmp_path, i):
    """A file stores the same grid with one axis in reverse. The volume comes back the same, with the same origin."""
    srow = [list(row) for row in SROW]
    srow[i][i], srow[i][3] = -0.5, SROW[i][3] + 0.5 * (VOL.shape[2 - i] - 1)
    a, origin, vs = nifti.load_volume(write_nifti(tmp_path / "v.nii", np.flip(VOL, 2 - i).astype(np.int16), srow, 4))
    assert np.array_equal(a, VOL), "the axis does not ascend"
    assert origin == [-1.0, -2.0, -1.5] and vs == 0.5


def test_load_volume_applies_slope_and_intercept(tmp_path):
    for slope, inter, want in ((2.0, -3.0, VOL * 2.0 - 3.0), (0.5, 0.0, VOL * 0.5), (1.0, 5.0, VOL + 5.0)):
        a = nifti.load_volume(write_nifti(tmp_path / "v.nii", VOL.astype(np.int16), SROW, 4, slope, inter))[0]
        assert np.array_equal(a, want), f"scl_slope {slope}, scl_inter {inter}"
    for slope in (1.0, 0.0):
        a = nifti.load_volume(write_nifti(tmp_path / "v.nii", VOL.astype(np.int16), SROW, 4, slope, 0.0))[0]
        assert a.dtype == np.int16 and np.array_equal(a, VOL), f"scl_slope {slope} must keep the stored values"


def test_load_volume_slope_zero_means_no_scaling(tmp_path):
    a = nifti.load_volume(write_nifti(tmp_path / "v.nii", VOL.astype(np.int16), SROW, 4, slope=0.0, inter=5.0))[0]
    assert np.array_equal(a, VOL)


@pytest.mark.parametrize("slope", [float("nan"), float("inf")], ids=["NaN", "inf"])
def test_load_volume_slope_not_finite_means_no_scaling(tmp_path, slope):
    """Some writers leave scl_slope as NaN. The stored values come back, not a volume of NaN."""
    a = nifti.load_volume(write_nifti(tmp_path / "v.nii", VOL.astype(np.int16), SROW, 4, slope=slope, inter=0.0))[0]
    assert a.dtype == np.int16 and np.array_equal(a, VOL)


@pytest.mark.parametrize("srow, code, sizeof_hdr", [
    ([(0.5, 0.1, 0, -1.0), SROW[1], SROW[2]], 4, 348),
    ([SROW[0], (0, 1.0, 0, -2.0), SROW[2]], 4, 348),
    (SROW, 128, 348),
    (SROW, 32, 348),
    (SROW, 4, 0x5C010000),
], ids=["oblique sform", "anisotropic", "RGB24", "complex64", "big-endian header"])
def test_load_volume_rejects_what_it_cannot_read(tmp_path, srow, code, sizeof_hdr):
    f = write_nifti(tmp_path / "v.nii.gz", VOL.astype(np.int16), srow, code, sizeof_hdr=sizeof_hdr)
    with pytest.raises(ValueError):
        nifti.load_volume(f)


def test_load_volume_rejects_a_missing_sform(tmp_path):
    f = write_nifti(tmp_path / "v.nii", VOL.astype(np.int16), [(0, 0, 0, 0)] * 3, 4, sform_code=0)
    with pytest.raises(ValueError):
        nifti.load_volume(f)


@pytest.fixture
def template(tmp_path):
    """A synthetic template directory. The three volumes have the configured file names and one grid.

    T1w is float32 and the label maps are int8, as in holiAtlas. The structure volume is stored with x in reverse.
    Each T1w value holds its voxel position: 100 y + 10 z + x.
    """
    z, y, x = np.mgrid[:4, :9, :6]
    vols = {"t1w": 100 * y + 10 * z + x, "tissue": (x + y + z) % 10, "structure": 6 * y + x}
    names = MRI["source"]["volumes"]
    write_nifti(tmp_path / names["t1w"], vols["t1w"].astype(np.float32), SROW, 16)
    write_nifti(tmp_path / names["tissue"], vols["tissue"].astype(np.int8), SROW, 256)
    write_nifti(tmp_path / names["structure"], vols["structure"][:, :, ::-1].astype(np.int8), [(-0.5, 0, 0, 1.5), SROW[1], SROW[2]], 256)
    return tmp_path, vols


def test_mri_extract(template):
    src, vols = template
    ys = [2, 0, -1.5]                                                 # y = -2 + 0.5 j, so j = 8, 4, 1
    out = X.extract(src, src / "planes.npz", ys, MRI, log=quiet)
    with np.load(out) as z:
        data, meta = {k: z[k] for k in M.CHANNELS}, json.loads(str(z["meta"]))
    assert meta["ys"] == ys and meta["js"] == [8, 4, 1]
    assert meta["dims"] == [6, 9, 4] and meta["vs"] == 0.5 and meta["origin_mm"] == [-1.0, -2.0, -1.5]
    for key, dtype in (("t1w", np.uint16), ("tissue", np.uint8), ("structure", np.uint8)):
        assert data[key].dtype == dtype and data[key].shape == (3, 4, 6)
        for i, j in enumerate(meta["js"]):
            assert np.array_equal(data[key][i], vols[key][:, j, :]), f"{key}: plane {i} is not the plane at y index {j}"
    assert MRI["orientation"] == "radiological"
    p = M.plane(data, meta, 0, MRI)
    assert all(np.array_equal(p[k], vols[k][::-1, 4, ::-1]) for k in M.CHANNELS), (
        "plane() must put the highest z in row 0 and, in the radiological orientation, the highest x in column 0")
    p = M.plane(data, meta, 0, {**MRI, "orientation": "neurological"})
    assert all(np.array_equal(p[k], vols[k][::-1, 4, :]) for k in M.CHANNELS), (
        "plane() must put the highest z in row 0 and, in the neurological orientation, the lowest x in column 0")


@pytest.mark.parametrize("y", [2.5, -2.5, 30])
def test_mri_extract_rejects_y_outside_the_volume(template, y):
    with pytest.raises(ValueError, match="outside"):
        X.extract(template[0], template[0] / "planes.npz", [0, y], MRI, log=quiet)
    assert not (template[0] / "planes.npz").exists()


@pytest.mark.parametrize("y", [0.2, 0.25, -1.75])
def test_mri_extract_rejects_y_between_two_planes(template, y):
    """The grid has a plane every 0.5 mm. A y between two planes must stop the script: it must not take the nearest plane."""
    with pytest.raises(ValueError, match="voxel grid"):
        X.extract(template[0], template[0] / "planes.npz", [0, y], MRI, log=quiet)
    assert not (template[0] / "planes.npz").exists()


@pytest.mark.parametrize("y_args, ys, js", [(["--y", "-1,0"], [-1, 0], [2, 4]), (["--y=-2,1"], [-2, 1], [0, 6]), (["--y", "2,-1"], [2, -1], [8, 2])],
                         ids=["--y -1,0", "--y=-2,1", "--y 2,-1"])
def test_mri_extract_command_line(template, capsys, y_args, ys, js):
    """`python -m medfig mri extract DIR OUT.npz --y ...`. A list that starts with a negative value must work."""
    src, vols = template
    out = src / "cut.npz"
    assert not cli.mri(["extract", str(src), str(out)] + y_args), "the step must return 0 or None"
    with np.load(out) as z:
        t1w, meta = z["t1w"], json.loads(str(z["meta"]))
    assert meta["ys"] == ys and meta["js"] == js
    assert all(np.array_equal(t1w[i], vols["t1w"][:, j, :]) for i, j in enumerate(js))
    assert "wrote" in capsys.readouterr().out


@pytest.mark.parametrize("argv, code, stream", [(["-h"], 0, "out"), (["nosuch"], 2, "err"), (["mri", "-h"], 0, "out"), (["mri", "nosuch"], 2, "err"),
                                                (["ct", "-h"], 0, "out"), (["ct", "nosuch"], 2, "err"), (["pet", "help"], 0, "out"), (["pet", "nosuch"], 2, "err")],
                         ids=["-h", "nosuch", "mri -h", "mri nosuch", "ct -h", "ct nosuch", "pet help", "pet nosuch"])
def test_command_line_exit_codes(monkeypatch, capsys, argv, code, stream):
    """`python -m medfig ...`: the help text goes to stdout with exit code 0. An unknown step gives exit code 2 and the text on stderr."""
    monkeypatch.setattr(sys, "argv", ["medfig"] + argv)
    assert cli.main() == code
    text = capsys.readouterr()
    assert "python -m medfig mri extract HOLIATLAS_DIR" in getattr(text, stream)
    assert (text.err == "") == (code == 0) and (text.out == "") == (code == 2)


def test_each_environment_variable_changes_its_own_modality(monkeypatch, tmp_path):
    """MEDFIG_PLANES and MEDFIG_CONFIG are for WSI. MEDFIG_<MODALITY>_PLANES and MEDFIG_<MODALITY>_CONFIG are for MRI, CT and PET."""
    assert sorted(config.MODALITIES) == ["ct", "mri", "pet", "wsi"]
    cfgs = {m: config.load(m) for m in config.MODALITIES}
    planes = {m: config.planes_path(c) for m, c in cfgs.items()}
    assert planes == {m: config.ROOT / c["source"]["planes_npz"] for m, c in cfgs.items()}
    assert len(set(planes.values())) == len(planes), "each modality has its own planes file"
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"source": {"planes_npz": "data/other.npz"}}), encoding="utf-8")
    for mine, (_, env_config, env_planes) in config.MODALITIES.items():
        rest = [m for m in config.MODALITIES if m != mine]
        with monkeypatch.context() as mp:
            mp.setenv(env_planes, str(tmp_path / "other.npz"))
            assert config.planes_path(config.load(mine)) == tmp_path / "other.npz", f"{env_planes} must change the {mine} planes file"
            assert all(config.planes_path(config.load(m)) == planes[m] for m in rest), f"{env_planes} must not change {rest}"
        with monkeypatch.context() as mp:
            mp.setenv(env_config, str(other))
            cfg = config.load(mine)
            assert cfg["_path"] == str(other) and cfg["_modality"] == mine and config.planes_path(cfg) == config.ROOT / "data" / "other.npz"
            assert all(config.load(m)["_path"] == str(config.ROOT / "config" / config.MODALITIES[m][0]) for m in rest), f"{env_config} must not change {rest}"
    assert (WSI["_modality"], MRI["_modality"]) == ("wsi", "mri") and "sections" in WSI and "slices" in MRI


def test_config_load_rejects_an_unknown_modality():
    """The argument is a modality, not a path. The message names the modalities and the variables for another file."""
    with pytest.raises(ValueError, match=r"unknown modality 'spect': use one of \['ct', 'mri', 'pet', 'wsi'\]\..*MEDFIG_<MODALITY>_CONFIG"):
        config.load("spect")
    with pytest.raises(ValueError, match="unknown modality"):
        config.load("config/mri.json")


def test_mri_extract_rejects_another_grid_and_values_that_do_not_fit(template):
    src, vols = template
    f = src / MRI["source"]["volumes"]["structure"]
    write_nifti(f, vols["structure"].astype(np.int8), [SROW[0], SROW[1], (0, 0, 0.5, 0.0)], 256)
    with pytest.raises(ValueError, match="grid"):
        X.extract(src, src / "planes.npz", [0], MRI, log=quiet)
    write_nifti(f, (vols["structure"] - 1).astype(np.int8), SROW, 256)                     # one label is -1
    with pytest.raises(ValueError, match="fit"):
        X.extract(src, src / "planes.npz", [-2], MRI, log=quiet)
    write_nifti(src / MRI["source"]["volumes"]["t1w"], (vols["t1w"] + 0.5).astype(np.float32), SROW, 16)
    with pytest.raises(ValueError, match="fit"):
        X.extract(src, src / "planes.npz", [0], MRI, log=quiet)


def shapes():
    """A disc of radius 18 px with its centre at row 25, column 40, and a 5 x 5 px dot far from it."""
    yy, xx = np.mgrid[:60, :90]
    dot = np.zeros((60, 90), bool)
    dot[50:55, 80:85] = True
    return (yy - 25) ** 2 + (xx - 40) ** 2 < 18 ** 2, dot


@pytest.mark.parametrize("v", [WSI["vector"], MRI["vector"]], ids=["wsi", "mri"])
def test_path_d_scale(v):
    disc, _ = shapes()
    d1 = V.path_d(disc, 3, 5, v)
    assert d1 == V.path_d(disc, 3, 5, v, 1.0) == V.path_d(disc, 3, 5, v, scale=1), "scale 1 must not change the string"
    assert d1.startswith("M") and d1.endswith("Z") and d1.count("M") == 1
    a = points(d1)
    assert (a.min(0) + a.max(0)) / 2 == pytest.approx([40 - 3, 25 - 5], abs=0.5), "x is the column, y is the row, less the frame origin"
    assert (a.max(0) - a.min(0)) / 2 == pytest.approx([18, 18], abs=1.0)
    d2 = V.path_d(disc, 3, 5, v, 2)
    assert re.sub(NUMBER, "n", d2) == re.sub(NUMBER, "n", d1), "scale must change the numbers only"
    assert np.abs(points(d2) - 2 * a).max() <= 0.15 + 1e-9, "scale 2 must double every coordinate (to one decimal)"


def test_path_d_leaves_out_empty_and_tiny_masks():
    v = MRI["vector"]
    one = np.zeros((10, 10), bool)
    one[5, 5] = True
    assert V.path_d(np.zeros((10, 10), bool), 0, 0, v) == "" and V.path_d(one, 0, 0, v) == ""


def test_round_mask_drops_small_objects():
    disc, dot = shapes()
    both = V.round_mask(disc | dot, 1.0)
    lab, n = ndi.label(both)
    small = int(np.bincount(lab.ravel())[1:].min())
    assert n == 2 and 0 < small < 30
    assert ndi.label(V.round_mask(disc | dot, 1.0, small - 1))[1] == 2, "an object of more than min_object_px pixels must stay"
    one = V.round_mask(disc | dot, 1.0, small)
    assert ndi.label(one)[1] == 1 and np.array_equal(one, V.round_mask(disc, 1.0)), "an object of min_object_px pixels must go"


def test_round_mask_rounds_corners_and_closes_gaps():
    sq = np.zeros((40, 60), bool)
    sq[10:30, 10:30] = sq[10:30, 31:50] = True                        # two blocks, a gap of one pixel at column 30
    r = V.round_mask(sq, 1.6)
    assert sq[10, 10] and not r[10, 10], "the corner pixel must go"
    assert not sq[20, 30] and r[20, 30], "the gap of one pixel must close"
    assert r[10, 20] and r[20, 10] and not r[9, 20], "a straight edge must stay in place"


@pytest.mark.parametrize("v", [WSI["vector"], MRI["vector"]], ids=["wsi", "mri"])
def test_contours_close_at_the_image_edge(v):
    full = np.ones((20, 30), bool)
    half = np.zeros((20, 30), bool)
    half[:, :12] = True
    for m, w in ((full, 30), (half, 12)):
        cs = V.contours(m)
        assert len(cs) == 1 and np.array_equal(cs[0][0], cs[0][-1]), "the contour must close"
        assert cs[0].min() == -0.5 and cs[0].max(0).tolist() == [19.5, w - 0.5]
        d = V.path_d(m, 0, 0, v)
        assert d.count("M") == 1 and d.endswith("Z")
        assert points(d).min() == -0.5 and points(d).max(0).tolist() == [w - 0.5, 19.5]


def delay(css, selector):
    return float(re.search(re.escape(selector) + r"\{[^}]*transition-delay:([\d.]+)s", css).group(1))


@pytest.mark.parametrize("cfg", [WSI, MRI], ids=["wsi", "mri"])
@pytest.mark.parametrize("n", [1, 3, 5])
def test_stack_css(cfg, n):
    """Five items are more than the delay lists of the configuration hold. The items beyond the lists fall back."""
    st = cfg["stack_3d"]
    css = stack.css(cfg, n)
    assert css.endswith("}\n") and css.count("{") == css.count("}")
    assert len(re.findall(r"transition-delay:[\d.]+s", css)) == 1 + 2 * (n - 1) + n
    assert f".cs{n + 1}" not in css and all(f".is-case .cs{k}{{" in css for k in range(2, n + 1))
    section = [delay(css, f".is-case .cs{k}") for k in range(2, n + 1)]
    marker = [delay(css, ".is-case .cs1 .bd")] + [delay(css, f".is-case .cs{k} .mk,.is-case .cs{k} .bd") for k in range(2, n + 1)]
    assert section == sorted(set(section)) and marker == sorted(set(marker)), "a later item must come in later"
    assert section[:2] == st["section_delay_s"][1:n][:2] and marker[:3] == st["marker_delay_s"][:n], "configured delays come first"
    for k in range(len(st["back_filter"]) + 1, n + 1):
        assert re.search(rf"\.is-case \.cs{k}\{{[^}}]*filter:none;", css), f"item {k} has no configured filter"
    for k in range(2, n + 1):
        assert f".is-case .cs{k}{{opacity:1;transform:translateZ({-st['case']['depth_step_px'] * (k - 1)}px)" in css


def test_stack_snippet_order_prefix_and_missing_rect():
    meta = {"viewBox": [0, 0, 100, 80], "items": [{"k": 1, "file": "a.svg", "rect": [10, 20, 30, 30]},
                                                  {"k": 2, "file": "b.svg", "rect": None}, {"k": 3, "file": "c.svg"}]}
    html = stack.snippet(meta, prefix="img/", key="items")
    assert re.findall(r'class="csec cs(\d)"', html) == ["3", "2", "1"], "the markup runs back to front"
    assert re.findall(r'<img src="([^"]*)"', html) == ["img/c.svg", "img/b.svg", "img/a.svg"]
    assert html.count('class="mk"') == html.count('class="bd"') == 1 and '<svg viewBox="0 0 100 80"' in html
    assert '<rect x="10" y="20" width="30" height="30"' in html and html.startswith('<div class="brainbox"><div class="stack">')


def test_marker_cells_and_scan_caption():
    """The cells of figures S7 and M6 (and C6, P7): one figure for each item, a dashed square only for an item with a rect.
    The caption and the credit line of the scanner modalities are in scan.py. No browser is needed."""
    items = [{"k": 1, "y_mm": 8, "roi": "left putamen", "file": "slice_1.svg", "rect": [1, 2, 3, 4]},
             {"k": 2, "y_mm": -22, "roi": None, "file": "slice_2.svg", "rect": None}]
    assert [scan._caption(s) for s in items] == ["y = +8 mm · marker 1: left putamen", "y = -22 mm"]
    html = render.marker_cells(items, [0, 0, 100, 80], scan._caption, "#123456")
    assert html.count("<figure") == html.count("</figure>") == 2 and html.count("<rect ") == html.count("<svg ") == 1
    assert re.findall(r"<img src='([^']*)'", html) == ["slice_1.svg", "slice_2.svg"]
    assert "viewBox='0 0 100 80'" in html and "x='1' y='2' width='3' height='4'" in html and "stroke='#123456'" in html
    assert re.findall(r"<figcaption>([^<]*)</figcaption>", html) == [scan._caption(s) for s in items]
    first, second = html.split("</figure>")[:2]
    assert "<rect " in first and "<rect " not in second, "the square belongs to the item with the rect"
    credit = scan.credit_line(MRI["source"]["license"], 12, 16)
    assert f">Source: {MRI['source']['license']}</p>" in credit and "12px" in credit and "16px" in credit and scan.CREDIT in credit


def test_wsi_sections_match_reference(tmp_path, monkeypatch):
    """The move of the tracing code to vector.py must not change WSI. This repeats test_reference_paths on a fresh build.

    It does not need out/. The stack page, its snippet and its CSS must equal the copies in tests/reference too:
    stack.py serves every modality, so a change for MRI must not change the published WSI page. If you change
    stack.py or the WSI stack_3d values on purpose, update the three copies in the same commit and say why.
    """
    monkeypatch.setattr(S, "OUT", tmp_path)
    out = S.build(WSI, log=quiet)
    refs = sorted((config.ROOT / "tests" / "reference").glob("section_*.svg"))
    assert [p.name for p in refs] == sorted(s["file"] for s in out["sections"]), "one reference copy for each section"
    for p in refs:
        assert path_data((tmp_path / p.name).read_text()) == path_data(p.read_text()), p.name
    meta = json.loads((tmp_path / "sections.json").read_text())
    html = WSI_WEB.snippet(meta)
    assert html == stack.snippet(meta) and html.count('class="csec ') == html.count('class="mk"') == len(meta["sections"])
    assert WSI_WEB.css(WSI) == stack.css(WSI, len(WSI["sections"]))
    monkeypatch.setattr(WSI_WEB, "OUT", tmp_path)
    WSI_WEB.build(WSI, log=quiet)
    for name in STACK_FILES:
        new, ref = (tmp_path / name).read_text(encoding="utf-8"), (config.ROOT / "tests" / "reference" / name).read_text(encoding="utf-8")
        assert new == ref, f"{name}: the WSI stack is not the copy in tests/reference"

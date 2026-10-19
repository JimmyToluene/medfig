"""Checks of the chain that makes the report numbers. Run: make test (or python -m pytest -q).

The figures steps compute the numbers of each modality and write build/fig/numbers.json (WSI), mri_numbers.json,
ct_numbers.json and pet_numbers.json. report.py reads these files, the slices.json files of out/, the planes files and the
four configurations, and writes report/numbers.tex. report.tex uses the macros of numbers.tex.

The checks build the WSI sections and the MRI, CT and PET slices once, into a temporary directory. They call the functions
of mri/figures.py that compute. They draw no figure and start no browser.
Two checks compare with tracked files: build/fig/mri_numbers.json and report/numbers.tex must be what the code gives now.
The report step itself checks that build/fig/ct_numbers.json and pet_numbers.json agree with a fresh out/ct and out/pet.
If one of them fails after a change on purpose, run the steps of the modality (`make sections figures`, `make mri`,
`make ct`, `make pet`), then `python -m medfig report`.
"""
import copy
import json
import re
import shutil

import numpy as np
import pytest

from medfig import config, report, sections as S
from medfig.ct import slices as C
from medfig.mri import figures as F, slices as M
from medfig.pet import slices as P

WSI, MRI, CT, PET = config.load(), config.load("mri"), config.load("ct"), config.load("pet")
FIG = config.BUILD / "fig"
REPORT = config.ROOT / "report"
SERIES = [se["key"] for se in PET["series"]]
FIRST_SERIES = SERIES[0]
INPUTS = ("out/sections.json", "out/mri/slices.json", "out/ct/slices.json", *(f"out/pet/{k}/slices.json" for k in SERIES),
          "build/fig/numbers.json", "build/fig/mri_numbers.json", "build/fig/ct_numbers.json", "build/fig/pet_numbers.json")
PANELS = ("_candidates", "_strip", "_classes", "_contours", "_frame", "_pairing")       # the functions of mri/figures.py that draw
FIRST, LAST = MRI["slices"][0], MRI["slices"][-1]
RADIO = {"orientation": "radiological"}
NEURO = {"orientation": "neurological"}


def quiet(*_):
    pass


def defined(numbers_tex):
    """The names that a numbers.tex text defines: (macros, colours)."""
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", numbers_tex)), set(re.findall(r"\\definecolor\{([A-Za-z]+)\}", numbers_tex))


def used(report_tex):
    """The names of numbers.tex that a report text uses: (macros, colours).

    A macro of numbers.tex starts with v, m, ct or pet and a capital letter, or ends with Rows, Delays or Sizes.
    A colour of numbers.tex starts with m, ct or pet and a capital letter, and stands alone in braces.
    """
    words = set(re.findall(r"\\([A-Za-z]+)", report_tex))
    macros = {w for w in words if re.match(r"(v|m|ct|pet)[A-Z]", w) or re.search(r"(Rows|Delays|Sizes)$", w)}
    return macros, set(re.findall(r"\{((?:m|ct|pet)[A-Z][A-Za-z]*)\}", report_tex))


def rewrite(path, change):
    """Read a JSON file, apply change(content) and write it back."""
    d = json.loads(path.read_text(encoding="utf-8"))
    change(d)
    path.write_text(json.dumps(d), encoding="utf-8")


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    """One build of the WSI sections and the MRI, CT and PET slices into a temporary directory with out/, out/mri/, out/ct/, out/pet/."""
    root = tmp_path_factory.mktemp("report")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "OUT", root / "out")
        mp.setattr(M, "MRI_OUT", root / "out" / "mri")
        mp.setattr(C, "CT_OUT", root / "out" / "ct")
        mp.setattr(P, "PET_OUT", root / "out" / "pet")
        S.build(WSI, log=quiet)
        M.build(MRI, log=quiet)
        C.build(CT, log=quiet)
        P.build(PET, log=quiet)
    return root


@pytest.fixture(scope="module")
def paired(fresh):
    """The records of mri/figures.py for the configured slices. Each slice is paired with the WSI section of the same y."""
    data, meta = M.load_planes(MRI)
    J = json.loads((fresh / "out" / "mri" / "slices.json").read_text(encoding="utf-8"))
    grid = F._grid(meta, MRI)
    items, wgrid = F._items(MRI, WSI, J, data, meta, grid)
    return {"J": J, "data": data, "meta": meta, "grid": grid, "items": items, "wgrid": wgrid}


@pytest.fixture
def site(fresh, tmp_path, monkeypatch):
    """A directory with every input of the report step, and report.py pointed at it. report/numbers.tex is not there yet.

    The slices.json files of out/ come from the fresh build. The numbers files are the tracked ones of build/fig.
    The planes files are the tracked ones of data/ (report.py finds them through the configurations).
    """
    for name in INPUTS:
        src = (fresh if name.startswith("out/") else config.ROOT) / name
        assert src.is_file(), (f"{name} is missing: run the steps of its modality (`make sections figures`, `make mri`, `make ct`, "
                               "`make pet`), or restore build/fig with `git checkout -- build/fig`")
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, tmp_path / name)
    (tmp_path / "report").mkdir()
    for name, val in (("ROOT", tmp_path), ("OUT", tmp_path / "out"), ("BUILD", tmp_path / "build")):
        monkeypatch.setattr(report, name, val)
    return tmp_path


def test_dice():
    a, b, c = (np.zeros((6, 6), bool) for _ in range(3))
    a[0:3], b[0:2], c[4:6] = True, True, True                         # 18 px; 12 px inside a; 12 px outside a
    assert F._dice(a, a) == 1.0 and F._dice(a, c) == 0.0
    assert F._dice(a, b) == F._dice(b, a) == pytest.approx(2 * 12 / (18 + 12))


def test_resample():
    """Nearest-neighbour copy of a mask to another grid. A grid is (x of column 0, z of row 0, pixel size, x step) in mm.
    z falls with the row. The x step is negative in the radiological orientation."""
    m = np.zeros((5, 7), bool)
    m[1, 2] = m[3, 5] = m[4, 0] = True
    g = (-3.0, 10.0, 0.5, 0.5)
    assert np.array_equal(F._resample(m, g, g, m.shape), m), "the same grid must give the same mask"
    moved = F._resample(m, g, (g[0] + 2 * 0.5, g[1] - 1 * 0.5, 0.5, 0.5), m.shape)      # the grid starts 2 columns to the right and 1 row lower
    want = np.zeros_like(m)
    want[:4, :5] = m[1:, 2:]
    assert np.array_equal(moved, want), "pixel (r, c) of the new grid is pixel (r + 1, c + 2) of the mask; outside the mask it is empty"
    fine = F._resample(m, g, (g[0] - 0.125, g[1] + 0.125, 0.25, 0.25), (10, 14))         # pixels of half the size on the same area
    assert np.array_equal(fine, m.repeat(2, axis=0).repeat(2, axis=1)), "each pixel becomes 2 x 2 pixels"
    wide = F._resample(m, g, (g[0] - 2 * 0.5, g[1] + 2 * 0.5, 0.5, 0.5), (9, 11))       # a margin of 2 pixels on every side
    assert np.array_equal(wide[2:7, 2:9], m) and wide.sum() == m.sum()
    mirror = (g[0] + 6 * 0.5, g[1], 0.5, -0.5)                                           # the same area, x falling with the column
    assert np.array_equal(F._resample(m, g, mirror, m.shape), m[:, ::-1]), "a radiological grid holds the mask mirrored"
    assert np.array_equal(F._resample(m[:, ::-1], mirror, g, m.shape), m), "and the copy back gives the mask"


def test_grids():
    """_grid: the MRI plane after slices.plane(), in each orientation. _wsi_grid: the WSI plane after the block mean of
    sections.plane() (x ascending). _extent: the edges of an image on a grid, for imshow(extent=...)."""
    meta = {"vs": 0.5, "origin_mm": [-90.0, -126.0, -72.0], "dims": [362, 434, 362]}
    assert F._grid(meta, NEURO) == (-90.0, 108.5, 0.5, 0.5)
    assert F._grid(meta, RADIO) == (90.5, 108.5, 0.5, -0.5), "radiological: column 0 is the highest x"
    assert F._grid(meta, MRI) == F._grid(meta, {"orientation": MRI["orientation"]})
    wmeta = {"srow": [[0.1, 0, 0, -7.0], [0, 0.1, 0, -12.0], [0, 0, 0.1, -5.0]], "vs": 0.1}
    data = np.zeros((1, 9, 6), np.uint8)                              # (plane, z, x): z from -5.0 to -4.2 mm, x from -7.0 to -6.5 mm
    assert F._wsi_grid(data, wmeta, 1) == pytest.approx((-7.0, -4.2, 0.1, 0.1))
    # block mean 3 x 3: pixel (0, 0) is the mean of x = -7.0, -6.9, -6.8 and z = -4.2, -4.3, -4.4, so its centre is (-6.9, -4.3)
    assert F._wsi_grid(data, wmeta, 3) == pytest.approx((-6.9, -4.3, 0.3, 0.3))
    assert F._extent((4, 6), (-3.0, 10.0, 0.5, 0.5)) == pytest.approx((-3.25, -0.25, 8.25, 10.25)), "edges: left, right, bottom, top"
    assert F._extent((4, 6), (2.0, 10.0, 0.5, -0.5)) == pytest.approx((2.25, -0.75, 8.25, 10.25)), "radiological: left > right"


def test_pairing_with_wsi(paired):
    """The MRI brain mask and the WSI tissue mask of the same y overlap when both are placed in mm."""
    items, grid, wgrid = paired["items"], paired["grid"], paired["wgrid"]
    assert [it["s"]["k"] for it in items] == [s["k"] for s in MRI["slices"]] and [it["J"]["k"] for it in items] == [s["k"] for s in MRI["slices"]]
    shared = [it for it in items if it["wsi"]]
    assert shared, "the report needs an MRI slice at the y of a WSI section"
    for it in shared:
        at = f"at y {it['s']['y_mm']}"
        assert it["wsi"]["y_mm"] == it["s"]["y_mm"] and it["wsi"] in WSI["sections"]
        assert 0.85 < it["dice"] < 1 and 0.85 < it["dice_tissue"] < 1, f"Dice {it['dice']:.3f}, {it['dice_tissue']:.3f} {at}"
        mask = it["m"]["mask"]
        assert it["dice"] == F._dice(mask, F._resample(it["wsi_t"], wgrid, grid, mask.shape))
        for dx, dz in ((10, 0), (-10, 0), (0, 10), (0, -10)):                  # the WSI mask 10 mm off: the overlap must drop
            off = F._resample(it["wsi_t"], (wgrid[0] + dx, wgrid[1] + dz, wgrid[2], wgrid[3]), grid, mask.shape)
            assert F._dice(mask, off) < it["dice"] - 0.05, f"a shift of ({dx}, {dz}) mm does not lower the Dice value {at}"
        if it["centre_mm"]:
            assert it["offset_mm"] == pytest.approx([b - c for b, c in zip(it["wsi"]["block_mm"], it["centre_mm"])])
            assert it["distance_mm"] == pytest.approx(float(np.hypot(*it["offset_mm"])))


def test_numbers_of_the_slices_are_the_tracked_numbers(paired):
    """build/fig/mri_numbers.json holds, for each slice, the values that the code gives now. The report quotes them."""
    f = FIG / "mri_numbers.json"
    assert f.is_file(), "build/fig/mri_numbers.json is missing: run `make mri-figures`"
    tracked = json.loads(f.read_text(encoding="utf-8"))
    new = F._numbers(MRI, paired["items"], paired["grid"][2])
    assert set(new) == {"area_mm2", "fractions", "mask_steps", "roi", "wsi_block", "pairing"}
    for key, val in new.items():
        assert json.loads(json.dumps(val)) == tracked[key], (
            f"{key}: build/fig/mri_numbers.json is not what the code gives now. Run `make mri-figures`, then `python -m medfig report`")
    for a, s in zip(new["area_mm2"], paired["J"]["slices"]):
        assert a["mask"] == s["area_mm2"]["mask"] == a["csf"] + a["grey_matter"] + a["white_matter"], "the classes do not add up to the mask"


@pytest.mark.parametrize("change, what", [
    (lambda J: J["slices"].pop(), f"slice {LAST['k']} (y = {LAST['y_mm']} mm)"),
    (lambda J: J["slices"][0].update(y_mm=FIRST["y_mm"] + 8), f"slice {FIRST['k']} (y = {FIRST['y_mm']} mm)"),
    (lambda J: J["slices"].append({**J["slices"][0], "k": 99}), "slices [99] are not in the configuration"),
    (lambda J: J["slices"][0].update(roi_mm=[0.0, 0.0]), f"marker of slice {FIRST['k']}"),
    (lambda J: J["slices"][0].update(roi_mm=None, rect=None), f"marker of slice {FIRST['k']}"),
], ids=["a slice is missing", "another y", "a slice that is not configured", "another marker", "no marker"])
def test_figures_reject_a_stale_slices_json(paired, change, what):
    """out/mri/slices.json of another configuration stops the figures step. The message names the slice and the command."""
    J = copy.deepcopy(paired["J"])
    change(J)
    with pytest.raises(ValueError) as e:
        F._items(MRI, WSI, J, paired["data"], paired["meta"], paired["grid"])
    assert str(e.value).startswith(what + ": ") and MRI["_path"] in str(e.value) and "python -m medfig mri slices" in str(e.value)


def test_example_slice_is_chosen_by_k(fresh, paired, tmp_path, monkeypatch):
    """build() takes the worked example by its k, not by its place in the list. The panels are not drawn here.

    The numbers file lists the size of the slice files of slices.json only, not of another file in out/mri/.
    """
    out = shutil.copytree(fresh / "out" / "mri", tmp_path / "mri")
    (out / "slice_99.svg").write_text("<svg/>", encoding="utf-8")       # a file of an older configuration
    by_k = {it["s"]["k"]: it for it in paired["items"]}
    assert 2 in by_k and len(by_k) >= 3, "this check needs three slices and one with k = 2"
    order = [k for k in by_k if k != 2] + [2]                          # k = 2 is the last slice, not the second
    drawn = []
    for name in PANELS:
        monkeypatch.setattr(F, name, lambda *a, **k: {})
    monkeypatch.setattr(F, "_strip", lambda cfg, p, c, *a: drawn.append(p) or {})
    monkeypatch.setattr(F, "FIG", tmp_path)
    monkeypatch.setattr(M, "MRI_OUT", out)

    def build(ks, example=None):
        monkeypatch.setattr(F, "_items", lambda *a: ([by_k[k] for k in ks], paired["wgrid"]))
        if example is None:
            monkeypatch.delenv("MEDFIG_MRI_EXAMPLE", raising=False)
        else:
            monkeypatch.setenv("MEDFIG_MRI_EXAMPLE", str(example))
        return F.build({**MRI, "slices": [by_k[k]["s"] for k in ks]}, log=quiet)

    for ks, example, want in ((order, None, 2), (order, order[0], order[0]), (order[:-1], None, order[0]), (order[1:2], None, order[1])):
        num = build(ks, example)
        assert (num["example_k"], num["example_y_mm"]) == (want, by_k[want]["s"]["y_mm"]), f"slices {ks}, MEDFIG_MRI_EXAMPLE {example}"
        assert drawn[-1] is by_k[want]["p"], "the panels must show the plane of the example"
        assert json.loads((tmp_path / "mri_numbers.json").read_text(encoding="utf-8")) == json.loads(json.dumps(num))
        assert set(num["svg_kb"]) == {s["file"] for s in paired["J"]["slices"]}
    with pytest.raises(ValueError, match=rf"MEDFIG_MRI_EXAMPLE=99: the configured slices are k = \[{', '.join(str(k) for k in sorted(by_k))}\]"):
        build(order, 99)


def test_report_macros_are_defined():
    """Every macro and every colour of numbers.tex that report.tex uses is defined in the tracked report/numbers.tex.
    The check covers the four modalities: macros \\v..., \\m..., \\ct... and \\pet..."""
    macros, colours = defined((REPORT / "numbers.tex").read_text(encoding="utf-8"))
    used_macros, used_colours = used((REPORT / "report.tex").read_text(encoding="utf-8"))
    assert len(used_macros) > 20 and used_colours, "the check found too few names in report.tex: the patterns in used() are wrong"
    for prefix in ("v", "m", "ct", "pet"):
        assert any(re.match(prefix + "[A-Z]", w) for w in used_macros), f"report.tex uses no macro \\{prefix}...: the patterns in used() are wrong"
    assert {c[:2] for c in used_colours} >= {"mC", "ct"} and any(c.startswith("pet") for c in used_colours)
    assert not sorted(used_macros - macros), "report.tex uses macros that numbers.tex does not define"
    assert not sorted(used_colours - colours), "report.tex uses colours that numbers.tex does not define"


def test_macro_names_follow_the_prefixes():
    """\\newcommand stops on a name that exists. Every name of numbers.tex is letters only and starts with the prefix of its
    modality (v, m, ct, pet) and a capital letter, or is a WSI table macro (...Rows, ...Delays, ...Sizes). LaTeX and the
    packages of report.tex have no command of this form, so a new macro cannot clash; `make report` would stop if one did."""
    macros, colours = defined((REPORT / "numbers.tex").read_text(encoding="utf-8"))
    rule = r"(v|m|ct|pet)[A-Z][A-Za-z]*|[A-Z][A-Za-z]*(Rows|Delays|Sizes)"
    assert not sorted(n for n in macros if not re.fullmatch(rule, n))
    assert not sorted(n for n in colours if not re.fullmatch(r"(m|ct|pet)[A-Z][A-Za-z]*", n))


def test_tex_helpers():
    assert report._tex("50% of a_b & #1") == r"50\% of a\_b \& \#1"
    assert report._pair([-22.25, -1.7]) == r"$(-22.2,\,-1.7)$" and report._pair([-24, 4], 0) == r"$(-24,\,4)$"
    assert report._cells([0, 0.12, 0.26]) == "0.00 & 0.12 & 0.26"
    assert report._command("mVbW", 517) == r"\newcommand{\mVbW}{517}"
    assert [report._half_up(x) for x in (0.965, 0.9652, 0.954, 0.925, 0.9)] == ["0.97", "0.97", "0.95", "0.93", "0.90"], (
        "Dice values stored with 3 decimals round half up, as the figure titles do")


def test_mri_rows():
    """The three MRI tables: one row per slice, per WSI block and per pair. A slice without a marker shows two dashes."""
    sl = {"slices": [{"k": 1, "y_mm": 8, "label": "frontal lobe & striatum", "roi": "left putamen", "roi_mm": [-22.2, -1.7]},
                     {"k": 2, "y_mm": -22, "label": "thalamus", "roi": None, "roi_mm": None}]}
    num = {"area_mm2": [{"k": 2, "y_mm": -22, "mask": 200, "csf": 20, "grey_matter": 120, "white_matter": 60},
                        {"k": 1, "y_mm": 8, "mask": 100, "csf": 10, "grey_matter": 60, "white_matter": 30}],
           "wsi_block": [{"k": 1, "y_mm": 8, "block": "putamen", "block_mm": [-24, 4], "roi_mm": [-22.2, -1.7], "offset_mm": [-1.8, 5.7], "distance_mm": 5.98}],
           "pairing": [{"k": 1, "y_mm": 8, "dice": 0.9018, "dice_tissue": 0.9151, "mri_mask_mm2": 100, "wsi_tissue_mm2": 90},
                       {"k": 2, "y_mm": -22, "dice": 0.9254, "dice_tissue": 0.9336, "mri_mask_mm2": 200, "wsi_tissue_mm2": 180}]}
    rows = report._mri_rows(num, sl)
    assert list(rows) == ["mSliceRows", "mBlockRows", "mPairRows"]
    assert rows["mSliceRows"] == [r"1 & $+8$ & frontal lobe \& striatum & left putamen & $(-22.2,\,-1.7)$ & 100 & 10 & 60 & 30 \\",
                                  r"2 & $-22$ & thalamus & -- & -- & 200 & 20 & 120 & 60 \\"]
    assert rows["mBlockRows"] == [r"1 & $+8$ & putamen & $(-24,\,4)$ & left putamen & $(-22.2,\,-1.7)$ & $(-1.8,\,5.7)$ & 6.0 \\"]
    assert rows["mPairRows"] == [r"1 & $+8$ & 100 & 90 & 0.90 & 0.92 \\", r"2 & $-22$ & 200 & 180 & 0.93 & 0.93 \\"]


def test_ct_rows():
    """The CT tables: one row per slice (ring, bone, intracranial space, CSF) and one per pair with holiAtlas."""
    sl = {"slices": [{"k": 1, "y_mm": 8, "label": "frontal lobe", "roi": "left putamen", "roi_mm": [-22.2, -1.7],
                      "area_mm2": {"ring": 160, "brain": 130, "csf": 7}},
                     {"k": 2, "y_mm": -22, "label": "thalamus", "roi": None, "roi_mm": None, "area_mm2": {"ring": 180, "brain": 150, "csf": 8}}]}
    num = {"bone_mm2": [{"k": 2, "y_mm": -22, "bone": 30}, {"k": 1, "y_mm": 8, "bone": 30}],
           "pairing": [{"k": 1, "y_mm": 8, "dice_icv": 0.965, "dice_mri_mask": 0.927, "ct_brain_mm2": 130, "holi_icv_mm2": 120}]}
    rows = report._ct_rows(num, sl)
    assert rows["ctSliceRows"] == [r"1 & $+8$ & frontal lobe & left putamen & $(-22.2,\,-1.7)$ & 160 & 30 & 130 & 7 \\",
                                   r"2 & $-22$ & thalamus & -- & -- & 180 & 30 & 150 & 8 \\"]
    assert rows["ctPairRows"] == [r"1 & $+8$ & 130 & 120 & 0.97 & 0.93 \\"]


def test_pet_bands():
    """One label and one colour per band (one more than the edges), and a key with white text on the dark colours."""
    cfg = {"bands": {"edges": [1.0, 1.5]}, "palette": {"bands": ["#000000", "#808080", "#ffffff"]}}
    labels, lines, m = report._pet_bands(cfg)
    assert labels == ["$<1.0$", "$1.0$--$1.5$", r"$\geq1.5$"]
    assert lines == [r"\definecolor{petBandA}{HTML}{000000}", r"\definecolor{petBandB}{HTML}{808080}", r"\definecolor{petBandC}{HTML}{FFFFFF}"]
    assert m["ColBandC"] == r"\#FFFFFF" and m["BandKey"].count(r"\colorbox") == 3
    assert r"\colorbox{petBandA}{\strut\ \color{white}$<1.0$\ }" in m["BandKey"] and r"{petBandC}{\strut\ \color{black}" in m["BandKey"]


def test_report_writes_the_tracked_numbers_tex(site):
    """With every input, the step writes the WSI, MRI, CT and PET parts. The result is the tracked report/numbers.tex."""
    said = []
    report.build(log=said.append)
    new = (site / "report" / "numbers.tex").read_text(encoding="utf-8")
    macros, colours = defined(new)
    used_macros, used_colours = used((REPORT / "report.tex").read_text(encoding="utf-8"))
    assert said == ["report/numbers.tex"] and new.startswith("% written by python -m medfig report; do not edit\n")
    assert {"vExampleY", "SectionRows", "mExampleY", "mPairRows", "ctExampleY", "ctPairRows", "petExampleY", "petShareRows"} <= macros
    want = ({"m" + name for _, name in report.MRI_COLOURS} | {"ct" + name for _, name in report.CT_COLOURS} | {"petLine"}
            | {"petBand" + report.LETTERS[i] for i in range(len(PET["palette"]["bands"]))})
    assert colours == want
    assert not sorted(used_macros - macros) and not sorted(used_colours - colours), "the step does not define a macro that report.tex uses"
    num = json.loads((site / "build" / "fig" / "mri_numbers.json").read_text(encoding="utf-8"))
    dice = [p["dice"] for p in num["pairing"]]
    assert f"\\newcommand{{\\mDiceMin}}{{{min(dice):.2f}}}\n" in new and f"\\newcommand{{\\mDiceMax}}{{{max(dice):.2f}}}\n" in new
    assert f"\\newcommand{{\\mOrientation}}{{{MRI['orientation']}}}\n" in new
    ct = json.loads((site / "build" / "fig" / "ct_numbers.json").read_text(encoding="utf-8"))
    dice = [p["dice_icv"] for p in ct["pairing"]]
    assert f"\\newcommand{{\\ctDiceIcvMin}}{{{report._half_up(min(dice))}}}\n" in new
    assert f"\\newcommand{{\\ctDiceIcvMax}}{{{report._half_up(max(dice))}}}\n" in new
    assert f"\\newcommand{{\\ctBoneHu}}{{{CT['hu']['bone']}}}\n" in new and f"\\newcommand{{\\ctOrientation}}{{{CT['orientation']}}}\n" in new
    pet = json.loads((site / "build" / "fig" / "pet_numbers.json").read_text(encoding="utf-8"))
    assert f"\\newcommand{{\\petNBands}}{{{len(PET['bands']['edges']) + 1}}}\n" in new
    rows = re.search(r"\\newcommand\{\\petShareRows\}\{([^{}]*)\}\n", new).group(1).splitlines()
    assert len(rows) == len(pet["band_share"]) == len(PET["slices"]) * len(SERIES), "one row of the PET table per slice and series"
    assert new == (REPORT / "numbers.tex").read_text(encoding="utf-8"), (
        "report/numbers.tex is not what the report step writes now. Run `python -m medfig report`")


def _pet_frame_moved(d):
    """Move the PET frame in pet_numbers.json and in every PET slices.json, so that the PET inputs agree with each other
    but not with MRI."""
    for name in ["build/fig/pet_numbers.json"] + [f"out/pet/{k}/slices.json" for k in SERIES]:
        rewrite(d / name, lambda j: j["frame_mm"].__setitem__(0, j["frame_mm"][0] + 1.0))


@pytest.mark.parametrize("damage, parts", [
    (lambda d: (d / "build/fig/numbers.json").unlink(), ("WSI input missing: ", "numbers.json", "`python -m medfig sections` and `python -m medfig figures`")),
    (lambda d: (d / "out/sections.json").unlink(), ("WSI input missing: ", "sections.json", "`python -m medfig sections` and `python -m medfig figures`")),
    (lambda d: rewrite(d / "out/sections.json", lambda j: j.pop("viewBox")), ("WSI input not usable: KeyError: 'viewBox'", "python -m medfig figures")),
    (lambda d: (d / "out/mri/slices.json").unlink(), ("MRI input missing: ", "slices.json", "Run `python -m medfig mri`, then")),
    (lambda d: (d / "build/fig/mri_numbers.json").unlink(), ("MRI input missing: ", "mri_numbers.json", "Run `python -m medfig mri`, then")),
    (lambda d: rewrite(d / "out/mri/slices.json", lambda j: j["viewBox"].append(0)), ("MRI input not usable: ValueError: ", "does not agree", "python -m medfig mri")),
    (lambda d: rewrite(d / "out/mri/slices.json", lambda j: j["slices"][0].update(roi_mm=[0.0, 0.0])), ("MRI input not usable: ValueError: ", "does not agree")),
    (lambda d: rewrite(d / "out/mri/slices.json", lambda j: j["slices"].pop()), ("MRI input not usable: ValueError: ", "does not agree")),
    (lambda d: rewrite(d / "out/mri/slices.json", lambda j: j.update(orientation="neurological")), ("MRI input not usable: ValueError: ", "orientation")),
    (lambda d: rewrite(d / "build/fig/mri_numbers.json", lambda j: j.pop("svg_kb")), ("MRI input not usable: KeyError: 'svg_kb'", "python -m medfig mri")),
    (lambda d: rewrite(d / "build/fig/mri_numbers.json", lambda j: j.update(pairing=[])), ("MRI input not usable: ", "no pairing with WSI")),
    (lambda d: (d / "out/ct/slices.json").unlink(), ("CT input missing: ", "slices.json", "Run `python -m medfig ct`, then")),
    (lambda d: (d / "build/fig/ct_numbers.json").unlink(), ("CT input missing: ", "ct_numbers.json", "Run `python -m medfig ct`, then")),
    (lambda d: rewrite(d / "out/ct/slices.json", lambda j: j["viewBox"].append(0)), ("CT input not usable: ValueError: ", "does not agree", "python -m medfig ct")),
    (lambda d: rewrite(d / "out/ct/slices.json", lambda j: j["slices"][0]["area_mm2"].update(ring=1)), ("CT input not usable: ValueError: ", "does not agree")),
    (lambda d: rewrite(d / "out/ct/slices.json", lambda j: j.update(orientation="neurological")), ("CT input not usable: ValueError: ", "orientation")),
    (lambda d: rewrite(d / "build/fig/ct_numbers.json", lambda j: j["hu"].update(bone=j["hu"]["bone"] + 1)),
     ("CT input not usable: ValueError: ", "hu thresholds", "ct extract CT_DIR")),
    (lambda d: rewrite(d / "build/fig/ct_numbers.json", lambda j: j.update(pairing=[])), ("CT input not usable: ValueError: ", "no pairing with MRI")),
    (lambda d: (d / f"out/pet/{FIRST_SERIES}/slices.json").unlink(), ("PET input missing: ", f"{FIRST_SERIES}/slices.json", "Run `python -m medfig pet`, then")),
    (lambda d: (d / "build/fig/pet_numbers.json").unlink(), ("PET input missing: ", "pet_numbers.json", "Run `python -m medfig pet`, then")),
    (lambda d: rewrite(d / f"out/pet/{SERIES[-1]}/slices.json", lambda j: j["slices"][0]["area_mm2"]["bands"].__setitem__(0, 1)),
     ("PET input not usable: ValueError: ", "does not agree", f"out/pet/{SERIES[-1]}/slices.json")),
    (lambda d: rewrite(d / f"out/pet/{FIRST_SERIES}/slices.json", lambda j: j.update(orientation="neurological")), ("PET input not usable: ValueError: ", "orientation")),
    (lambda d: rewrite(d / "build/fig/pet_numbers.json", lambda j: j["edges"].append(9.9)), ("PET input not usable: ValueError: ", "series, bands or reference")),
    (_pet_frame_moved, ("PET input not usable: ValueError: ", "is not the MRI one")),
    (lambda d: rewrite(d / "build/fig/pet_numbers.json", lambda j: j.pop("tissue_suvr")), ("PET input not usable: KeyError: 'tissue_suvr'", "python -m medfig pet")),
], ids=["no WSI numbers", "no sections.json", "sections.json without a frame", "no slices.json", "no MRI numbers", "another MRI frame",
        "another marker", "a slice less", "another MRI orientation", "MRI numbers of an older version", "no pairing",
        "no CT slices.json", "no CT numbers", "another CT frame", "another CT area", "another CT orientation", "other CT thresholds",
        "no CT pairing", "no PET slices.json", "no PET numbers", "another PET band area", "another PET orientation", "other PET bands",
        "PET frame is not the MRI frame", "PET numbers of an older version"])
def test_report_stops_and_keeps_numbers_tex(site, damage, parts):
    """A missing or unusable input stops the step with a message (exit code 1). It must not write or change report/numbers.tex."""
    damage(site)
    tex = site / "report" / "numbers.tex"
    for old in (None, "% the file of the last good run\n"):
        if old:
            tex.write_text(old, encoding="utf-8")
        with pytest.raises(SystemExit) as e:
            report.build(log=quiet)
        message = e.value.code
        assert isinstance(message, str), "a text as the exit value: Python prints it on stderr and exits with code 1"
        assert all(part in message for part in parts) and "report/numbers.tex is not changed" in message, message
        assert (tex.read_text(encoding="utf-8") if tex.exists() else None) == old, "the step wrote report/numbers.tex"


@pytest.mark.parametrize("modality, env, command", [
    ("MRI", "MEDFIG_MRI_PLANES", "python -m medfig mri extract"),
    ("CT", "MEDFIG_CT_PLANES", "python -m medfig ct extract"),
    ("PET", "MEDFIG_PET_PLANES", "python -m medfig pet extract"),
])
def test_report_stops_without_the_planes_file(site, monkeypatch, modality, env, command):
    monkeypatch.setenv(env, str(site / "none.npz"))
    with pytest.raises(SystemExit) as e:
        report.build(log=quiet)
    assert str(e.value.code).startswith(f"{modality} planes file missing: {site / 'none.npz'}. report/numbers.tex is not changed.")
    assert command in e.value.code and not (site / "report" / "numbers.tex").exists()

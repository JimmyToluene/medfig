"""Checks of the medfig pipeline. Run: make test (or python -m pytest -q)."""
import hashlib
import json
import re

import numpy as np
import pytest

from medfig import config, sections as S

CFG = config.load()


@pytest.fixture(scope="module")
def planes():
    return S.load_planes(CFG)


def test_planes_file(planes):
    data, meta = planes
    assert data.dtype == np.uint8 and data.ndim == 3
    assert data.shape[0] == len(meta["ys"])
    for s in CFG["sections"]:
        assert s["y_mm"] in meta["ys"], "every configured section must be in the planes file"


def test_masks_are_sane(planes):
    data, meta = planes
    r = CFG["raster"]
    for s in CFG["sections"]:
        a = S.plane(data, meta, s["y_mm"], r["downsample"])
        t, w, info = S.masks(a, r)
        frac_t, frac_w = t.mean(), w.sum() / t.sum()
        assert 0.1 < frac_t < 0.6, f"tissue fraction {frac_t:.2f} at y {s['y_mm']}"
        assert 0.25 < frac_w < 0.75, f"white-matter fraction {frac_w:.2f} at y {s['y_mm']}"
        assert 150 < info["threshold"] < 230


def test_floor_matters_for_cerebellum(planes):
    """Without the floor, the dark granular layer pulls Otsu of the cerebellar section far down."""
    data, meta = planes
    r = CFG["raster"]
    s = min(CFG["sections"], key=lambda s: s["y_mm"])
    a = S.plane(data, meta, s["y_mm"], r["downsample"])
    t = S.tissue_mask(a, r)
    m = S.local_mean(a, t, r["local_mean_sigma"])
    from skimage.filters import threshold_otsu
    assert threshold_otsu(m[t & (m > r["otsu_floor"])]) - threshold_otsu(m[t]) > 20


def test_build_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "OUT", tmp_path)
    a = S.build(CFG, log=lambda *_: None)
    h1 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("section_*.svg")}
    S.build(CFG, log=lambda *_: None)
    h2 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("section_*.svg")}
    assert h1 == h2 and len(h1) == len(CFG["sections"])
    vb = a["viewBox"]
    for s in a["sections"]:
        x, y, w, h = s["rect"]
        assert 0 <= x and x + w <= vb[2] and 0 <= y and y + h <= vb[3], "block marker outside the frame"
        svg = (tmp_path / s["file"]).read_text()
        assert svg.count("<path") == 3 and re.search(r'viewBox="0 0 \d+ \d+"', svg)
        assert "CC BY-NC-SA 4.0" in svg


def test_reference_paths():
    """The SVG path data equal the reference copies in tests/reference (published on the lab landing page, 2026-10-13)."""
    ref = config.ROOT / "tests" / "reference"
    if not ref.exists():
        pytest.skip("no reference copies")
    for p in sorted(ref.glob("section_*.svg")):
        out = config.OUT / p.name
        if not out.exists():
            pytest.skip("run `make sections` first")
        d = lambda s: re.findall(r' d="([^"]*)"', s)  # noqa: E731
        assert d(out.read_text()) == d(p.read_text()), p.name

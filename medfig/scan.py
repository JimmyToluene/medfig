"""Shared code of the scanner modalities (MRI, CT, PET): orientation, frame, markers, SVG text, scan thumbnails, stack page.

Every scanner modality draws coronal planes of MNI space on the 0.5 mm grid of the holiAtlas planes file, with one frame
unit = vector.unit_mm (the WSI unit), so the figures of the modalities pair. The markers of every scanner modality are
centroids of labels of the holiAtlas structure atlas.

Orientation (key "orientation" of the configuration file of a modality):
  radiological  the left of the patient (x < 0) is on the right of the image, as in clinical viewers
  neurological  the left of the patient is on the left of the image, as in the WSI figure
In both, rows run superior to inferior. A grid is (x of column 0, z of row 0, pixel size, x step from one column to the
next): the x step is minus the pixel size in the radiological orientation. All mm values in the output files are MNI mm.

`out` is the output directory of one series of slices (for example out/mri/). It holds slices.json and slice_k.svg.
"""
import json
import re

import numpy as np

from . import render, stack

ORIENTATIONS = {"radiological": -1, "neurological": 1}      # sign of the x step from one column to the next
FONT = 'font-family="Lora, Georgia, serif"'
CREDIT = "#666"     # grey of the credit line under the figures of the SVG slices and of the thumbnails


def x_sign(cfg):
    """+1 or -1: the sign of the x step from one image column to the next. Raises ValueError for an unknown orientation."""
    o = cfg.get("orientation")
    if o not in ORIENTATIONS:
        raise ValueError(f"{cfg.get('_path', 'configuration')}: orientation {o!r}: use one of {sorted(ORIENTATIONS)}")
    return ORIENTATIONS[o]


def orient(a, cfg):
    """A plane stored (z, x) with z and x ascending -> rows superior to inferior, columns in the configured orientation."""
    a = a[::-1]
    return a[:, ::-1] if x_sign(cfg) < 0 else a


def grid(meta, cfg):
    """The grid of a plane after orient(). meta holds origin_mm (x, y, z of voxel 0), dims (nx, ny, nz) and vs."""
    vs, (nx, _, nz), o = meta["vs"], meta["dims"], meta["origin_mm"]
    s = x_sign(cfg)
    return (o[0] if s > 0 else o[0] + vs * (nx - 1)), o[2] + vs * (nz - 1), vs, s * vs


def to_mm(row, col, g):
    """(x, z) in mm of a pixel position (row, column) on grid g."""
    return g[0] + col * g[3], g[1] - row * g[2]


def frame(masks, v, scale):
    """The frame of a series: the bounding box of the union of the masks, padded by pad_units (rounded to whole pixels).

    Returns (ox, oy, vw, vh): the frame origin in pixels and the viewBox size in units. Raises ValueError for empty masks."""
    rr, cc = np.where(np.any(masks, axis=0))
    if not rr.size:
        raise ValueError("the masks of the configured planes are empty")
    pad = int(round(v["pad_units"] / scale))
    oy, ox = int(rr.min()) - pad, int(cc.min()) - pad
    return ox, oy, int(round((cc.max() - cc.min() + 2 * pad) * scale)), int(round((rr.max() - rr.min() + 2 * pad) * scale))


def frame_mm(ox, oy, g):
    """(x, z) in mm of the top left corner of the frame: u = s (x - x_left) / unit, v = (z_top - z) / unit, s = x_sign."""
    return [round(g[0] + ox * g[3], 3), round(g[1] - oy * g[2], 3)]


def roi_centre(structure, labels, r):
    """Centroid (row, col) of the structure labels in the plane. Fewer than roi_min_px pixels raise ValueError."""
    m = np.isin(structure, labels)
    if m.sum() < r["roi_min_px"]:
        raise ValueError(f"structure labels {labels}: {int(m.sum())} px in this plane, fewer than roi_min_px")
    rr, cc = np.where(m)
    return float(rr.mean()), float(cc.mean())


def slice_centre(s, structure, r):
    """Marker centre of a configured slice, or None for a slice without roi_labels. An error names the slice."""
    if not s.get("roi_labels"):
        return None
    try:
        return roi_centre(structure, s["roi_labels"], r)
    except ValueError as e:
        raise ValueError(f"slice {s['k']} (y = {s['y_mm']} mm): {e}") from None


def item(s, centre, ox, oy, g, scale, side, f, areas):
    """The record of one slice in slices.json. side is the side of the marker square in units."""
    it = {**{k: s[k] for k in ("k", "y_mm", "label")}, "roi": s.get("roi") if centre else None, "roi_mm": None, "rect": None,
          "file": f, "area_mm2": areas}
    if centre:
        cr, cc = centre
        it["roi_mm"] = [round(v, 1) for v in to_mm(cr, cc, g)]
        it["rect"] = [round((cc - ox) * scale - side / 2, 1), round((cr - oy) * scale - side / 2, 1), side, side]
    return it


def fill(d, colour):
    return f'<path d="{d}" fill="{colour}" fill-rule="evenodd"/>'


def stroke(d, colour, width):
    return f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}" stroke-linejoin="round"/>'


def svg(vw, vh, label, credit, body):
    """One SVG slice: the viewBox, the text alternative, the credit line of the source in a comment, and the body."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw} {vh}" role="img" aria-label="{label}"><!-- Drawn from {credit} -->'
            + body + "</svg>")


def read_slices(out):
    return json.loads((out / "slices.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------------- thumbnails
def _inner(out, sec):
    s = (out / sec["file"]).read_text(encoding="utf-8")
    s = re.sub(r"^<svg[^>]*>|</svg>\s*$", "", s.strip())
    return re.sub(r"<!--.*?-->", "", s, flags=re.S)


def _card(x, y, w, h, lw, rx, sw, pal):
    """Rounded card with a label strip of width lw at the left. sw is the stroke width."""
    e = sw / 2
    a, t, b = x + e, y + e, y + h - e
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{pal["card"]}" stroke="{pal["accent"]}" stroke-width="{sw}"/>'
            f'<path d="M{x+rx} {t:g} H{x+lw} V{b:g} H{x+rx} Q{a:g} {b:g} {a:g} {y+h-rx} V{y+rx} Q{a:g} {t:g} {x+rx} {t:g} Z" fill="{pal["label"]}"/>'
            f'<line x1="{x+lw}" y1="{t:g}" x2="{x+lw}" y2="{b:g}" stroke="{pal["accent"]}" stroke-width="{sw * .72:.1f}"/>')


def _screen(sec, vb, tx, ty, ts, rx, m, pal):
    """Dark square of side ts at (tx, ty). The whole slice is centred in it with margin m."""
    return (f'<rect x="{tx:g}" y="{ty:g}" width="{ts}" height="{ts}" rx="{rx}" fill="{pal["screen"]}"/>'
            f'<svg x="{tx + m:g}" y="{ty + m:g}" width="{ts - 2 * m:g}" height="{ts - 2 * m:g}" viewBox="{vb}"><use href="#s{sec["k"]}"/></svg>')


def icons(cfg, out, noun, log=print):
    """Scan thumbnails made from the slices, so scan k shows the whole slice k on a dark screen.

    out/scan_one.svg    one scan (126 x 48 px), slice 1
    out/scan_strip.svg  scans 1 to N (244 x 36 px for N = 3) and two card edges behind the last scan
    An inner <svg> with the viewBox of the slices draws slice k in the screen of scan k. noun names the modality ("MRI").
    """
    pal = cfg["palette"]
    P, EDGE, CARD, GREY = pal["accent"], pal["card_edge"], pal["card"], pal["label_bar"]
    meta = read_slices(out)
    secs = meta["slices"]
    vb = " ".join(str(v) for v in meta["viewBox"])
    credit = f"<!-- Slices drawn from {meta['license']} -->"

    SW, SH, LW, GAP = 74, 26, 16, 6
    n = len(secs)
    W, H = n * SW + (n - 1) * GAP + 10, SH + 10
    defs = [f'<g id="s{s["k"]}">{_inner(out, s)}</g>' for s in secs]
    body = []
    for d in (2, 1):
        x = (n - 1) * (SW + GAP) + 4 * d
        body.append(f'<rect x="{x}" y="{1 + 4 * d}" width="{SW}" height="{SH}" rx="4" fill="{CARD}" stroke="{EDGE}" stroke-width="1.2"/>')
    for i, s in enumerate(secs):
        x, y = i * (SW + GAP) + 1, 1
        ts = SH - 4
        body.append(_card(x, y, SW, SH, LW, 4, 1.5, pal))
        body.append(f'<circle cx="{x+LW/2:g}" cy="{y+SH/2:g}" r="5" fill="{P}"/><text x="{x+LW/2:g}" y="{y+SH/2+2.6:g}" text-anchor="middle" '
                    f'font-size="{7 if s["k"] < 10 else 5.5}" font-weight="700" fill="{pal["accent_ink"]}" {FONT}>{s["k"]}</text>')
        body.append(_screen(s, vb, x + LW + (SW - LW - ts) / 2, y + 2, ts, 3, 1, pal))
    (out / "scan_strip.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-label="Attached {noun} slices, numbered 1 to {n}, with more slices behind">{credit}<defs>{"".join(defs)}</defs>{"".join(body)}</svg>',
        encoding="utf-8")

    SW1, SH1, LW1 = 120, 42, 26
    W1, H1 = SW1 + 6, SH1 + 6
    ts = SH1 - 6
    (out / "scan_one.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W1} {H1}" width="{W1}" height="{H1}" role="img" aria-label="One attached {noun} slice">'
        f'{credit}<defs><g id="s{secs[0]["k"]}">{_inner(out, secs[0])}</g></defs>'
        + _card(3, 3, SW1, SH1, LW1, 6, 1.8, pal) +
        f'<rect x="9" y="12" width="14" height="3" rx="1.5" fill="{EDGE}"/><rect x="9" y="19" width="10" height="3" rx="1.5" fill="{GREY}"/>'
        + _screen(secs[0], vb, 3 + LW1 + (SW1 - LW1 - ts) / 2, 3 + 3, ts, 4, 1.5, pal) + "</svg>", encoding="utf-8")
    log(f"scan_strip.svg {W} x {H}, scan_one.svg {W1} x {H1}")


# ---------------------------------------------------------------------------------------------------------- stack page
def web(cfg, out, title, noun, what, extra="", log=print):
    """out/stack.html (standalone page with a Slice / Volume switch), out/stack_snippet.html and out/stack.css.

    title is the page title, noun names the modality in the text alternatives, `what` ends the note ("of the
    skull-stripped brain"), extra is HTML put after the note (for example a colour key)."""
    meta = read_slices(out)
    (out / "stack.css").write_text(stack.css(cfg, len(cfg["slices"])))
    (out / "stack_snippet.html").write_text(stack.snippet(meta, key="slices") + "\n")
    n = len(meta["slices"])
    ys = ", ".join(f"{s['y_mm']:+d}" for s in meta["slices"])
    side = {"radiological": "the left of the patient is on the right of the image",
            "neurological": "the left of the patient is on the left of the image"}[cfg["orientation"]]
    (out / "stack.html").write_text(stack.page(
        cfg, meta, "slices", title=title, first="Slice", second="Volume",
        one="scan_one.svg", one_alt=f"One attached {noun} slice", strip="scan_strip.svg", strip_alt=f"Attached {noun} slices 1 to {n}",
        note=f"Coronal slices y = {ys} mm {what}, drawn from {meta['license']}; {side}. The slices and markers are a schematic.{extra}"),
        encoding="utf-8")
    log(f"{out.name}/stack.html, stack.css, stack_snippet.html")


# ---------------------------------------------------------------------------------------------------------- captures
def target(out, frames, prefix):
    """The capture target of medfig/render.py for one series."""
    return {"root": out, "frames": frames, "prefix": prefix, "states": ("slice", "volume")}


def _width(out, name):
    """Width attribute of an SVG file."""
    return int(re.search(r'<svg[^>]*\swidth="(\d+)"', (out / name).read_text(encoding="utf-8")).group(1))


def _caption(s):
    return f"y = {s['y_mm']:+d} mm" + (f" · marker {s['k']}: {s['roi']}" if s.get("roi") else "")


def credit_line(text, px, pad):
    """The visible credit line under a figure. px is the font size, pad the padding at the sides and below, both in CSS px."""
    return f"<p style='margin:0;padding:0 {pad}px {pad}px;font:{px}px sans-serif;color:{CREDIT}'>Source: {text}</p>"


def capture_figs(p, url, log, cfg, out, fig, svgs_png, icons_png, zoom=4, extra=""):
    """Two report figures that need a browser: the SVG slices with their markers, and the scan thumbnails (zoom x).

    Each shows the credit line of the source under the drawing. extra is HTML put after the slices (for example a key)."""
    fig.mkdir(parents=True, exist_ok=True)
    pal = cfg["palette"]
    meta = read_slices(out)
    n = len(meta["slices"])
    head = "<meta charset='utf-8'>"
    cells = render.marker_cells(meta["slices"], meta["viewBox"], _caption, pal["accent"])
    ics = [(f, zoom * _width(out, f)) for f in ("scan_one.svg", "scan_strip.svg")]
    a = (f"{head}<body style='margin:0;background:{pal['page']};font:14px sans-serif'>"
         f"<div style='display:flex;gap:16px;padding:12px;align-items:flex-start'>{cells}{extra}</div>{credit_line(meta['license'], 12, 12)}</body>")
    b = (f"{head}<body style='margin:0;background:{render.ICON_BG}'><div style='padding:16px;display:flex;gap:24px;align-items:center'>"
         + "".join(f"<img src='{f}' style='width:{w}px'>" for f, w in ics) + f"</div>{credit_line(meta['license'], 14, 16)}</body>")
    br = render._launch(p)
    render.shoot(br, url, out, "_svgs.html", a, fig / svgs_png, (416 * n + 12 + (220 if extra else 0), 460), 1.5)
    render.shoot(br, url, out, "_icons.html", b, fig / icons_png, (sum(w for _, w in ics) + 80, 240), 1)
    br.close()
    log(f"fig/{svgs_png}, fig/{icons_png}")

"""Step S8: slide icons cut from the sections at each block marker, so slide k shows the tissue of block k.

Usage: python -m medfig.icons
Input:  out/section_{k}.svg, out/sections.json
Output: out/slide_one.svg    one slide (126 x 48 px), block 1
        out/slide_strip.svg  slides 1 to N (244 x 36 px for N = 3) and two deck edges behind the last slide

The tile of slide k is an inner <svg> whose viewBox is the block-k rectangle of section k.
"""
import json
import re
import sys

from .config import OUT, load

FONT = 'font-family="Lora, Georgia, serif"'


def _inner(k):
    s = (OUT / f"section_{k}.svg").read_text()
    s = re.sub(r"^<svg[^>]*>|</svg>\s*$", "", s.strip())
    return re.sub(r"<!--.*?-->", "", s)


def _tile(sec, tx, ty, ts, rx):
    k = sec["k"]
    bx, by, bw, bh = sec["rect"]
    return (f'<clipPath id="k{k}"><rect x="{tx}" y="{ty}" width="{ts}" height="{ts}" rx="{rx}"/></clipPath>',
            f'<g clip-path="url(#k{k})"><rect x="{tx}" y="{ty}" width="{ts}" height="{ts}" fill="#fff"/>'
            f'<svg x="{tx}" y="{ty}" width="{ts}" height="{ts}" viewBox="{bx} {by} {bw} {bh}"><use href="#s{k}"/></svg></g>'
            f'<rect x="{tx}" y="{ty}" width="{ts}" height="{ts}" rx="{rx}" fill="none" stroke="#b9c6e0" stroke-width=".8"/>')


def build(cfg=None, log=print):
    cfg = cfg or load()
    pal = cfg["palette"]
    P, LAV, GLASS, LABEL = pal["accent"], pal["glass_edge"], pal["glass"], pal["label"]
    meta = json.loads((OUT / "sections.json").read_text())
    secs = meta["sections"]
    credit = f"<!-- Tissue drawn from {meta['license']} -->"

    SW, SH, LW, GAP = 74, 26, 16, 6
    n = len(secs)
    W, H = n * SW + (n - 1) * GAP + 10, SH + 10
    defs = [f'<g id="s{s["k"]}">{_inner(s["k"])}</g>' for s in secs]
    body = []
    for d in (2, 1):
        x = (n - 1) * (SW + GAP) + 4 * d
        body.append(f'<rect x="{x}" y="{1 + 4 * d}" width="{SW}" height="{SH}" rx="4" fill="{GLASS}" stroke="{LAV}" stroke-width="1.2"/>')
    for i, s in enumerate(secs):
        x, y = i * (SW + GAP) + 1, 1
        ts = SH - 6
        tx, ty = x + LW + (SW - LW - ts) / 2, y + 3
        body.append(f'<rect x="{x}" y="{y}" width="{SW}" height="{SH}" rx="4" fill="{GLASS}" stroke="{P}" stroke-width="1.5"/>')
        body.append(f'<path d="M{x+4} {y+.75} H{x+LW} V{y+SH-.75} H{x+4} Q{x+.75} {y+SH-.75} {x+.75} {y+SH-4} V{y+4} Q{x+.75} {y+.75} {x+4} {y+.75} Z" fill="{LABEL}"/>')
        body.append(f'<line x1="{x+LW}" y1="{y+.75}" x2="{x+LW}" y2="{y+SH-.75}" stroke="{P}" stroke-width="1.1"/>')
        body.append(f'<circle cx="{x+LW/2}" cy="{y+SH/2}" r="5" fill="{P}"/><text x="{x+LW/2}" y="{y+SH/2+2.6}" text-anchor="middle" '
                    f'font-size="7" font-weight="700" fill="#fff" {FONT}>{s["k"]}</text>')
        c, g = _tile(s, tx, ty, ts, 3)
        defs.append(c)
        body.append(g)
    (OUT / "slide_strip.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-label="Attached slides, numbered 1 to {n}, with more slides behind">{credit}<defs>{"".join(defs)}</defs>{"".join(body)}</svg>')

    SW1, SH1, LW1 = 120, 42, 26
    W1, H1 = SW1 + 6, SH1 + 6
    ts = SH1 - 8
    tx, ty = 3 + LW1 + (SW1 - LW1 - ts) / 2, 3 + 4
    c, g = _tile(secs[0], tx, ty, ts, 4)
    (OUT / "slide_one.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W1} {H1}" width="{W1}" height="{H1}" role="img" aria-label="One attached slide">{credit}'
        f'<defs><g id="s{secs[0]["k"]}">{_inner(secs[0]["k"])}</g>{c}</defs>'
        f'<rect x="3" y="3" width="{SW1}" height="{SH1}" rx="6" fill="{GLASS}" stroke="{P}" stroke-width="1.8"/>'
        f'<path d="M9 4 H{3+LW1} V{3+SH1-1} H9 Q4 {3+SH1-1} 4 {3+SH1-6} V9 Q4 4 9 4 Z" fill="{LABEL}"/>'
        f'<line x1="{3+LW1}" y1="4" x2="{3+LW1}" y2="{3+SH1-1}" stroke="{P}" stroke-width="1.3"/>'
        f'<rect x="9" y="12" width="14" height="3" rx="1.5" fill="#b9acdf"/><rect x="9" y="19" width="10" height="3" rx="1.5" fill="#c9daf1"/>'
        f'{g}</svg>')
    log(f"slide_strip.svg {W} x {H}, slide_one.svg {W1} x {H1}")


if __name__ == "__main__":
    build()
    sys.exit(0)

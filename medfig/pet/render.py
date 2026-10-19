"""Step P8, render: capture the stack page of each PET series (medfig/render.py) and draw figures P6 and P7 of the report.

Usage: python -m medfig.pet.render [--gif]
Output: build/frames/pet/<series>/*.png, build/fig/pet_<series>_frame_sheet.png, pet_<series>_end_slice.png,
        pet_<series>_end_volume.png, build/pet_<series>_stack.gif (with --gif)
        build/fig/pet_step_6_pair.png   figure P6: the SVG slices of every series, one row each, with the colour key
        build/fig/pet_step_7_icons.png  figure P7: the scan thumbnails of the first series, 4x
Needs Playwright with Chromium (see medfig/render.py).
"""
import sys

from .. import render, scan
from ..config import BUILD, load
from .slices import PET_OUT


def report_figs(p, url, log):
    """Figure P6 (both series, one row each, no markers, with the colour key) and figure P7 (thumbnails of the first series).
    P6 needs the files of every series, so it is captured from a server of out/pet/."""
    cfg = load("pet")
    fig, pal = BUILD / "fig", cfg["palette"]
    first = PET_OUT / cfg["series"][0]["key"]
    rows = []
    for se in cfg["series"]:
        meta = scan.read_slices(PET_OUT / se["key"])
        cells = render.marker_cells([{**s, "file": f"{se['key']}/{s['file']}", "rect": None} for s in meta["slices"]], meta["viewBox"],
                                    lambda s: f"y = {s['y_mm']:+d} mm", pal["accent"])
        rows.append(f"<div style='display:flex;gap:16px;align-items:center'><div style='width:150px;font-weight:bold'>{se['name']}</div>{cells}</div>")
    html = (f"<meta charset='utf-8'><body style='margin:0;background:{pal['page']};font:14px sans-serif'><div style='padding:12px;display:grid;gap:10px'>"
            + "".join(rows) + f"<div style='padding-left:166px'><img src='{cfg['series'][0]['key']}/key.svg' style='width:260px'></div></div>"
            + scan.credit_line(f"{cfg['source']['license']}; outline and markers: {cfg['source']['outline_credit']}", 12, 12) + "</body>")
    srv, url_all = render._serve(PET_OUT)
    try:
        b = render._launch(p)
        render.shoot(b, url_all, PET_OUT, "_pair.html", html, fig / "pet_step_6_pair.png", (166 + 3 * 416 + 24, 600), 1.5)
        b.close()
    finally:
        srv.shutdown()
        srv.server_close()
    scan.capture_figs(p, url, log, cfg, first, fig, "_pet_svgs.png", "pet_step_7_icons.png")
    (fig / "_pet_svgs.png").unlink()
    log("fig/pet_step_6_pair.png")


def main(argv=None):
    cfg = load("pet")
    a = [a for a in (argv if argv is not None else sys.argv[1:])]
    for i, se in enumerate(cfg["series"]):
        tg = scan.target(PET_OUT / se["key"], BUILD / "frames" / "pet" / se["key"], f"pet_{se['key']}_")
        render.run(tg, report_figs if i == 0 else (lambda p, url, log: None), a)


if __name__ == "__main__":
    sys.exit(main())

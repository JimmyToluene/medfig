"""Capture the 3D stack: frames, a frame sheet, end-state PNGs with a transparent background, and a GIF.

Usage: python -m medfig.render [--gif]
Input:  out/stack.html and the files it loads (served over local HTTP, so the browser loads the SVGs)
Output: build/frames/desktop_*.png, phone_*.png   frames at fixed times after each switch
        build/fig/frame_sheet.png                slide, case at 0.15/0.4/0.7/1.4 s, back to slide
        build/fig/end_slide.png, end_case.png    stack only, transparent, 2x
        build/stack.gif                          (with --gif) one full switch and back, 20 fps
        build/fig/step_7_svgs.png                the three SVG sections with their block markers
        build/fig/step_8_icons.png               the slide icons, 4x

Needs Playwright with Chromium: pip install playwright && python -m playwright install chromium
(aarch64 Linux is supported). Set MEDFIG_CHROMIUM to use another Chromium binary.
"""
import argparse
import functools
import http.server
import os
import sys
import threading

from PIL import Image, ImageDraw

from .config import BUILD, OUT

TIMES = [150, 400, 700, 1400]


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def _serve():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(OUT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/stack.html"


def _launch(p):
    exe = os.environ.get("MEDFIG_CHROMIUM")
    return p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()


def _box(pg):
    return pg.evaluate("(()=>{const b=document.querySelector('.hero').getBoundingClientRect();"
                       "return {x:Math.max(0,b.left-40),y:Math.max(0,b.top-60),width:Math.min(innerWidth,b.width+80),height:b.height+120}})()")


def frames(p, url, log):
    d = BUILD / "frames"
    d.mkdir(parents=True, exist_ok=True)
    b = _launch(p)
    out = {}
    for tag, vp in (("desktop", {"width": 900, "height": 640}), ("phone", {"width": 390, "height": 700})):
        pg = b.new_page(viewport=vp, device_scale_factor=2)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url)
        pg.wait_for_timeout(600)
        clip = _box(pg)
        names = []
        pg.screenshot(path=str(d / f"{tag}_slide.png"), clip=clip); names.append("slide")
        pg.click("button[data-mode=case]")
        last = 0
        for t in TIMES:
            pg.wait_for_timeout(t - last); last = t
            pg.screenshot(path=str(d / f"{tag}_case_{t}.png"), clip=clip); names.append(f"case_{t}")
        pg.click("button[data-mode=slide]")
        pg.wait_for_timeout(1400)
        pg.screenshot(path=str(d / f"{tag}_back.png"), clip=clip); names.append("back")
        sw = pg.evaluate("document.documentElement.scrollWidth")
        log(f"{tag}: scrollWidth {sw} for {vp['width']}, page errors {errs}")
        out[tag] = (names, sw, errs)
        pg.close()
    b.close()
    names = out["desktop"][0]
    ims = [Image.open(d / f"desktop_{n}.png") for n in names]
    w, h = ims[0].size
    s = 0.5
    W, H = int(w * s), int(h * s)
    sheet = Image.new("RGB", (3 * W + 40, 2 * H + 70), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (im, n) in enumerate(zip(ims, names)):
        x, y = 10 + (i % 3) * (W + 10), 25 + (i // 3) * (H + 25)
        sheet.paste(im.resize((W, H)), (x, y))
        dr.text((x, y - 16), n.replace("_", " ") + (" ms" if n.startswith("case") else ""), fill="black")
    (BUILD / "fig").mkdir(parents=True, exist_ok=True)
    sheet.save(BUILD / "fig" / "frame_sheet.png")
    log("fig/frame_sheet.png")
    return out


def end_states(p, url, log):
    b = _launch(p)
    pg = b.new_page(viewport={"width": 900, "height": 640}, device_scale_factor=2)
    pg.goto(url)
    pg.wait_for_timeout(600)
    pg.add_style_tag(content="html,body{background:transparent!important}.mode,.att,small{visibility:hidden!important}")
    box = pg.evaluate("(()=>{const b=document.querySelector('.brainbox').getBoundingClientRect();return {x:b.left-90,y:b.top-40,width:b.width*1.9,height:b.height+80}})()")
    pg.screenshot(path=str(BUILD / "fig" / "end_slide.png"), clip=box, omit_background=True)
    pg.evaluate("document.querySelector('button[data-mode=case]').click()")      # the switch is hidden here
    pg.wait_for_timeout(1600)
    pg.screenshot(path=str(BUILD / "fig" / "end_case.png"), clip=box, omit_background=True)
    b.close()
    log("fig/end_slide.png, fig/end_case.png")


def gif(p, url, log, fps=20):
    b = _launch(p)
    pg = b.new_page(viewport={"width": 900, "height": 640})
    pg.goto(url)
    pg.wait_for_timeout(600)
    clip = _box(pg)
    shots = []
    for mode, dur in (("case", 1600), ("slide", 1400)):
        pg.click(f"button[data-mode={mode}]")
        for _ in range(int(dur / 1000 * fps)):
            shots.append(Image.open(__import__("io").BytesIO(pg.screenshot(clip=clip))).convert("P", palette=Image.ADAPTIVE))
            pg.wait_for_timeout(1000 / fps)
    b.close()
    shots[0].save(BUILD / "stack.gif", save_all=True, append_images=shots[1:], duration=int(1000 / fps), loop=0)
    log(f"stack.gif, {len(shots)} frames (timing is approximate: each screenshot takes time)")


def report_figs(p, url, log):
    """Figures S7 and S8 of the report: they need a browser to draw the SVG files."""
    import json
    fig = BUILD / "fig"
    fig.mkdir(parents=True, exist_ok=True)
    meta = json.loads((OUT / "sections.json").read_text())
    vb = meta["viewBox"]
    cells = "".join(
        f"<figure><div style='position:relative'><img src='{s['file']}' style='width:100%'>"
        f"<svg viewBox='0 0 {vb[2]} {vb[3]}' style='position:absolute;left:0;top:0;width:100%'><rect x='{s['rect'][0]}' y='{s['rect'][1]}' "
        f"width='{s['rect'][2]}' height='{s['rect'][3]}' rx='10' fill='none' stroke='#5b43a8' stroke-width='5' stroke-dasharray='12 8'/></svg></div>"
        f"<figcaption>y = {s['y_mm']:+d} mm · block {s['k']}: {s['block']}</figcaption></figure>" for s in meta["sections"])
    (OUT / "_s7.html").write_text("<body style='margin:0;background:#faf9f5;font:14px sans-serif'><div style='display:flex;gap:16px;padding:12px'>"
                                  + cells.replace("<figure>", "<figure style='margin:0;width:400px'>") + "</div></body>")
    (OUT / "_s8.html").write_text("<body style='margin:0;background:#e6eff9'><div style='padding:16px;display:flex;gap:24px;align-items:center'>"
                                  "<img src='slide_one.svg' style='width:504px'><img src='slide_strip.svg' style='width:976px'></div></body>")
    b = _launch(p)
    base = url.rsplit("/", 1)[0]
    pg = b.new_page(viewport={"width": 1260, "height": 460}, device_scale_factor=1.5)
    pg.goto(base + "/_s7.html"); pg.wait_for_timeout(400)
    pg.screenshot(path=str(fig / "step_7_svgs.png"), full_page=True)
    pg = b.new_page(viewport={"width": 1560, "height": 240}, device_scale_factor=1)
    pg.goto(base + "/_s8.html"); pg.wait_for_timeout(400)
    pg.screenshot(path=str(fig / "step_8_icons.png"), full_page=True)
    b.close()
    for f in ("_s7.html", "_s8.html"):
        (OUT / f).unlink()
    log("fig/step_7_svgs.png, fig/step_8_icons.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args(argv)
    from playwright.sync_api import sync_playwright
    BUILD.mkdir(parents=True, exist_ok=True)
    srv, url = _serve()
    try:
        with sync_playwright() as p:
            frames(p, url, print)
            end_states(p, url, print)
            report_figs(p, url, print)
            if a.gif:
                gif(p, url, print)
    finally:
        srv.shutdown()


if __name__ == "__main__":
    sys.exit(main())

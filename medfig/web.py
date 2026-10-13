"""Step S9: the oblique 3D stack as a standalone web page, and the snippets to put it in another page.

Usage: python -m medfig.web
Input:  out/section_{k}.svg, out/sections.json, out/slide_*.svg, config stack_3d
Output: out/stack.html        standalone page: Slide-level / Case-level switch, the stack and the slide icons
        out/stack_snippet.html  the .brainbox markup only (for a host page)
        out/stack.css           the CSS only (for a host page)

Slide-level shows section 1 with its block marker. Case-level turns the stack to an oblique view, pushes the
back sections in from behind one after the other, then shows each block number. All motion is CSS transitions,
so the switch back is the same motion reversed. prefers-reduced-motion turns the motion off.
"""
import json
import sys

from .config import OUT, load


def css(cfg):
    st, pal = cfg["stack_3d"], cfg["palette"]
    c, ph = st["case"], st["phone"]
    n = len(cfg["sections"])
    rules = [
        f":root{{--accent:{pal['accent']};--accent-ink:{pal['accent_ink']};--ease:cubic-bezier(.2,.7,.2,1)}}",
        f".brainbox{{position:relative;width:100%;perspective:{st['perspective_px']}px;perspective-origin:{st['perspective_origin']}}}",
        f".stack{{position:relative;display:grid;transform-style:preserve-3d;transform-origin:{st['transform_origin']};transition:transform .75s var(--ease) .05s}}",
        ".csec{grid-area:1/1;display:grid;transform-style:preserve-3d;transition:transform .7s var(--ease),opacity .4s ease,filter .4s ease}",
        ".csec>*{grid-area:1/1;width:100%;height:auto}",
        ".mk{transform-box:fill-box;transform-origin:center;transition:opacity .38s var(--ease),transform .5s var(--ease)}",
        ".bd{opacity:0;transform:scale(.6);transition:opacity .3s var(--ease),transform .45s var(--ease)}",
        f".is-case .stack{{transform:translateX({c['translate_x']}) rotateY({c['rotate_y_deg']}deg) rotateX({c['rotate_x_deg']}deg) scale({c['scale']});transition-delay:0s}}",
        ".is-case .mk,.is-case .bd{opacity:1;transform:none}",
    ]
    back = ",".join(f".cs{k}" for k in range(2, n + 1))
    if back:
        rules.append(f"{back}{{opacity:0;transform:translateZ({st['start_depth_px']}px)}}")
        rules.append(",".join(f".cs{k} .mk" for k in range(2, n + 1)) + "{opacity:0;transform:scale(.82)}")
    for k in range(2, n + 1):
        d = st["section_delay_s"][k - 1] if k - 1 < len(st["section_delay_s"]) else 0.12 * (k - 1)
        f = st["back_filter"][k - 1] if k - 1 < len(st["back_filter"]) else "none"
        rules.append(f".cs{k}{{transition-delay:{max(0, round(0.06 * (n - k), 2))}s}}")
        rules.append(f".is-case .cs{k}{{opacity:1;transform:translateZ({-c['depth_step_px'] * (k - 1)}px);filter:{f};transition-delay:{d}s}}")
    for k in range(1, n + 1):
        d = st["marker_delay_s"][k - 1] if k - 1 < len(st["marker_delay_s"]) else 0.35 + 0.2 * (k - 1)
        sel = f".is-case .cs{k} .bd" if k == 1 else f".is-case .cs{k} .mk,.is-case .cs{k} .bd"
        rules.append(f"{sel}{{transition-delay:{d}s}}")
    phone = [f".is-case .stack{{transform:translateX({ph['translate_x']}) rotateY({ph['rotate_y_deg']}deg) rotateX({c['rotate_x_deg']}deg) scale({ph['scale']})}}"]
    phone += [f".is-case .cs{k}{{transform:translateZ({-ph['depth_step_px'] * (k - 1)}px)}}" for k in range(2, n + 1)]
    rules.append(f"@media (max-width:{ph['max_width_px']}px){{{''.join(phone)}}}")
    rules.append("@media (prefers-reduced-motion:reduce){*,*::before,*::after{transition:none!important;animation:none!important}}")
    return "\n".join(rules) + "\n"


def snippet(meta, prefix=""):
    vb = " ".join(str(v) for v in meta["viewBox"])
    parts = []
    for s in sorted(meta["sections"], key=lambda s: -s["k"]):           # back to front
        x, y, w, h = s["rect"]
        k = s["k"]
        parts.append(
            f'<div class="csec cs{k}"><img src="{prefix}{s["file"]}" alt=""><svg viewBox="{vb}" aria-hidden="true">'
            f'<g class="mk"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="none" stroke="var(--accent)" stroke-width="5" stroke-dasharray="12 8"/></g>'
            f'<g class="bd"><circle cx="{x}" cy="{y}" r="21" fill="var(--accent)"/><text x="{x}" y="{round(y + 10, 1)}" text-anchor="middle" '
            f'font-size="28" font-weight="700" fill="var(--accent-ink)" font-family="Lora, Georgia, serif">{k}</text></g></svg></div>')
    return '<div class="brainbox"><div class="stack">' + "".join(parts) + "</div></div>"


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coronal stack</title>
<style>
body{{margin:0;background:{page};font:15px Lora,Georgia,serif;color:#141413}}
main{{max-width:760px;margin:0 auto;padding:48px 24px}}
.mode{{display:inline-flex;gap:24px;border-bottom:1px solid #e8e6dc;margin-bottom:24px}}
.mode button{{all:unset;cursor:pointer;padding:4px 0 7px;color:#73726c}}
.mode button[aria-pressed=true]{{color:#141413;box-shadow:inset 0 -2px #141413}}
.hero{{position:relative;display:grid;grid-template-columns:45.6% 1fr;align-items:end;gap:0}}
.att{{display:grid;margin-left:-14%;padding:12px;background:#fff;border-radius:8px;box-shadow:0 3px 3px rgba(0,0,0,.25);width:max-content;max-width:100%;min-width:0;box-sizing:border-box}}
.att>img{{grid-area:1/1;max-width:100%;height:auto;transition:opacity .3s ease}}
@media (max-width:600px){{main{{padding:32px 16px}}.hero{{grid-template-columns:1fr}}.hero .brainbox{{width:62%}}.att{{margin:16px 0 0}}}}
.att .strip{{opacity:0}}.is-case .att .one{{opacity:0}}.is-case .att .strip{{opacity:1}}
small{{display:block;margin-top:28px;color:#5f5b70;font-size:12.5px}}
{css}</style></head>
<body><main>
<div class="mode" role="group" aria-label="Input"><button data-mode="slide" aria-pressed="true">Slide-level</button><button data-mode="case" aria-pressed="false">Case-level</button></div>
<div class="hero" id="hero">{snippet}<div class="att"><img class="one" src="slide_one.svg" alt="One attached slide"><img class="strip" src="slide_strip.svg" alt="Attached slides 1 to {n}"></div></div>
<small>Coronal sections y = {ys} mm, drawn from {lic}. The sections and blocks are a schematic, not a cutting plan.</small>
</main>
<script>
var hero=document.getElementById("hero"),bs=[].slice.call(document.querySelectorAll(".mode button"));
bs.forEach(function(b){{b.addEventListener("click",function(){{bs.forEach(function(x){{x.setAttribute("aria-pressed",x===b?"true":"false")}});hero.classList.toggle("is-case",b.dataset.mode==="case")}})}});
</script></body></html>
"""


def build(cfg=None, log=print):
    cfg = cfg or load()
    meta = json.loads((OUT / "sections.json").read_text())
    c = css(cfg)
    (OUT / "stack.css").write_text(c)
    (OUT / "stack_snippet.html").write_text(snippet(meta) + "\n")
    ys = ", ".join(f"{s['y_mm']:+d}" for s in meta["sections"])
    (OUT / "stack.html").write_text(PAGE.format(page=cfg["palette"]["page"], css=c, snippet=snippet(meta), n=len(meta["sections"]),
                                                ys=ys, lic=meta["license"]))
    log("stack.html, stack.css, stack_snippet.html")


if __name__ == "__main__":
    build()
    sys.exit(0)

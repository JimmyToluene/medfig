"""Write report/numbers.tex: the values the report quotes, read from the configuration and the pipeline outputs.

Usage: python -m medfig report   (after sections and figures; `make report` runs it before LaTeX)
Every number in report.tex that comes from a run is a macro defined here, so the text and the outputs agree.
"""
import json
import sys

from .config import BUILD, OUT, ROOT, load


def _tex(s):
    return str(s).replace("%", r"\%").replace("_", r"\_").replace("#", r"\#").replace("&", r"\&")


def build(cfg=None, log=print):
    cfg = cfg or load()
    num = json.loads((BUILD / "fig" / "numbers.json").read_text())
    sec = json.loads((OUT / "sections.json").read_text())
    r, v, st = cfg["raster"], cfg["vector"], cfg["stack_3d"]
    m = {
        "ExampleY": f"{num['example_y_mm']:+d}",
        "RawW": num["raw_px"][0], "RawH": num["raw_px"][1], "DownW": num["down_px"][0], "DownH": num["down_px"][1],
        "NPlanes": num["planes_shape"][0], "TissueChanged": num["tissue_changed_px"],
        "ExOtsu": f"{num['example_otsu']:.1f}", "ExThr": f"{num['example_threshold']:.1f}",
        "PtsRaw": num["contour_points_raw"], "PtsSimple": num["contour_points_simplified"],
        "VbW": sec["viewBox"][2], "VbH": sec["viewBox"][3], "MmPerUnit": f"{sec['mm_per_unit']:.1f}",
        "Ds": r["downsample"], "TBlur": r["tissue_blur_sigma"], "TMax": r["tissue_max_intensity"],
        "TObj": r["tissue_min_object_px"], "THole": r["tissue_max_hole_px"], "LSig": r["local_mean_sigma"],
        "Floor": r["otsu_floor"], "Offset": r["wm_offset"], "RSig": r["mask_round_sigma"], "WIsl": r["wm_min_island_px"],
        "CSig": v["contour_sigma"], "Eps": v["dp_epsilon"], "Pad": v["pad_units"], "Blk": v["block_units"],
        "Persp": st["perspective_px"], "Tx": _tex(st["case"]["translate_x"]), "Ry": st["case"]["rotate_y_deg"],
        "Rx": st["case"]["rotate_x_deg"], "Sc": st["case"]["scale"], "Dz": st["case"]["depth_step_px"],
        "PhTx": _tex(st["phone"]["translate_x"]), "PhRy": st["phone"]["rotate_y_deg"], "PhSc": st["phone"]["scale"],
        "PhDz": st["phone"]["depth_step_px"], "PhMax": st["phone"]["max_width_px"], "ZStart": st["start_depth_px"],
    }
    lines = ["% written by python -m medfig report; do not edit"]
    for k, val in m.items():
        lines.append(f"\\newcommand{{\\v{k}}}{{{val}}}")
    rows = []
    for s, t in zip(sec["sections"], num["thresholds"]):
        rows.append(f"{s['k']} & ${s['y_mm']:+d}$ & {_tex(s['label'])} & {_tex(s['block'])} & $({s['block_mm'][0]:.0f},\\,{s['block_mm'][1]:.0f})$ & "
                    f"{t['otsu_all']:.1f} & {t['otsu_floor']:.1f} & {s['threshold']:.1f} \\\\")
    lines.append("\\newcommand{\\SectionRows}{" + "\n".join(rows) + "}")
    delays = " & ".join(f"{d:.2f}" for d in st["section_delay_s"])
    mdel = " & ".join(f"{d:.2f}" for d in st["marker_delay_s"])
    lines.append(f"\\newcommand{{\\SectionDelays}}{{{delays}}}")
    lines.append(f"\\newcommand{{\\MarkerDelays}}{{{mdel}}}")
    kb = ", ".join(_tex(k) + f": {val} KB" for k, val in num["svg_kb"].items())
    lines.append(f"\\newcommand{{\\SvgSizes}}{{{kb}}}")
    (ROOT / "report" / "numbers.tex").write_text("\n".join(lines) + "\n")
    log("report/numbers.tex")


if __name__ == "__main__":
    build()
    sys.exit(0)

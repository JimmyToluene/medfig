"""Write report/numbers.tex: the values the report quotes, read from the configuration and the pipeline outputs.

Usage: python -m medfig report   (after the steps of every modality; `make report` runs it before LaTeX)
Input:  WSI: config/sections.json, out/sections.json, build/fig/numbers.json          -> macros \\v..., \\SectionRows, ...
        MRI: config/mri.json, out/mri/slices.json, build/fig/mri_numbers.json         -> macros \\m..., colours m...
             the MRI planes .npz (config source.planes_npz): its size and its meta    -> \\mPlanesMB, \\mVolX, \\mVolY, \\mVolZ
        CT:  config/ct.json, out/ct/slices.json, build/fig/ct_numbers.json, the CT planes .npz   -> macros \\ct..., colours ct...
        PET: config/pet.json, out/pet/<series>/slices.json, build/fig/pet_numbers.json, the PET planes .npz
                                                                                      -> macros \\pet..., colours pet...
Output: report/numbers.tex
Every number in report.tex that comes from a run is a macro defined here, so the text and the outputs agree.
The prefixes v, m, ct and pet, each followed by a capital letter, name no command of LaTeX or of the packages of report.tex
(\\newcommand stops on a name that exists).
report.tex needs the macros of every modality. If an input is missing or not usable, the step stops with exit code 1.
The message names the input and the command to run. report/numbers.tex is not changed.
CT and PET are checked against MRI where the report says that they agree with it: the PET frame and brain mask are the MRI ones.
"""
import json
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

from . import scan
from .config import BUILD, OUT, ROOT, load, planes_path

MRI_COLOURS = (("csf", "Csf"), ("grey_matter", "Gm"), ("white_matter", "Wm"), ("outline", "Line"))   # palette key, name in TeX
CT_COLOURS = (("screen", "Screen"), ("bone", "Bone"), ("brain", "Brain"), ("csf", "Csf"), ("outline", "Line"))
UNUSABLE = (FileNotFoundError, KeyError, IndexError, ValueError, TypeError)    # what a missing input or an input of another run raises
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"     # PET: band k and series k get the letter k in their macro names (TeX names have no digits)


def _tex(s):
    return str(s).replace("%", r"\%").replace("_", r"\_").replace("#", r"\#").replace("&", r"\&")


def _pair(p, nd=1):
    """Two values (x, z) as one TeX cell."""
    return f"$({p[0]:.{nd}f},\\,{p[1]:.{nd}f})$"


def _cells(vals, fmt="{:.2f}"):
    return " & ".join(fmt.format(v) for v in vals)


def _half_up(x, nd=2):
    """x with nd decimals, halves rounded up. The numbers files hold values rounded to 3 decimals (a Dice value of 0.9652
    is stored as 0.965); f"{0.965:.2f}" gives 0.96 (0.965 is 0.96499... in binary), the figure titles 0.97."""
    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def _command(name, val):
    return f"\\newcommand{{\\{name}}}{{{val}}}"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _colours(prefix, pal, keys):
    """\\definecolor lines for palette entries, and the macros Col<Name> with the hex code as text."""
    lines, m = [], {}
    for key, name in keys:
        hx = pal[key].lstrip("#").upper()
        lines.append(f"\\definecolor{{{prefix}{name}}}{{HTML}}{{{hx}}}")
        m["Col" + name] = "\\#" + hx
    return lines, m


def _wsi(cfg, num, sec):
    """The WSI lines."""
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
    lines = [_command("v" + k, val) for k, val in m.items()]
    rows = []
    for s, t in zip(sec["sections"], num["thresholds"]):
        rows.append(f"{s['k']} & ${s['y_mm']:+d}$ & {_tex(s['label'])} & {_tex(s['block'])} & $({s['block_mm'][0]:.0f},\\,{s['block_mm'][1]:.0f})$ & "
                    f"{t['otsu_all']:.1f} & {t['otsu_floor']:.1f} & {s['threshold']:.1f} \\\\")
    lines.append(_command("SectionRows", "\n".join(rows)))
    lines.append(_command("SectionDelays", _cells(st["section_delay_s"])))
    lines.append(_command("MarkerDelays", _cells(st["marker_delay_s"])))
    kb = ", ".join(_tex(k) + f": {val} KB" for k, val in num["svg_kb"].items())
    lines.append(_command("SvgSizes", kb))
    return lines


def _planes_file(cfg, name, cmd):
    """A planes file: its size in MB (1 MB = 10^6 bytes) and its meta. A missing file stops the step (exit code 1)."""
    p = planes_path(cfg)
    if not p.is_file():
        raise SystemExit(f"{name} planes file missing: {p}. report/numbers.tex is not changed. "
                         f"Restore the file, or cut it again with `{cmd}`. Then run this step again.")
    with np.load(p) as z:
        return p.stat().st_size / 1e6, json.loads(str(z["meta"]))


def _planes(cfg):
    """The MRI planes file: its size in MB and the volume size (nx, ny, nz) from its meta."""
    mb, meta = _planes_file(cfg, "MRI", "python -m medfig mri extract HOLIATLAS_DIR")
    return mb, meta["dims"]


def _side(cfg):
    """The orientation macros of a scanner modality: its name and the sign s of the x step from one column to the next."""
    s = scan.x_sign(cfg)
    return {"Orientation": cfg["orientation"], "XSign": f"{s:+d}"}


def _mri_macros(cfg, num, sl, half_mm, planes_mb, dims):
    """The MRI values. half_mm is half the side of the WSI block marker in mm. planes_mb and dims are the result of _planes()."""
    r, v, st, tl = cfg["raster"], cfg["vector"], cfg["stack_3d"], cfg["tissue_labels"]
    vs, upp, ys, names = num["voxel_mm"], num["units_per_px"], num["candidate_y_mm"], num["tissue_names"]
    named = lambda ks: ", ".join(f"{_tex(names[str(k)])} ({k})" for k in ks)  # noqa: E731
    used = tl["grey_matter"] + tl["white_matter"]
    cut = max(num["mask_steps"], key=lambda s: s["icv_removed_px"])
    layers = num["contour_points_layers"].values()
    dist, dice = [b["distance_mm"] for b in num["wsi_block"]], [p["dice"] for p in num["pairing"]]
    inside = sum(1 for b in num["wsi_block"] if max(abs(d) for d in b["offset_mm"]) <= half_mm)
    return {
        "ExampleY": f"{num['example_y_mm']:+d}", "ExampleK": num["example_k"], "NPlanes": num["n_candidates"],
        "YFirst": f"{max(ys):+d}", "YLast": f"{min(ys):+d}", "NSlices": len(sl["slices"]), "NShared": len(num["pairing"]),
        "SliceYs": ",\\,".join(f"{s['y_mm']:+d}" for s in sl["slices"]),
        "Voxel": vs, "PlaneW": num["plane_px"][0], "PlaneH": num["plane_px"][1], "XZero": num["x0_mm"], "ZTop": num["z_top_mm"],
        **_side(cfg), "XStep": f"{scan.x_sign(cfg) * vs:g}",
        "PlanesMB": f"{planes_mb:.1f}", "VolX": dims[0], "VolY": dims[1], "VolZ": dims[2],
        "GmList": ", ".join(str(k) for k in tl["grey_matter"]), "WmList": ", ".join(str(k) for k in tl["white_matter"]),
        "GmLabels": named(tl["grey_matter"]), "WmLabels": named(tl["white_matter"]),
        "OtherLabels": named(k for k in sorted(int(k) for k in names) if k not in used),
        "IcvPx": num["example_px"]["intracranial"], "MaskPx": num["example_px"]["mask"],
        "Close": r["mask_close_px"], "CloseMm": f"{r['mask_close_px'] * vs:g}",
        "CloseAdded": num["closing_added_px"], "CloseAddedMm": num["closing_added_mm2"],
        "IcvCut": num["icv_removed_px"], "IcvCutMax": cut["icv_removed_px"], "IcvCutMaxY": f"{cut['y_mm']:+d}",
        "RSig": r["mask_round_sigma"], "MinObj": r["min_object_px"], "RoiMin": r["roi_min_px"],
        "PtsRaw": num["contour_points_raw"], "PtsSimple": num["contour_points_simplified"],
        "PtsRawAll": sum(n["raw"] for n in layers), "PtsSimpleAll": sum(n["simplified"] for n in layers),
        "CSig": v["contour_sigma"], "Eps": v["dp_epsilon"], "EpsUnits": f"{v['dp_epsilon'] * upp:.2f}", "MinPts": v["min_contour_points"],
        "Unit": v["unit_mm"], "UnitsPerPx": f"{upp:.2f}", "Pad": v["pad_units"],
        "Roi": v["roi_units"], "RoiMm": f"{v['roi_units'] * v['unit_mm']:.1f}", "Outline": v["outline_width"],
        "RoiLabels": "; ".join(", ".join(str(k) for k in s.get("roi_labels") or ["none"]) for s in cfg["slices"]),
        "VbW": sl["viewBox"][2], "VbH": sl["viewBox"][3], "FrameX": sl["frame_mm"][0], "FrameZ": sl["frame_mm"][1],
        "DistMin": f"{min(dist):.1f}", "DistMax": f"{max(dist):.1f}", "BlkHalf": f"{half_mm:.1f}",
        "InBlock": f"{inside} of {len(dist)}", "DiceMin": f"{min(dice):.2f}", "DiceMax": f"{max(dice):.2f}",
        "Persp": st["perspective_px"], "Tx": _tex(st["case"]["translate_x"]), "Ry": st["case"]["rotate_y_deg"],
        "Rx": st["case"]["rotate_x_deg"], "Sc": st["case"]["scale"], "Dz": st["case"]["depth_step_px"],
        "PhTx": _tex(st["phone"]["translate_x"]), "PhRy": st["phone"]["rotate_y_deg"], "PhSc": st["phone"]["scale"],
        "PhDz": st["phone"]["depth_step_px"], "PhMax": st["phone"]["max_width_px"], "ZStart": st["start_depth_px"],
    }


def _mri_rows(num, sl):
    """The MRI table rows: the slices, the WSI block centres against the atlas centroids, the pairing."""
    area = {a["k"]: a for a in num["area_mm2"]}
    by_k = {s["k"]: s for s in sl["slices"]}
    slices, blocks, pairs = [], [], []
    for s in sl["slices"]:
        a, centre = area[s["k"]], _pair(s["roi_mm"]) if s["roi_mm"] else "--"
        slices.append(f"{s['k']} & ${s['y_mm']:+d}$ & {_tex(s['label'])} & {_tex(s['roi'] or '--')} & {centre} & "
                      f"{a['mask']} & {a['csf']} & {a['grey_matter']} & {a['white_matter']} \\\\")
    for b in num["wsi_block"]:
        blocks.append(f"{b['k']} & ${b['y_mm']:+d}$ & {_tex(b['block'])} & {_pair(b['block_mm'], 0)} & {_tex(by_k[b['k']]['roi'])} & "
                      f"{_pair(b['roi_mm'])} & {_pair(b['offset_mm'])} & {b['distance_mm']:.1f} \\\\")
    for p in num["pairing"]:
        pairs.append(f"{p['k']} & ${p['y_mm']:+d}$ & {p['mri_mask_mm2']} & {p['wsi_tissue_mm2']} & {p['dice']:.2f} & {p['dice_tissue']:.2f} \\\\")
    return {"mSliceRows": slices, "mBlockRows": blocks, "mPairRows": pairs}


def _mri_lines(wcfg, sec):
    """The MRI lines and the content of out/mri/slices.json.
    Raises one of UNUSABLE if an input is missing or if mri_numbers.json and slices.json are not from the same run."""
    cfg = load("mri")
    sl = _read(OUT / "mri" / "slices.json")
    num = _read(BUILD / "fig" / "mri_numbers.json")
    planes_mb, dims = _planes(cfg)
    if not num["wsi_block"] or not num["pairing"]:
        raise SystemExit("MRI input not usable: build/fig/mri_numbers.json has no pairing with WSI. report/numbers.tex is not changed. "
                         "The report needs an MRI slice with a marker at the y of a WSI section. "
                         "Change config/mri.json, then run `python -m medfig mri` and this step again.")
    same = (num["viewBox"] == sl["viewBox"] and num["frame_mm"] == sl["frame_mm"]
            and [(a["k"], a["y_mm"], a["mask"]) for a in num["area_mm2"]] == [(s["k"], s["y_mm"], s["area_mm2"]["mask"]) for s in sl["slices"]]
            and [(q["k"], q["roi_mm"]) for q in num["roi"]] == [(s["k"], s["roi_mm"]) for s in sl["slices"] if s["roi_mm"]])
    if not same:
        raise ValueError("build/fig/mri_numbers.json does not agree with out/mri/slices.json (frame, slices or markers)")
    if sl.get("orientation") != cfg["orientation"]:
        raise ValueError(f"out/mri/slices.json has the orientation {sl.get('orientation')!r}, {cfg['_path']} has {cfg['orientation']!r}")
    st = cfg["stack_3d"]
    lines, cols = _colours("m", cfg["palette"], MRI_COLOURS)
    lines = ["% MRI: read from config/mri.json, the planes file, out/mri/slices.json and build/fig/mri_numbers.json"] + lines
    m = _mri_macros(cfg, num, sl, wcfg["vector"]["block_units"] * sec["mm_per_unit"] / 2, planes_mb, dims)
    m.update(cols, License=_tex(cfg["source"]["license"]))
    lines += [_command("m" + k, val) for k, val in m.items()]
    lines += [_command(k, "\n".join(rows)) for k, rows in _mri_rows(num, sl).items()]
    lines.append(_command("mSliceDelays", _cells(st["section_delay_s"])))
    lines.append(_command("mMarkerDelays", _cells(st["marker_delay_s"])))
    lines.append(_command("mSvgSizes", ", ".join(_tex(s["file"]) + f": {num['svg_kb'][s['file']]} KB" for s in sl["slices"])))
    return lines, sl


# ---------------------------------------------------------------------------------------------------------------- CT
def _ct_check(cfg, num, sl, meta):
    """CT inputs of one run: raises ValueError if slices.json, ct_numbers.json, the planes file and config/ct.json disagree."""
    areas = lambda rows: [(a["k"], a["y_mm"], a["ring"], a["brain"], a["csf"]) for a in rows]  # noqa: E731
    same = (num["viewBox"] == sl["viewBox"] and num["frame_mm"] == sl["frame_mm"] and num["cut_z_mm"] == sl["cut_z_mm"]
            and areas(num["area_mm2"]) == areas({"k": s["k"], "y_mm": s["y_mm"], **s["area_mm2"]} for s in sl["slices"])
            and [(q["k"], q["roi_mm"]) for q in num["roi"]] == [(s["k"], s["roi_mm"]) for s in sl["slices"]])
    if not same:
        raise ValueError("build/fig/ct_numbers.json does not agree with out/ct/slices.json (frame, slices or markers)")
    if sl.get("orientation") != cfg["orientation"]:
        raise ValueError(f"out/ct/slices.json has the orientation {sl.get('orientation')!r}, {cfg['_path']} has {cfg['orientation']!r}")
    if num["hu"] != meta["hu"] or num["icv_ml"] != meta["icv_ml"] or any(cfg["hu"][k] != val for k, val in num["hu"].items()):
        raise ValueError(f"the CT planes file, build/fig/ct_numbers.json and {cfg['_path']} do not hold the same hu thresholds. "
                         "If config hu changed, run `python -m medfig ct extract CT_DIR` first")
    if not num["pairing"]:
        raise ValueError("build/fig/ct_numbers.json has no pairing with MRI")


def _ct_macros(cfg, num, sl, mb):
    h, r, v, f, sc = cfg["hu"], cfg["raster"], cfg["vector"], cfg["figures"], cfg["screen"]
    vs, ct, pair = num["voxel_mm"], num["ct"], num["pairing"]
    di, dm = [p["dice_icv"] for p in pair], [p["dice_mri_mask"] for p in pair]
    more = [100 * (p["ct_brain_mm2"] - p["holi_icv_mm2"]) / p["holi_icv_mm2"] for p in pair]
    return {
        "ExampleY": f"{num['example_y_mm']:+d}", "ExampleK": num["example_k"], "NSlices": len(sl["slices"]),
        "SliceYs": ",\\,".join(f"{s['y_mm']:+d}" for s in sl["slices"]),
        "NPlanes": num["planes_shape"][0], "PlaneW": num["planes_shape"][2], "PlaneH": num["planes_shape"][1], "Voxel": f"{vs:g}",
        "PlanesMB": f"{mb:.1f}", "VolX": ct["dims"][0], "VolY": ct["dims"][1], "VolZ": ct["dims"][2], "VolVs": f"{ct['vs']:g}",
        "ZMin": f"{ct['z_min_mm']:g}", "IcvMl": f"{num['icv_ml']:.0f}", **_side(cfg),
        "SoftLo": h["soft_window"][0], "SoftHi": h["soft_window"][1], "Erode": h["erode_vox"], "Grow": h["erode_vox"] + 1,
        "GrowFrom": h["grow_from"], "BoneHu": h["bone"], "CsfHu": h["csf"], "Outside": h["outside"],
        "Suture": r["suture_close_px"], "SutureMm": f"{r['suture_close_px'] * vs:g}", "CutPad": r["cut_pad_px"],
        "CutPadMm": f"{r['cut_pad_px'] * vs:g}", "RSig": r["mask_round_sigma"], "MinObj": r["min_object_px"], "RoiMin": r["roi_min_px"],
        "CSig": v["contour_sigma"], "Eps": v["dp_epsilon"], "MinPts": v["min_contour_points"], "Unit": v["unit_mm"], "Pad": v["pad_units"],
        "Roi": v["roi_units"], "RoiMm": f"{v['roi_units'] * v['unit_mm']:.1f}", "Outline": v["outline_width"],
        "Inset": sc["inset_units"], "Radius": sc["radius_units"],
        "BrainLo": f["brain_window_hu"][0], "BrainHi": f["brain_window_hu"][1], "BoneLo": f["bone_window_hu"][0], "BoneHi": f["bone_window_hu"][1],
        "CutRow": num["cut_row"], "Rows": num["cut_row"] + 1 + r["cut_pad_px"],
        "BonePx": num["bone_px"], "IcvPx": num["icv_px"], "CsfPx": num["csf_px"], "CloseAdded": num["closing_added_px"],
        "RingDropped": num["ring_dropped_px"],
        "VbW": sl["viewBox"][2], "VbH": sl["viewBox"][3], "FrameX": sl["frame_mm"][0], "FrameZ": sl["frame_mm"][1], "CutZ": f"{sl['cut_z_mm']:g}",
        "DiceIcvMin": _half_up(min(di)), "DiceIcvMax": _half_up(max(di)), "DiceMaskMin": _half_up(min(dm)), "DiceMaskMax": _half_up(max(dm)),
        "MoreMin": f"{min(more):.0f}", "MoreMax": f"{max(more):.0f}",
        "License": _tex(cfg["source"]["license"]), "MarkerCredit": _tex(cfg["source"]["marker_credit"]),
        "SvgSizes": ", ".join(_tex(s["file"]) + f": {num['svg_kb'][s['file']]} KB" for s in sl["slices"]),
    }


def _ct_rows(num, sl):
    """The CT table rows: the slices with their areas, and the pairing with MRI."""
    bone = {b["k"]: b["bone"] for b in num["bone_mm2"]}
    slices, pairs = [], []
    for s in sl["slices"]:
        a, centre = s["area_mm2"], _pair(s["roi_mm"]) if s["roi_mm"] else "--"
        slices.append(f"{s['k']} & ${s['y_mm']:+d}$ & {_tex(s['label'])} & {_tex(s['roi'] or '--')} & {centre} & "
                      f"{a['ring']} & {bone[s['k']]} & {a['brain']} & {a['csf']} \\\\")
    for p in num["pairing"]:
        pairs.append(f"{p['k']} & ${p['y_mm']:+d}$ & {p['ct_brain_mm2']} & {p['holi_icv_mm2']} & {_half_up(p['dice_icv'])} & {_half_up(p['dice_mri_mask'])} \\\\")
    return {"ctSliceRows": slices, "ctPairRows": pairs}


def _ct_lines():
    """The CT lines. Raises one of UNUSABLE if an input is missing or if the inputs are not from the same run."""
    cfg = load("ct")
    sl = _read(OUT / "ct" / "slices.json")
    num = _read(BUILD / "fig" / "ct_numbers.json")
    mb, meta = _planes_file(cfg, "CT", "python -m medfig ct extract CT_DIR")
    _ct_check(cfg, num, sl, meta)
    lines, cols = _colours("ct", cfg["palette"], CT_COLOURS)
    lines = ["% CT: read from config/ct.json, the CT planes file, out/ct/slices.json and build/fig/ct_numbers.json"] + lines
    m = {**_ct_macros(cfg, num, sl, mb), **cols}
    lines += [_command("ct" + k, val) for k, val in m.items()]
    lines += [_command(k, "\n".join(rows)) for k, rows in _ct_rows(num, sl).items()]
    return lines


# ---------------------------------------------------------------------------------------------------------------- PET
def _pet_check(cfg, num, sls, meta, msl):
    """PET inputs of one run, and the agreement with MRI that the report states: the PET frame and brain mask are the MRI ones.
    Raises ValueError if they disagree."""
    edges = cfg["bands"]["edges"]
    if num["series"] != cfg["series"] or num["edges"] != edges or num["reference"]["mean"] != meta["reference"]["mean"]:
        raise ValueError(f"build/fig/pet_numbers.json does not agree with {cfg['_path']} or with the PET planes file (series, bands or reference)")
    for se in cfg["series"]:
        sl, key = sls[se["key"]], se["key"]
        same = (sl["viewBox"] == num["viewBox"] and sl["frame_mm"] == num["frame_mm"] and sl["bands"] == edges
                and sl["reference_mean"] == meta["reference"]["mean"][se["channel"]]
                and [(a["k"], a["y_mm"], a["mask"], a["bands"]) for a in num["area_mm2"][key]]
                == [(s["k"], s["y_mm"], s["area_mm2"]["mask"], s["area_mm2"]["bands"]) for s in sl["slices"]])
        if not same:
            raise ValueError(f"build/fig/pet_numbers.json does not agree with out/pet/{key}/slices.json (frame, slices, bands or reference)")
        if sl.get("orientation") != cfg["orientation"]:
            raise ValueError(f"out/pet/{key}/slices.json has the orientation {sl.get('orientation')!r}, {cfg['_path']} has {cfg['orientation']!r}")
    mask = {s["y_mm"]: s["area_mm2"]["mask"] for s in msl["slices"]}
    if (num["viewBox"], num["frame_mm"]) != (msl["viewBox"], msl["frame_mm"]) or any(
            mask.get(a["y_mm"]) != a["mask"] for a in num["area_mm2"][cfg["series"][0]["key"]]):
        raise ValueError("the PET frame or brain mask is not the MRI one (out/mri/slices.json), but the report says it is. "
                         "Run `python -m medfig mri` and `python -m medfig pet`, or change the text of report.tex")


def _luminance(hx):
    r, g, b = (int(hx.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _pet_bands(cfg):
    """Band labels (math), the colour lines and the macros of the bands: one colour petBand<letter> per band, a key."""
    edges, pal = cfg["bands"]["edges"], cfg["palette"]["bands"]
    labels = [f"$<{edges[0]:.1f}$"] + [f"${a:.1f}$--${b:.1f}$" for a, b in zip(edges, edges[1:])] + [f"$\\geq{edges[-1]:.1f}$"]
    lines, m = _colours("pet", {f"band{i}": c for i, c in enumerate(pal)}, [(f"band{i}", "Band" + LETTERS[i]) for i in range(len(pal))])
    key = "".join(f"\\colorbox{{petBand{LETTERS[i]}}}{{\\strut\\ \\color{{{'white' if _luminance(c) < 0.45 else 'black'}}}{t}\\ }}"
                  for i, (c, t) in enumerate(zip(pal, labels)))
    return labels, lines, {**m, "BandKey": key}


def _pet_macros(cfg, num, sls, mb):
    b, r, v, f, kk, ref = cfg["bands"], cfg["raster"], cfg["vector"], cfg["figures"], cfg["key"], cfg["reference"]
    first = sls[cfg["series"][0]["key"]]
    kb = [x for d in num["svg_kb"].values() for x in d.values()]
    m = {
        "ExampleY": f"{num['example_y_mm']:+d}", "ExampleK": num["example_k"], "NSlices": len(first["slices"]),
        "SliceYs": ",\\,".join(f"{s['y_mm']:+d}" for s in first["slices"]),
        "NPlanes": num["planes_shape"][0], "NativeVs": f"{num['native_vs']:g}",
        "VolX": num["native_dims"][0], "VolY": num["native_dims"][1], "VolZ": num["native_dims"][2], "PlanesMB": f"{mb:.1f}",
        **_side(cfg), "NSeries": len(cfg["series"]),
        "RefName": _tex(ref["name"]), "RefLabels": ", ".join(str(x) for x in ref["tissue_labels"]),
        "Edges": ", ".join(f"{e:.1f}" for e in b["edges"]), "NBands": len(b["edges"]) + 1, "NEdges": len(b["edges"]),
        "EdgeFirst": f"{b['edges'][0]:.1f}", "EdgeLast": f"{b['edges'][-1]:.1f}", "BandUnit": _tex(b["unit"]),
        "Smooth": f"{b['smooth_sigma_mm']:g}",
        "RSig": r["mask_round_sigma"], "MinObj": r["min_object_px"], "RoiMin": r["roi_min_px"],
        "CSig": v["contour_sigma"], "Eps": v["dp_epsilon"], "MinPts": v["min_contour_points"], "Unit": v["unit_mm"], "Pad": v["pad_units"],
        "Roi": v["roi_units"], "RoiMm": f"{v['roi_units'] * v['unit_mm']:.1f}", "Outline": v["outline_width"],
        "SuvrMax": f"{f['suvr_max']:g}", "HistBins": f["hist_bins"], "KeyW": kk["swatch_w"], "KeyH": kk["swatch_h"], "KeyFont": kk["font_px"],
        "VbW": num["viewBox"][2], "VbH": num["viewBox"][3], "FrameX": num["frame_mm"][0], "FrameZ": num["frame_mm"][1],
        "License": _tex(cfg["source"]["license"]), "OutlineCredit": _tex(cfg["source"]["outline_credit"]),
        "SvgMin": f"{min(kb):.1f}", "SvgMax": f"{max(kb):.1f}",
    }
    for i, se in enumerate(cfg["series"]):            # series i: its name, and the range of its grey- and white-matter SUVR
        t = [q for q in num["tissue_suvr"] if q["series"] == se["key"]]
        if not t:
            raise ValueError(f"build/fig/pet_numbers.json has no tissue SUVR of the series {se['key']!r}")
        x = LETTERS[i]
        m.update({"Series" + x: _tex(se["name"]), "GmMin" + x: f"{min(q['grey_matter'] for q in t):.2f}",
                  "GmMax" + x: f"{max(q['grey_matter'] for q in t):.2f}", "WmMin" + x: f"{min(q['white_matter'] for q in t):.2f}",
                  "WmMax" + x: f"{max(q['white_matter'] for q in t):.2f}"})
    return m


def _pet_rows(cfg, num, labels):
    """The PET table rows: the two series, and per slice and series the band shares and the grey- and white-matter SUVR."""
    names = {se["key"]: se for se in cfg["series"]}
    share = {(q["k"], q["series"]): q for q in num["band_share"]}
    tissue = {(q["k"], q["series"]): q for q in num["tissue_suvr"]}
    series = [f"{_tex(se['name'])} & \\nolinkurl{{{cfg['source']['volumes'][se['channel']]}}} & {num['reference']['mean'][se['channel']]:.3f} & "
              f"{num['reference']['points'][se['channel']]} \\\\" for se in cfg["series"]]
    rows, ks = [], [a["k"] for a in num["area_mm2"][cfg["series"][0]["key"]]]
    for k in ks:
        for key in names:
            q, t = share[(k, key)], tissue[(k, key)]
            if len(q["share"]) != len(labels):
                raise ValueError(f"build/fig/pet_numbers.json: slice {k}, series {key}: {len(q['share'])} band shares, the bands are {len(labels)}")
            rows.append(f"{k} & ${q['y_mm']:+d}$ & {_tex(names[key]['name'])} & {q['mask_mm2']} & "
                        + " & ".join(f"{100 * x:.0f}" for x in q["share"]) + f" & {t['grey_matter']:.2f} & {t['white_matter']:.2f} \\\\")
    return {"petSeriesRows": series, "petShareRows": rows, "petShareHead": [" & ".join(labels)]}


def _pet_lines(msl):
    """The PET lines. msl is out/mri/slices.json. Raises one of UNUSABLE if an input is missing or the inputs disagree."""
    cfg = load("pet")
    num = _read(BUILD / "fig" / "pet_numbers.json")
    sls = {se["key"]: _read(OUT / "pet" / se["key"] / "slices.json") for se in cfg["series"]}
    mb, meta = _planes_file(cfg, "PET", "python -m medfig pet extract PET_DIR HOLIATLAS_DIR")
    _pet_check(cfg, num, sls, meta, msl)
    labels, cl, bm = _pet_bands(cfg)
    ol, om = _colours("pet", cfg["palette"], (("outline", "Line"),))
    lines = ["% PET: read from config/pet.json, the PET planes file, out/pet/<series>/slices.json and build/fig/pet_numbers.json"] + cl + ol
    n = len(labels)                                                   # the columns of the band shares in the table: 5 to 4 + n
    m = {**_pet_macros(cfg, num, sls, mb), **bm, **om, "ShareRule": f"\\cmidrule(lr){{5-{4 + n}}}\\cmidrule(l){{{5 + n}-{6 + n}}}"}
    lines += [_command("pet" + k, val) for k, val in m.items()]
    lines += [_command(k, "\n".join(rows)) for k, rows in _pet_rows(cfg, num, labels).items()]
    return lines


# ---------------------------------------------------------------------------------------------------------------- all
def _wsi_lines(cfg):
    """The WSI lines and the content of out/sections.json. Raises one of UNUSABLE if an input is missing or from another run."""
    num = _read(BUILD / "fig" / "numbers.json")
    sec = _read(OUT / "sections.json")
    return _wsi(cfg, num, sec), sec


def _stop(modality, e, run):
    """Stop the step with exit code 1. The message names the input and the command to run."""
    why = f"input missing: {e.filename}" if isinstance(e, FileNotFoundError) else f"input not usable: {type(e).__name__}: {e}"
    raise SystemExit(f"{modality} {why}. report/numbers.tex is not changed. Run {run}, then this step again.")


def build(cfg=None, log=print):
    """Write report/numbers.tex. If an input is missing or not usable, the step stops before it writes the file."""
    cfg = cfg or load()
    try:
        lines, sec = _wsi_lines(cfg)
    except UNUSABLE as e:
        _stop("WSI", e, "`python -m medfig sections` and `python -m medfig figures`")
    try:
        mri, msl = _mri_lines(cfg, sec)
    except UNUSABLE as e:
        _stop("MRI", e, "`python -m medfig mri`")
    try:
        ct = _ct_lines()
    except UNUSABLE as e:
        _stop("CT", e, "`python -m medfig ct`")
    try:
        pet = _pet_lines(msl)
    except UNUSABLE as e:
        _stop("PET", e, "`python -m medfig pet`")
    lines = ["% written by python -m medfig report; do not edit"] + lines + mri + ct + pet
    (ROOT / "report" / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("report/numbers.tex")


if __name__ == "__main__":
    build()
    sys.exit(0)

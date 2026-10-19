"""medfig command line.

WSI (BigBrain):
    python -m medfig all            sections -> icons -> web -> figures -> render -> overview
    python -m medfig sections|icons|web|figures|overview|render|site|report
    python -m medfig extract VOLUME.nii.gz [OUT.npz] [--y ...]

MRI (holiAtlas, skull-stripped brain):
    python -m medfig mri [all]      slices -> icons -> web -> figures -> render
    python -m medfig mri slices|icons|web|figures|render
    python -m medfig mri extract HOLIATLAS_DIR [OUT.npz] [--y ...]

CT (Clinical Toolbox CT template, skull and brain) and PET (rPOP florbetaben templates, amyloid-negative and -positive):
    python -m medfig ct|pet [all]   slices -> icons -> web -> figures -> render
    python -m medfig ct|pet slices|icons|web|figures|render
    python -m medfig ct extract CT_DIR [OUT.npz] [--y ...]
    python -m medfig pet extract PET_DIR HOLIATLAS_DIR [OUT.npz] [--y ...]
CT and PET read the MRI planes file (grid, structure labels for the markers; PET also the brain mask).

Arguments after `render` go to the render module (for example --gif). An unknown step gives exit code 2.
`python -m medfig -h` and `python -m medfig mri -h` (or ct, pet) print this text.
"""
import sys

WSI = ["sections", "icons", "web", "figures", "render", "overview"]     # the steps of `all`, in order
MRI = ["slices", "icons", "web", "figures", "render"]                   # the steps of `mri all`, `ct all` and `pet all`, in order
SCANNERS = ("mri", "ct", "pet")


def unknown(name, cmd, known):
    print(f"{name}: unknown step '{cmd}'. Steps: {', '.join(known)}.\n{__doc__}", file=sys.stderr)
    return 2


def scanner(modality, args):
    """The steps of one scanner modality (mri, ct or pet)."""
    args = args or ["all"]
    cmd, rest = args[0], args[1:]
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if cmd not in ["all", "extract"] + MRI:
        return unknown(f"medfig {modality}", cmd, ["all", "extract"] + MRI)
    if cmd == "extract":
        return __import__(f"medfig.{modality}.extract", fromlist=["main"]).main(rest)
    for s in MRI if cmd == "all" else [cmd]:
        print(f"== {modality} {s}")
        mod = __import__(f"medfig.{modality}.{s}", fromlist=["build"])
        if s == "render":
            mod.main(rest)
        else:
            mod.build()
    return 0


def mri(args):
    return scanner("mri", args)


def main():
    args = sys.argv[1:] or ["all"]
    cmd, rest = args[0], args[1:]
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if cmd in SCANNERS:
        return scanner(cmd, rest)
    if cmd == "extract":
        from . import extract
        return extract.main(rest)
    if cmd not in ["all", "site", "report"] + WSI:
        return unknown("medfig", cmd, ["all", "extract", "site", "report", *SCANNERS] + WSI)
    steps = WSI if cmd == "all" else [cmd]
    for s in steps:
        print(f"== {s}")
        if s == "render":
            from . import render
            render.main(rest)
        elif s == "site":
            from . import site_export
            site_export.main(rest)
        else:
            mod = __import__(f"medfig.{s}", fromlist=["build"])
            mod.build()
    return 0


if __name__ == "__main__":
    sys.exit(main())

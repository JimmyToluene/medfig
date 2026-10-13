"""medfig command line.

    python -m medfig all            sections -> icons -> web -> figures -> render -> overview
    python -m medfig sections|icons|web|figures|overview|render|site
    python -m medfig extract VOLUME.nii.gz [OUT.npz] [--y ...]
"""
import sys


def main():
    args = sys.argv[1:] or ["all"]
    cmd, rest = args[0], args[1:]
    if cmd == "extract":
        from . import extract
        return extract.main(rest)
    steps = ["sections", "icons", "web", "figures", "render", "overview"] if cmd == "all" else [cmd]
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

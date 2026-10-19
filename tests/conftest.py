"""Setup of the checks. pytest runs this file before it loads the test modules.

The test modules call config.load() when they load. A MEDFIG_* variable of the shell would give them another
configuration or another planes file, and the checks would fail for a reason that is not in the code.
So this file removes the variables here, at module level. A fixture would run too late.
MEDFIG_CHROMIUM stays: only the render steps read it, and no check starts a browser.
"""
import os

for _k in ("MEDFIG_CONFIG", "MEDFIG_PLANES", "MEDFIG_EXAMPLE",
           "MEDFIG_MRI_CONFIG", "MEDFIG_MRI_PLANES", "MEDFIG_MRI_EXAMPLE",
           "MEDFIG_CT_CONFIG", "MEDFIG_CT_PLANES", "MEDFIG_CT_EXAMPLE",
           "MEDFIG_PET_CONFIG", "MEDFIG_PET_PLANES", "MEDFIG_PET_EXAMPLE"):
    os.environ.pop(_k, None)

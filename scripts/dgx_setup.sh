#!/usr/bin/env bash
# Set up medfig on a DGX Spark (aarch64, Ubuntu-based DGX OS) or any Linux machine.
# The pipeline uses the CPU only. It does not need the GPU. The script does not use sudo.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "machine: $(uname -m), python: $(python3 --version)"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Chromium for linux-arm64 or linux-x64. Only the render targets need it (`make render`, `mri-render`, `ct-render`, `pet-render`).
python -m playwright install chromium || echo "Chromium download failed. Set MEDFIG_CHROMIUM to a Chromium binary and run this script again."
if python -c "
import os
from playwright.sync_api import sync_playwright
exe = os.environ.get('MEDFIG_CHROMIUM')
with sync_playwright() as p:
    (p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()).close()
" 2>/dev/null; then
  echo "Chromium starts."
  ready="make all   (the last step of 'make all', the PDF, needs LaTeX)"
else
  echo "Chromium does not start. The render targets (render, mri-render, ct-render, pet-render) do not work until it does. The other targets work."
  echo "If system libraries are missing, install them (this command needs sudo):"
  echo "  python -m playwright install --with-deps chromium"
  ready="make sections icons web figures mri-slices mri-icons mri-web mri-figures ct-slices ct-icons ct-web ct-figures pet-slices pet-icons pet-web pet-figures   ('make all' needs Chromium, see above, and LaTeX for the PDF)"
fi
# LaTeX for the report (optional, needs sudo; the lab DGX Spark has no system LaTeX):
#   sudo apt-get install -y latexmk texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-science
sha256sum -c data/SHA256SUMS
make test
echo "ready: . .venv/bin/activate && $ready"

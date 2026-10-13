#!/usr/bin/env bash
# Set up medfig on a DGX Spark (aarch64, Ubuntu-based DGX OS) or any Linux machine.
# The pipeline uses the CPU only. It does not need the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "machine: $(uname -m), python: $(python3 --version)"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install --with-deps chromium     # Chromium for linux-arm64 or linux-x64
# LaTeX for the report (optional):
#   sudo apt-get install -y latexmk texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-science
sha256sum -c data/SHA256SUMS
make test
echo "ready: . .venv/bin/activate && make all"

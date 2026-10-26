#!/usr/bin/env bash
# Figure 1 (protocol diagram) from its TikZ source; needs pdflatex and pdftoppm. Run from the repository root.
set -euo pipefail
cd figures
pdflatex -interaction=nonstopmode fig01_protocol.tex >/dev/null
pdftoppm -png -r 300 -singlefile fig01_protocol.pdf fig01_protocol
rm -f fig01_protocol.aux fig01_protocol.log fig01_protocol.pdf

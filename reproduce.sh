#!/usr/bin/env bash
# Reproduce the results and figures. Run from the repository root.
#   bash reproduce.sh --reuse-only   check and re-evaluate the stored heads without fitting
#   bash reproduce.sh                train every missing head (CPU, up to four workers)
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
reuse_args=()
if [[ "${1:-}" == "--reuse-only" ]]; then reuse_args=(--reuse-only); fi
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -B codes_and_data/tests/validate_gallery.py
python -B codes_and_data/tests/validate_output_labels.py
python -B codes_and_data/pipelines/run_gallery.py --mode single-shot --workers 4 "${reuse_args[@]}"
python -B codes_and_data/pipelines/refine_1d_quadrature.py --workers 4 "${reuse_args[@]}"
python -B codes_and_data/pipelines/refine_2d_quadrature.py --workers 4 "${reuse_args[@]}"
python -B codes_and_data/evaluation/evaluate.py "${reuse_args[@]}"
python -B codes_and_data/figures/make_fig02_gallery_1d.py
python -B codes_and_data/figures/make_fig03_gallery_2d.py
if [[ -f codes_and_data/figures/make_fig04_haar_depth.py ]]; then
  python -B codes_and_data/figures/make_fig04_haar_depth.py
fi

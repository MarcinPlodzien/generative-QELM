# generative-QELM

Code and data for

M. Płodzień, *Generative quantum extreme learning machines* (2026).

A random latent vector is encoded by single-qubit rotations at dyadic frequencies, passes through a
fixed reservoir unitary and is measured. Only an affine output layer (the head) is trained. Training
minimizes the squared Wasserstein distance to the target in one dimension, and the squared sliced Wasserstein
distance with 32 directions in two dimensions. The squared maximum mean discrepancy (MMD²) is used only
for evaluation. For Clifford reservoirs the features and the latent-averaged bitstring weights are
obtained exactly by Pauli propagation.

## Setup used in the paper

- N = 12 qubits; Z parities of weight up to three, M = 298 features; affine head with d(M+1) parameters.
- Reservoirs: identity, cluster layer, and random Clifford brickworks of depth L = 4 (seeds 0–9; the
  figures use seed 1).
- Two readout modes: expectation values (trained on 2^15 latent points, evaluated on 2^17 others)
  and single shots (exact weights of all 4096 bitstrings).
- Brickworks of Haar-random two-qubit gates of depth K = 1, 2, 4, 8, 16 (seeds 0–2) for the 1D targets,
  with features and weights from the full state vector.
- Three random head initializations (seeds 12000–12002); the head with the lowest training loss is kept.
- Targets: five-peak Gaussian mixture and Gamma distribution (1D); two rings, two moons, diagonal cross,
  three stripes and four lobes (2D), on grids of 4096 nodes. Two further 1D targets, `narrow_broad` and
  `unequal_mixture`, were run with the same protocol but are not shown in the paper.

## Contents

| Path | Content |
|---|---|
| `codes_and_data/qgen/` | encoding, circuits, Pauli propagation, features, bitstring weights, losses, training |
| `codes_and_data/pipelines/` | training entry points, one per target |
| `codes_and_data/configs/` | protocol and settings |
| `codes_and_data/targets/` | target distributions on the grid, with their definitions |
| `codes_and_data/results/` | trained heads and evaluated output distributions |
| `codes_and_data/evaluation/` | MMD² and related metrics; `summary.csv` holds every number quoted in the paper |
| `codes_and_data/figures/` | one script per figure, writing to `figures/` |
| `codes_and_data/tests/` | independent validation of the propagation, weights, losses and output labels |
| `codes_and_data/cache/` | the two-qubit Clifford group used to draw the random circuits |
| `figures/` | the figures of the paper; Fig. 1 also as TikZ source |

## Requirements

Python 3 with NumPy, SciPy, Numba and Matplotlib, CPU only. The results were produced with Python 3.13.5,
NumPy 2.1.3, SciPy 1.15.3, Numba 0.61.0 and Matplotlib 3.10.8, with one BLAS thread per process; the
random streams of SciPy may differ between versions.

## Reproducing

From the repository root, to redraw the figures from the stored results:

```bash
bash codes_and_data/figures/make_fig01_protocol.sh
python -B codes_and_data/figures/make_fig02_gallery_1d.py
python -B codes_and_data/figures/make_fig03_gallery_2d.py
python -B codes_and_data/figures/make_fig04_haar_depth.py
```

To validate, re-evaluate and redraw everything (`--reuse-only` uses the stored heads; without it, missing
heads are trained):

```bash
bash reproduce.sh --reuse-only
```

Training all 678 heads (three starts for each of the 226 settings) took about 10 hours of single-thread
CPU time in total, at most 42 minutes for one head. The feature matrices of the expectation-value mode
(about 7 GB in total) are not stored. For the Clifford reservoirs they are recomputed when needed; for the
Haar brickworks, delete `codes_and_data/results/haar_depth/circuits/seed*_complete.json` and run
`python -B codes_and_data/pipelines/run_haar_depth.py --stage features`.

## License

MIT, see `LICENSE`.

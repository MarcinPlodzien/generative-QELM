"""Fixed Clifford features x, trainable affine outputs y, and grid labels.

Training uses Wasserstein loss. MMD squared is evaluation only.
"""
import os
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/qelm-numba')
os.environ.setdefault('MPLCONFIGDIR','/tmp/qelm-matplotlib')

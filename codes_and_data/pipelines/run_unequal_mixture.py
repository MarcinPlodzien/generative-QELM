"""Train the unequal Gaussian mixture in both modes."""
from _training_cli import main
from qgen.target_builders import build_unequal_mixture
import sys

if __name__=='__main__':
 build_unequal_mixture(reuse_only='--reuse-only' in sys.argv)
 main(['unequal_mixture'])

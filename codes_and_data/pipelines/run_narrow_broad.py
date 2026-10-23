"""Train the fixed thin Gaussian plus broad shifted-Gamma target in both modes."""
from _training_cli import main
from qgen.target_builders import build_narrow_broad
import sys

if __name__=='__main__':
 build_narrow_broad(reuse_only='--reuse-only' in sys.argv)
 main(['narrow_broad'])

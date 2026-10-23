"""Train the cluster reservoir in single-shot mode for eight targets."""
from _training_cli import main
from qgen.features import prepare_features
from qgen.data import PAPER_1D,TARGETS_2D
import sys

if __name__=='__main__':
 # Prepare shared D1/D2 caches before concurrent target jobs can request them.
 if '--reuse-only' not in sys.argv:
  for d in [1,2]:prepare_features('shot_cluster',d,n=4096)
 main(PAPER_1D+TARGETS_2D,mode='single-shot',default_models=['shot_cluster'])

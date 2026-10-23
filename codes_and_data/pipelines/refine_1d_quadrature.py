"""Final 1D expectation-value training on 32768 latent points; same generic starts."""
import sys
sys.dont_write_bytecode=True
from _training_cli import main
from qgen.data import TARGETS_1D
if __name__=="__main__":main(TARGETS_1D,mode="expectation-value")

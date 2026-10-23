"""Final 2D expectation-value training on 32768 latent points; same generic starts."""
import sys
sys.dont_write_bytecode=True
from _training_cli import main
if __name__=="__main__":main(['two_rings', 'two_moons', 'diagonal_cross', 'three_stripes', 'four_lobes'],mode="expectation-value")

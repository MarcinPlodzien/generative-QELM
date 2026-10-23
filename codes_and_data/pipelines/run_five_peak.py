"""Train five_peak in expectation-value and single-shot modes; reuse completed heads."""
import sys
sys.dont_write_bytecode=True
from _training_cli import main
if __name__=="__main__":main(["five_peak"])

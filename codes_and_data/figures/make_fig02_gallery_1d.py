"""Fig. 2: one-dimensional targets for the three reservoirs U_1 (no gates), U_2 (cluster layer)
and U_3 (random Clifford brickwork, seed MAIN_SEED).

Rows: U_1, U_2, U_3. Columns: five-peak mixture and Gamma distribution, each in the
expectation-value mode (blue bars) and the single-shot mode (orange bars). Histograms: final
4096-node output PMFs grouped into 128 display bins; lines: target densities.
"""
import sys
sys.dont_write_bytecode=True
from paper_style import *

RESERVOIRS = [('identity', r'$\hat U_1$'), ('cluster', r'$\hat U_2$'), (f'random{MAIN_SEED}', r'$\hat U_3$')]
COLUMNS = [('five_peak', 'expectation'), ('five_peak', 'shot'), ('gamma', 'expectation'), ('gamma', 'shot')]


def main():
    fig, axes = plt.subplots(3, 4, figsize=(PAGE, 3.9), constrained_layout=True, sharex='col', sharey='col')
    for r, (res, rlabel) in enumerate(RESERVOIRS):
        for c, (name, mode) in enumerate(COLUMNS):
            ax = axes[r, c]
            face, edge = MODE_BARS[mode]
            histogram_1d(ax, name, quantized_law(name, f'{mode}_{res}'), face=face, edge=edge)
            right = name == 'gamma'
            ax.text(.96 if right else .04, .95, f'({"abcdefghijkl"[4*r+c]})', transform=ax.transAxes,
                    va='top', ha='right' if right else 'left')
        axes[r, 0].set_ylabel(f'{rlabel}\nprobability density')
    for c in range(4):
        axes[2, c].set_xlabel('$y$')
    save(fig, 'fig02_gallery_1d.png')


if __name__ == '__main__':
    main()

"""Fig. 3: two-dimensional targets for the three reservoirs U_1 (no gates), U_2 (cluster layer)
and U_3 (random Clifford brickwork, seed MAIN_SEED).

Rows: two rings, two moons, four lobes. Columns: target (gray), then U_1, U_2, U_3 in the
expectation-value mode (blue), then U_1, U_2, U_3 in the single-shot mode (orange). Each row shares
one square-root color scale; pixels are the actual 64x64 output PMF, without smoothing.
"""
import sys
sys.dont_write_bytecode=True
from paper_style import *

RES = [('identity', r'$\hat U_1$'), ('cluster', r'$\hat U_2$'), (f'random{MAIN_SEED}', r'$\hat U_3$')]


def main():
    fig, axes = plt.subplots(3, 7, figsize=(PAGE, 3.2), constrained_layout=True, sharex=True, sharey=True)
    titles = ['target'] + [l for _, l in RES] * 2
    cmaps = [CMAP_TARGET] + [CMAP]*3 + [CMAP_SHOT]*3
    for r, name in enumerate(MAIN_2D):
        pmfs = [target(name)['p']] + [quantized_law(name, f'expectation_{m}') for m, _ in RES] \
               + [quantized_law(name, f'shot_{m}') for m, _ in RES]
        norm = shared_norm(*pmfs)
        for c, pmf in enumerate(pmfs):
            ax = axes[r, c]
            image_2d(ax, name, pmf, norm, cmap=cmaps[c])
            panel_label(ax, 7*r+c, fontsize=6)
            if r == 0:
                ax.set_title(titles[c], fontsize=8)
        axes[r, 0].set_ylabel('$y_2$')
    for c in range(7):
        axes[2, c].set_xlabel('$y_1$')
    save(fig, 'fig03_gallery_2d.png')


if __name__ == '__main__':
    main()

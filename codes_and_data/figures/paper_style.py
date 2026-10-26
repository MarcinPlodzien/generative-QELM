"""Shared conventions for the paper figures: fonts, colors, data access.

Every make_fig*.py script imports this module, reads saved results only (no
training), and writes one PNG into figures/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
DEST = ROOT/'figures'
sys.path.insert(0, str(CODE))
from qgen.data import target, result_folder, TARGETS_1D, TARGETS_2D  # noqa: E402

# Random Clifford circuit shown in the main-text figures. Fixed for all targets
# (not selected per panel); every circuit is included in evaluation/summary.csv.
MAIN_SEED = 1

MAIN_1D = ['five_peak', 'gamma']
MAIN_2D = ['two_rings', 'two_moons', 'four_lobes']
EXPECTATION = ['expectation_identity', 'expectation_cluster'] + [f'expectation_random{s}' for s in range(3)]
SHOT = ['shot_identity'] + [f'shot_random{s}' for s in range(3)]

NAMES = {'unequal_mixture': 'unequal mixture', 'narrow_broad': 'narrow + skewed', 'five_peak': 'five peaks', 'gamma': 'Gamma', 'two_rings': 'two rings', 'two_moons': 'two moons',
         'diagonal_cross': 'diagonal cross', 'three_stripes': 'three stripes', 'four_lobes': 'four lobes'}
CIRCUITS = {'identity': 'identity', 'cluster': 'cluster', 'random0': 'random 1',
            'random1': 'random 2', 'random2': 'random 3'}

BAR_FACE, BAR_EDGE, LINE = '#9ecae1', '#4292c6', 'black'
# expectation-value mode in blue, single-shot mode in orange
MODE_BARS = {'expectation': ('#9ecae1', '#4292c6'), 'shot': ('#fdc590', '#e6550d')}
CMAP = LinearSegmentedColormap.from_list('white_blue', ['#ffffff', '#9ecae1', '#4292c6', '#08519c', '#08306b'])
CMAP_SHOT = LinearSegmentedColormap.from_list('white_orange', ['#ffffff', '#fdc590', '#fd8d3c', '#d94701', '#7f2704'])
CMAP_TARGET = LinearSegmentedColormap.from_list('white_gray', ['#ffffff', '#bdbdbd', '#737373', '#252525'])
COLUMN, PAGE = 3.4, 7.0  # widths in inches for one and two columns

plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'savefig.dpi': 300})


def circuit_label(model):
    return CIRCUITS[model.split('_', 1)[1]]


def quantized_law(name, model):
    """Final 4096-node PMF; expectation-value panels use the refined heads."""
    folder = result_folder(name, model)
    return np.load(folder/'evaluated_law.npz')['quantized_pmf']


def histogram_1d(ax, name, pmf, bins=128, face=BAR_FACE, edge=BAR_EDGE):
    """Generated PMF as a normalized histogram, target density as a line."""
    t = target(name)
    y = t['raw'][:, 0]
    if name in ('five_peak', 'narrow_broad'):  # bin index -> [0, 1]
        y = y/(len(y)-1)
    dy = y[1]-y[0]
    edges = np.linspace(y[0]-dy/2, y[-1]+dy/2, bins+1)
    mass, _ = np.histogram(y, bins=edges, weights=pmf)
    ax.bar(edges[:-1], mass/np.diff(edges), width=np.diff(edges), align='edge',
           color=face, edgecolor=edge, linewidth=.25 if bins <= 128 else 0)
    ax.plot(y, t['p']/dy, color=LINE, lw=.9)
    ax.set_xlim(edges[0], 9 if name == 'gamma' else edges[-1])
    ax.set_ylim(bottom=0)


def image_2d(ax, name, pmf, norm, cmap=CMAP):
    t = target(name)
    a = np.unique(t['raw'][:, 0])
    h = a[1]-a[0]
    ax.imshow(pmf.reshape(64, 64).T, origin='lower', extent=[a[0]-h/2, a[-1]+h/2]*2,
              cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([-3, 0, 3])


def shared_norm(*pmfs):
    return PowerNorm(.5, 0, max(p.max() for p in pmfs))


def panel_label(ax, k, **kw):
    ax.text(.03, .96, f'({"abcdefghijklmnopqrstuvwxyz"[k]})', transform=ax.transAxes, va='top', **kw)


def save(fig, filename):
    DEST.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST/filename, facecolor='white')
    plt.close(fig)
    print(DEST/filename, flush=True)

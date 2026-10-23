"""Fig.4: Haar depth scan, two MMD panels and raw feature spread.

Read saved results only. Small points are seeds 0,1,2; solid lines are medians.
Dashed lines show identity; the dotted spread reference is1/sqrt(2**N+1).
"""
import sys
sys.dont_write_bytecode=True
from paper_style import *

def main():
    rows=json.loads((CODE/'evaluation/summary.json').read_text())
    lookup={(r['target'],r['model']):r for r in rows}
    depths=[1,2,4,8,16]
    colors={'expectation':'#2171b5','shot':'#e6550d'}
    fig,axes=plt.subplots(3,1,figsize=(COLUMN,5.6),constrained_layout=True,sharex=True)
    for i,name in enumerate(['five_peak','gamma']):
        ax=axes[i]
        for mode in ['expectation','shot']:
            values=[[lookup[name,f'{mode}_haar_K{K}_s{seed}']['MMD2_quantized'] for seed in range(3)] for K in depths]
            for K,v in zip(depths,values):ax.scatter([K]*len(v),v,color=colors[mode],s=9,alpha=.5,lw=0)
            ax.plot(depths,[np.median(v) for v in values],'-o',color=colors[mode],ms=3.5,lw=1.2)
            ax.axhline(lookup[name,f'{mode}_identity']['MMD2_quantized'],color=colors[mode],ls='--',lw=.9)
        ax.set_yscale('log');ax.set_ylabel(r'MMD$^2$')
        ax.text(.03,.93,f'({"ab"[i]})',transform=ax.transAxes,va='top')
    spread=[[json.loads((CODE/f'results/haar_depth/circuits/K{K}_s{seed}.json').read_text())['feature_spread'] for seed in range(3)] for K in depths]
    ax=axes[2]
    for K,v in zip(depths,spread):ax.scatter([K]*3,v,color='k',s=9,alpha=.5,lw=0)
    ax.plot(depths,[np.median(v) for v in spread],'-o',color='k',ms=3.5,lw=1.2)
    identity=json.loads((CODE/'results/haar_depth/identity_spread.json').read_text())
    ax.axhline(identity['haar_reference'],color='gray',ls=':',lw=1)
    ax.axhline(identity['feature_spread'],color='k',ls='--',lw=.9)
    ax.set_yscale('log');ax.set_ylabel('feature spread');ax.set_xscale('log',base=2);ax.set_xlabel('depth $K$')
    ax.text(.97,.93,'(c)',transform=ax.transAxes,va='top',ha='right')
    save(fig,'fig04_haar_depth.png')

if __name__=='__main__':main()

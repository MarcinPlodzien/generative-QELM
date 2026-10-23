"""Only the two 1D targets, five depths and three Haar seeds; CPU and one BLAS thread."""
from _training_cli import ProcessPoolExecutor,train_setting
from concurrent.futures import as_completed
import argparse,json
import numpy as np
from qgen.io import CODE,save,sha
from qgen.data import HAAR_MODELS
from qgen.haar import build_seed,ensure_quadrature,environment,ROOT

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['features','train','all'],default='all')
 parser.add_argument('--workers',type=int,default=4);parser.add_argument('--reuse-only',action='store_true');args=parser.parse_args()
 if not 1<=args.workers<=4:parser.error('Use one to four workers.')
 if args.stage in ('features','all'):
  if args.reuse_only:
   for seed in range(3):assert (ROOT/f'circuits/seed{seed}_complete.json').exists()
  else:
   ensure_quadrature()
   with ProcessPoolExecutor(max_workers=min(args.workers,3)) as pool:
    for done in as_completed([pool.submit(build_seed,seed) for seed in range(3)]):print('SEED READY',done.result(),flush=True)
  path=CODE/'results/features/D1_expectation_identity_32768.npz'
  identity=ROOT/'identity_spread.json'
  if not identity.exists():
   with np.load(path) as data:spread=float(data['F'].std(0).mean())
   save(identity,dict(feature_spread=spread,N=12,M=298,training_points=32768,feature_sha256=sha(path),haar_reference=1/np.sqrt(4097)))
 if args.stage in ('train','all'):
  jobs=[(name,model,args.reuse_only) for name in ['five_peak','gamma'] for model in HAAR_MODELS]
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
   for i,done in enumerate(as_completed([pool.submit(train_setting,j) for j in jobs]),1):print('TRAINED',i,60,done.result(),flush=True)
 print('HAAR STAGE COMPLETE',args.stage,environment(),flush=True)

if __name__=='__main__':main()

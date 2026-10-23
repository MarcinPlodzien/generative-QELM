"""CPU-bounded training entry points; completed heads are reused without modification."""
import os,sys,argparse
from pathlib import Path
sys.dont_write_bytecode=True
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ProcessPoolExecutor
from qgen.data import TARGETS,MODELS,models_for_target
from qgen.training import train_setting

def main(targets=None,mode='both',default_models=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--model',action='append',choices=MODELS)
 parser.add_argument('--mode',choices=['both','expectation-value','single-shot'],default=mode)
 parser.add_argument('--workers',type=int,default=4)
 parser.add_argument('--reuse-only',action='store_true',help='Fail instead of fitting any missing head.')
 args=parser.parse_args()
 if not 1<=args.workers<=4:parser.error('Use one to four CPU workers.')
 models=args.model or default_models or [m for m in MODELS if args.mode=='both' or m.startswith('expectation' if args.mode=='expectation-value' else 'shot')]
 names=targets or TARGETS
 if args.model and any(model not in models_for_target(name) for name in names for model in models):
  parser.error('Requested model is outside the target/mode/seed scope of the paper.')
 jobs=[(name,model,args.reuse_only) for name in names for model in models if model in models_for_target(name)]
 if args.workers==1:completed=list(map(train_setting,jobs))
 else:
  with ProcessPoolExecutor(max_workers=args.workers) as pool:completed=list(pool.map(train_setting,jobs))
 print(f'{len(completed)} settings ready ({3*len(completed)} heads); existing heads reused.',flush=True)

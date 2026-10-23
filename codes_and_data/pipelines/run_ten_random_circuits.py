"""Run the additional random-circuit seeds, with separate ordered stages."""
from _training_cli import ProcessPoolExecutor,train_setting
import argparse
from concurrent.futures import as_completed
from qgen.data import PAPER_1D,TARGETS_2D
from qgen.features import prepare_features
from qgen.io import CODE
import json,time

def prepare(job):
 model,d,n=job
 return str(prepare_features(model,d,n=n))

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--stage',required=True,choices=['1d','2d-single-shot'])
 parser.add_argument('--workers',type=int,default=4)
 parser.add_argument('--reuse-only',action='store_true')
 args=parser.parse_args()
 if not 1<=args.workers<=4:parser.error('Use one to four CPU workers.')
 if args.stage=='2d-single-shot' and not args.reuse_only:
  report=CODE/'evaluation/ten_random_circuits_1d.json'
  if not report.exists() or json.loads(report.read_text()).get('status')!='PASS':
   parser.error('Complete the 1D stage before 2D single-shot training.')
 names=PAPER_1D if args.stage=='1d' else TARGETS_2D
 modes=['expectation','shot'] if args.stage=='1d' else ['shot']
 models=[f'{mode}_random{s}' for mode in modes for s in range(3,10)]
 d=1 if args.stage=='1d' else 2
 start=time.time()
 if not args.reuse_only:
  # Build each cache once before target jobs can request the same file concurrently.
  feature_jobs=[(m,d,32768 if m.startswith('expectation') else 4096) for m in models]
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
   for done in as_completed([pool.submit(prepare,j) for j in feature_jobs]):
    print('FEATURE READY',done.result(),flush=True)
 jobs=[(name,m,args.reuse_only) for name in names for m in models]
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  futures=[pool.submit(train_setting,j) for j in jobs]
  for count,done in enumerate(as_completed(futures),1):
   name,model=done.result();print(f'{count}/{len(jobs)} SETTINGS READY {name}/{model}; elapsed={time.time()-start:.1f}s',flush=True)
 print(f'COMPLETE stage={args.stage}; settings={len(jobs)}; candidates={3*len(jobs)}; seconds={time.time()-start:.1f}',flush=True)

if __name__=='__main__':main()

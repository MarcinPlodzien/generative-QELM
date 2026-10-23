"""Evaluate the stored heads and produce the combined summary of the paper.

Existing heads and evaluated laws are never overwritten. Missing laws are
computed from the stored heads; --reuse-only requires them to exist.
"""
import sys,os,argparse,csv,json
from pathlib import Path
sys.dont_write_bytecode=True
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from qgen.io import CODE,save,sha
from qgen.data import TARGETS,TARGETS_2D,MODELS,target,result_folder,models_for_target,settings
from concurrent.futures import ProcessPoolExecutor,as_completed
from qgen.features import expectation_features,latent_grid
from qgen.losses import DistributionLoss
from qgen.output_labels import quantized_law
from evaluation.metrics import KernelTarget,SIGMAS,total_variation

def ensure_law(name,model,reuse_only=False):
 folder=result_folder(name,model)
 if (folder/'evaluation.json').exists() and (folder/'evaluated_law.npz').exists():return
 if reuse_only:raise FileNotFoundError(f'Existing law required: {folder}')
 t=target(name);records=[json.loads((folder/f'restart{r}.json').read_text()) for r in range(3)]
 best=min(records,key=lambda r:r['final_loss']);head_path=folder/f'restart{best["restart"]}.npz'
 with np.load(head_path) as head:
  if model.startswith('expectation'):
   z=latent_grid(t['d'],131072,heldout=True)
   if '_haar_' in model:
    from qgen.haar import heldout_features
    features=heldout_features(model)
   else:features=expectation_features(model,t['d'],z)
   y=features@head['W']+head['b'];q=np.full(len(y),1/len(y))
  else:y=head['atoms'];q=head['q']
 pmf=quantized_law(y,q,t);live=pmf>0;metric=KernelTarget(t)
 loss=DistributionLoss(t['y'],t['p'])(y,q)[0]
 continuous=None;bound=None
 if model.startswith('shot'):continuous=metric.score(y,q)
 elif t['d']==1:
  blocks=np.argsort(y[:,0]).reshape(8192,-1);centroids=y[blocks].mean(1)
  radius=float(abs(y[blocks]-centroids[:,None]).max())
  bound=float(sum(4*radius/(s*np.sqrt(np.e)) for s in SIGMAS))
  continuous=metric.score(centroids,np.full(8192,1/8192))
 overflow=np.any((y<t['y'].min(0))|(y>t['y'].max(0)),axis=1)
 record=dict(target=name,model=model,selected_restart=best['restart'],all_restarts=records,
  selection='minimum training loss across3 generic starts',training_loss=best['final_loss'],
  heldout_distribution_loss=loss,heldout_distribution_loss_directions=32,
  MMD2_quantized=metric.score(t['y'][live],pmf[live]),MMD2_continuous=continuous,
  overflow_mass=float(q[overflow].sum()),solver_success=best['solver_success'],
  target_sha256=t['source_sha256'],selected_head_sha256=sha(head_path),
  law_note='131072-point independent latent quadrature in expectation-value mode; exact weighted4096-atom law in single-shot mode.')
 if bound is not None:record['MMD2_continuous_compression_absolute_bound']=bound
 if '_haar_' in model:
  from qgen.haar import parse_model,ROOT as HAAR_ROOT
  _,K,seed=parse_model(model)
  provenance=json.loads((HAAR_ROOT/f'circuits/K{K}_s{seed}.json').read_text())
  record.update(family='haar',K=K,circuit_seed=seed,feature_spread=provenance['feature_spread'],
   prior_TV_uniform=provenance['prior_TV_uniform'],gate_sha256=provenance['gate_sha256'],
   weight_quadrature_nodes=8192,law_note='131072-point held-out latent quadrature for expectation-value mode;8192-node Gauss-Legendre latent average for the4096-string single-shot law, validated to numerical precision.')
  if model.startswith('expectation'):record['evaluation_feature_sha256']=provenance['evaluation_feature_sha256']
 if t['d']==2:record['heldout_distribution_loss_64']=DistributionLoss(t['y'],t['p'],64)(y,q)[0]
 np.savez_compressed(folder/'evaluated_law.npz',atoms=y,probabilities=q,quantized_pmf=pmf,physical_atoms=y*t['scale']+t['offset'])
 save(folder/'evaluation.json',record)

def evaluate_job(job):
 name,model=job;ensure_law(name,model);return name,model

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--reuse-only',action='store_true')
 parser.add_argument('--stage',choices=['all','1d'],default='all')
 parser.add_argument('--workers',type=int,default=1);args=parser.parse_args()
 if not 1<=args.workers<=4:parser.error('Use one to four CPU workers.')
 pending=[(name,model) for name,model in settings(args.stage) if not all((result_folder(name,model)/f).exists() for f in ['evaluation.json','evaluated_law.npz'])]
 if args.reuse_only and pending:raise FileNotFoundError(f'Existing laws required: {pending}')
 if pending:
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
   futures=[pool.submit(evaluate_job,job) for job in pending]
   for i,done in enumerate(as_completed(futures),1):print('LAW READY',i,len(pending),done.result(),flush=True)
 rows=[]
 for name in TARGETS:
  t=target(name);metric=KernelTarget(t);loss=DistributionLoss(t['y'],t['p'],32)
  window=(abs(t['raw'][:,0]-1228.5)<=48) if name=='narrow_broad' else ((abs(t['y'][:,0]+3)<=.3) if name=='unequal_mixture' else None)
  for model in models_for_target(name,args.stage):
   ensure_law(name,model,args.reuse_only)
   folder=result_folder(name,model);record=json.loads((folder/'evaluation.json').read_text())
   best=min(record['all_restarts'],key=lambda r:r['final_loss'])
   with np.load(folder/'evaluated_law.npz') as law:y,q,pmf=law['atoms'],law['probabilities'],law['quantized_pmf']
   live=pmf>0;mmd=metric.score(t['y'][live],pmf[live]);heldout=loss(y,q)[0]
   assert abs(mmd-record['MMD2_quantized'])<1e-10
   overflow=float(q[np.any((y<t['y'].min(0))|(y>t['y'].max(0)),axis=1)].sum())
   rows.append(dict(target=name,model=model,mode='expectation-value mode' if model.startswith('expectation') else 'single-shot mode',
    circuit=model.split('_',1)[1],selected_restart=best['restart'],training_loss=best['final_loss'],
    heldout_loss_32=heldout,heldout_loss_64=DistributionLoss(t['y'],t['p'],64)(y,q)[0] if t['d']==2 else heldout,
    heldout_to_training_ratio=heldout/best['final_loss'],MMD2_quantized=mmd,
    MMD2_continuous=record.get('MMD2_continuous'),
    MMD2_continuous_compression_absolute_bound=record.get('MMD2_continuous_compression_absolute_bound'),
    TV_quantized=total_variation(pmf,t['p']),overflow_mass=overflow,
    solver_success=best['solver_success'],solver_message=best['solver_message'],iterations=best['iterations'],
    training_atoms=32768 if model.startswith('expectation') else 4096,evaluation_atoms=len(q),
    coarse_grid_gap_ratio=record.get('baseline_heldout_distribution_loss_32',0)/record['baseline_training_loss'] if 'baseline_training_loss' in record else None,
    source_folder=str(folder.relative_to(CODE)),
    narrow_window_target_mass=float(t['p'][window].sum()) if window is not None else None,
    narrow_window_generated_mass=float(pmf[window].sum()) if window is not None else None))
  print('EVALUATED',name,flush=True)
 save(CODE/'evaluation/summary.json',rows)
 with (CODE/'evaluation/summary.csv').open('w') as file:
  writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 gaps=[r['heldout_to_training_ratio'] for r in rows if r['target'] in TARGETS_2D and r['model'].startswith('expectation')]
 coarse_gaps=[r['coarse_grid_gap_ratio'] for r in rows if r['coarse_grid_gap_ratio'] is not None]
 save(CODE/'evaluation/manuscript_numbers.json',dict(selected_heads=len(rows),candidate_heads=3*len(rows),
  converged_selected_heads=sum(r['solver_success'] for r in rows),
  iteration_limit_settings=[f"{r['target']}/{r['model']}" for r in rows if not r['solver_success']],
  refined_2d_gap_min=min(gaps),refined_2d_gap_max=max(gaps),
  coarse_grid_2d_gap_min=min(coarse_gaps) if coarse_gaps else None,
  coarse_grid_2d_gap_max=max(coarse_gaps) if coarse_gaps else None,
  N=12,L_random=4,M=298,target_nodes=4096,grid_2d=[64,64],head_starts=3,
  expectation_training_points=32768,expectation_evaluation_points=131072,
  average_evaluation_points_per_output_node=32,display_histogram_bins=128,
  random_circuit_labels={str(s+1):s for s in range(10)},main_circuit_seed=1,
  kernel_sigmas=SIGMAS,training_directions=32,additional_evaluation_directions=64))
 print(f'WROTE evaluation/summary.csv: {len(rows)} final paper settings',flush=True)

if __name__=='__main__':main()

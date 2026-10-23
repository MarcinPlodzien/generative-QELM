"""Train only y = W x + b; circuits and feature dictionaries remain fixed.

Saved batches use the row-vector convention y = x @ W + b.
"""
import json,time
import numpy as np
from scipy.optimize import minimize
from .io import CODE,save,sha
from .data import target,result_folder
from .features import prepare_features
from .losses import DistributionLoss

def initial_head(A,q,t,seed):
 """Generic Gaussian directions, matched first/second moments only."""
 d=t['d'];rng=np.random.default_rng(12000+seed);W=rng.normal(size=(A.shape[1],d));initial_y=A@W;mu=q@initial_y;cov=(initial_y-mu).T@(q[:,None]*(initial_y-mu));e,V=np.linalg.eigh(cov)
 y=t['y'];p=t['p'];ym=p@y;yc=y-ym;et,T=np.linalg.eigh(yc.T@(p[:,None]*yc));B=V@np.diag(1/np.sqrt(e))@V.T@T@np.diag(np.sqrt(et))@T.T
 return np.vstack([W@B,ym-mu@B])

def fit(name,model,restart,reuse_only=False):
 folder=result_folder(name,model);dest=folder/f'restart{restart}.json'
 if dest.exists():
  if not dest.with_suffix('.npz').exists():raise FileNotFoundError(dest.with_suffix('.npz'))
  return json.loads(dest.read_text())
 if reuse_only:raise FileNotFoundError(f'Existing head required: {dest}')
 t=target(name);n=32768 if model.startswith('expectation') else 4096
 fp=prepare_features(model,t['d'],n=n)
 with np.load(fp) as archive:f={k:archive[k] for k in archive.files}
 q=f['q'];design=np.column_stack([f['A'],np.ones(len(q))])
 theta=initial_head(f['A'],q,t,restart);initial=theta.copy()
 loss=DistributionLoss(t['y'],t['p']);start=time.time();initial_loss=loss(design@theta,q)[0]
 def fun(flat):
  y=design@flat.reshape(theta.shape);value,gradient=loss(y,q)
  return value,(design.T@gradient).ravel()
 res=minimize(fun,theta.ravel(),jac=True,method='L-BFGS-B',options={
  'maxiter':2200 if t['d']==1 else 1600,'gtol':2e-8,'ftol':1e-12,'maxcor':20,'maxls':40})
 theta=res.x.reshape(theta.shape);W=f['transform']@theta[:-1];b=theta[-1]-f['mean']@W
 y=f['F']@W+b
 assert np.isfinite(y).all() and abs(y-design@theta).max()<1e-8
 folder.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(dest.with_suffix('.npz'),W=W,b=b,atoms=y,q=q,initial_theta=initial,theta=theta)
 record=dict(target=name,model=model,restart=restart,rng_seed=12000+restart,
  initialization='Generic Gaussian directions; matched target mean/covariance only',
  initial_loss=initial_loss,final_loss=float(res.fun),iterations=int(res.nit),evaluations=int(res.nfev),
  solver_success=bool(res.success),solver_message=str(res.message),gradient_inf=float(abs(res.jac).max()),
  seconds=time.time()-start,feature_file=str(fp.relative_to(CODE)),feature_sha256=sha(fp),
  target_sha256=t['source_sha256'],source_sha256=sha(__file__),training_latent_points=len(q))
 save(dest,record);return record

def train_setting(job):
 name,model,reuse_only=job
 for restart in range(3):fit(name,model,restart,reuse_only=reuse_only)
 return name,model

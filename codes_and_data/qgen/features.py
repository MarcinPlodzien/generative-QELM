"""Exact expectation-value features x=f(z) and single-shot parities s_B.

A measured bitstring supplies head inputs, not final output labels.
"""
import numpy as np
from scipy.stats import qmc
from .io import CODE,save,sha
from .encoding import Encoding
from .circuits import build_circuit
from .paulis import readout_strings
from .pauliprop import propagate


def circuit(model,N=12):
 if '_haar_' in model:
  from .haar import parse_model,haar_brickwork
  _,K,seed=parse_model(model);return haar_brickwork(K,seed,N)
 if model.endswith('identity'):return []
 if model.endswith('cluster'):return build_circuit('cz_h_layers',N,depth=1)
 return build_circuit('random_clifford',N,seed=int(model[-1]),depth=4)

def bits_and_parities(N):
 bits=(np.arange(2**N)[:,None]>>(N-1-np.arange(N)))&1
 ro=np.asarray(readout_strings(N,kind='z',max_k=3))
 return bits,1.-2*((bits@(ro==3).T)%2),ro

def integrated_product(letters,D):
 """Exact integrals on independent Uniform[0,1] dyadic latent blocks.

Product-to-sum frequencies are nonzero for any nonempty subset of distinct
powers of two. Even sine count gives cosines, all integrating to zero. An odd
sine count can survive only if the frequency-pi qubit is included.
"""
 value=1.
 for block in np.array_split(letters,D):
  inds=np.flatnonzero(block)
  if not len(inds):continue
  nsin=np.count_nonzero((block==1)|(block==2))
  if nsin%2==0 or block[0]==0:return 0.
  freq=np.array([0],dtype=np.int64);coeff=np.array([1.+0j])
  for i in inds:
   k=2**int(i);old=coeff
   if block[i]==3:coeff=np.r_[old*.5,old*.5]
   else:coeff=np.r_[old*(-.5j/np.sqrt(2)),old*(.5j/np.sqrt(2))]
   freq=np.r_[freq+k,freq-k]
  value*=float(np.real(np.sum(coeff*(2j/(np.pi*freq)))))
 return value

def exact_prior(model,D,N=12):
 if '_haar_' in model:
  from .haar import parse_model,cache_path
  _,K,seed=parse_model(model)
  if D!=1 or N!=12:raise ValueError((model,D,N))
  with np.load(cache_path(f'shot_haar_K{K}_s{seed}')) as data:return data['q']
 bits,_,_=bits_and_parities(N)
 if model.endswith('identity'):return np.full(2**N,2.**-N)
 ps=propagate(3*bits,circuit(model,N));mom=np.zeros(2**N)
 for l,o,c in zip(ps.letters,ps.owner,ps.coeff):mom[o]+=c*integrated_product(l,D)
 h=1
 while h<len(mom):
  v=mom.reshape(-1,2*h);a=v[:,:h].copy();b=v[:,h:].copy();v[:,:h]=a+b;v[:,h:]=a-b;h*=2
 q=mom/len(mom);assert q.min()>-1e-11 and abs(q.sum()-1)<1e-11
 return np.maximum(q,0)/np.maximum(q,0).sum()

def latent_grid(D,n,heldout=False):
 if D==1:return ((np.arange(n)+(.38196601125 if heldout else .5))/n)[:,None]
 return qmc.Sobol(D,scramble=True,seed=7921 if heldout else 1921).random_base2(int(np.log2(n)))

def expectation_features(model,D,z,N=12,return_terms=False):
 if '_haar_' in model:
  from .haar import features
  if return_terms:raise ValueError('Haar reservoirs use dense state vectors, not sparse Clifford propagation.')
  return features(z,D,circuit(model,N),N)
 _,_,ro=bits_and_parities(N);ps=propagate(ro,circuit(model,N));enc=Encoding(N,D,freq_base=np.pi,phi=np.pi/4,layout='blocks')
 theta=z[:,enc.variable()]*enc.frequencies();g=np.stack([np.ones_like(theta),np.sin(theta)/np.sqrt(2),np.sin(theta)/np.sqrt(2),np.cos(theta)],axis=-1)
 F=np.zeros((len(z),len(ro)))
 for l,o,c in zip(ps.letters,ps.owner,ps.coeff):F[:,o]+=c*np.prod(g[:,np.arange(N),l],axis=1)
 return (F,ps,ro) if return_terms else F

def prepare_features(model,D,n=4096):
 if '_haar_' in model:
  from .haar import require_features
  return require_features(model,D,n)
 path=CODE/f'results/features/D{D}_{model}_{n}.npz';meta=path.with_suffix('.json')
 if path.exists() and meta.exists():return path
 if model.startswith('shot'):
  _,F,ro=bits_and_parities(12);q=exact_prior(model,D);diag={'prior_TV_uniform':float(abs(q-1/4096).sum()/2),'exact_one_shot_law':True}
 else:
  F,ps,ro=expectation_features(model,D,latent_grid(D,n),return_terms=True);q=np.full(n,1/n)
  canonical=np.where(ps.letters==2,1,ps.letters);diag={'distinct_products_ignoring_sign':len(np.unique(canonical,axis=0)),'changed_strings_vs_identity':int(sum(not np.array_equal(l,ro[o]) or c!=1 for l,o,c in zip(ps.letters,ps.owner,ps.coeff))),'propagated_letters':ps.letters.tolist(),'owners':ps.owner.tolist(),'coefficients':ps.coeff.tolist(),'supports':[[int(i) for i in np.flatnonzero(l)] for l in ps.letters]}
 mean=q@F;sd=np.sqrt(q@((F-mean)**2));live=sd>1e-14;sd=np.where(live,sd,1.);Z=(F-mean)/sd;cov=Z.T@(q[:,None]*Z);ev,V=np.linalg.eigh(cov);keep=ev>1e-12*max(ev.max(),1.)
 transform=(V[:,keep]/np.sqrt(ev[keep]))/sd[:,None];A=(F-mean)@transform
 assert np.max(abs(A.T@(q[:,None]*A)-np.eye(A.shape[1])))<1e-7
 path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,F=F,A=A,q=q,mean=mean,transform=transform)
 diag.update(model=model,D=D,N=12,M=298,training_atoms=len(q),numerical_feature_rank=int(keep.sum()),whitening='Centered exact/quadrature covariance; discard only eigenvalues below1e-12 of largest eigenvalue; affine feature reparameterization.',feature_sha256=sha(path));save(meta,diag)
 return path

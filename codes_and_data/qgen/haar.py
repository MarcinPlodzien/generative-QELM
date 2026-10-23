"""Dense Haar brickworks.

Features and latent-averaged bitstring weights are obtained from the full state vector.
Depths share exact gate prefixes; checkpoints never modify the state.
"""
import os,re,sys,time,json
from pathlib import Path
from functools import lru_cache
import numpy as np
import scipy
from scipy.stats import unitary_group
from .io import CODE,save,sha
from .encoding import Encoding
from .circuits import brickwork_bonds

DEPTHS=(1,2,4,8,16)
SEEDS=(0,1,2)
ROOT=CODE/'results/haar_depth'
CHUNK=2048

def parse_model(model):
 match=re.fullmatch(r'(expectation|shot)_haar_K(1|2|4|8|16)_s([012])',model)
 if not match:raise ValueError(model)
 return match[1],int(match[2]),int(match[3])

def haar_brickwork(depth,seed,N=12):
 rng=np.random.default_rng(10000+seed)
 return [(unitary_group.rvs(4,random_state=rng),bond) for layer in brickwork_bonds(N,depth) for bond in layer]

def product_states(z,D,N=12,phi=np.pi/4):
 enc=Encoding(N,D,freq_base=np.pi,phi=phi,layout='blocks')
 theta=z[:,enc.variable()]*enc.frequencies()
 a0=np.cos(theta/2)*np.exp(-1j*phi/2);a1=np.sin(theta/2)*np.exp(1j*phi/2)
 psi=np.ones((len(z),1),complex)
 for i in range(N):psi=(psi[:,:,None]*np.stack([a0[:,i],a1[:,i]],1)[:,None,:]).reshape(len(z),-1)
 return psi

def apply(psi,gates,N=12):
 batch=len(psi);state=psi.reshape((batch,)+(2,)*N)
 for U,qubits in gates:
  k=len(qubits);tensor=np.asarray(U).reshape((2,)*(2*k))
  state=np.tensordot(state,tensor,axes=([1+x for x in qubits],list(range(k,2*k))))
  state=np.moveaxis(state,list(range(-k,0)),[1+x for x in qubits])
 return state.reshape(batch,-1)

@lru_cache(None)
def parities(N=12):
 from .features import bits_and_parities
 return bits_and_parities(N)[1]

def probabilities(z,D,gates,N=12,chunk=CHUNK):
 out=np.empty((len(z),2**N))
 for i in range(0,len(z),chunk):out[i:i+chunk]=abs(apply(product_states(z[i:i+chunk],D,N),gates,N))**2
 return out

def features(z,D,gates,N=12,chunk=CHUNK):
 out=np.empty((len(z),parities(N).shape[1]))
 for i in range(0,len(z),chunk):out[i:i+chunk]=(abs(apply(product_states(z[i:i+chunk],D,N),gates,N))**2)@parities(N)
 return out

def checkpoint_probabilities(z,gates,depths=DEPTHS,N=12,chunk=CHUNK):
 ends={sum(len(b) for b in brickwork_bonds(N,K)):K for K in depths}
 for i in range(0,len(z),chunk):
  batch=z[i:i+chunk];state=product_states(batch,1,N).reshape((len(batch),)+(2,)*N)
  for j,(U,qubits) in enumerate(gates,1):
   k=len(qubits);tensor=U.reshape((2,)*(2*k))
   state=np.tensordot(state,tensor,axes=([1+x for x in qubits],list(range(k,2*k))))
   state=np.moveaxis(state,list(range(-k,0)),[1+x for x in qubits])
   if j in ends:yield slice(i,i+len(batch)),ends[j],abs(state.reshape(len(batch),-1))**2

def ensure_quadrature():
 path=ROOT/'quadrature_8192.npz'
 if not path.exists():
  path.parent.mkdir(parents=True,exist_ok=True)
  x,w=np.polynomial.legendre.leggauss(8192)
  np.savez_compressed(path,z=((x+1)/2)[:,None],weights=w/2)
 return path

def environment():
 info=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__,
  threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
 try:
  from threadpoolctl import threadpool_info
  info['threadpools']=[{k:r.get(k) for k in ['internal_api','num_threads','version','architecture']} for r in threadpool_info()]
 except ImportError:info['threadpools']='threadpoolctl not installed; environment recorded'
 return info

def cache_path(model):
 mode,_,_=parse_model(model)
 return CODE/f'results/features/D1_{model}_{32768 if mode=="expectation" else 4096}.npz'

def require_features(model,D,n):
 mode,_,_=parse_model(model)
 if D!=1 or n!=(32768 if mode=='expectation' else 4096):raise ValueError((model,D,n))
 path=cache_path(model)
 if not path.exists() or not path.with_suffix('.json').exists():raise FileNotFoundError(f'Prepare Haar features first: {path}')
 return path

def heldout_features(model):
 _,K,seed=parse_model(model)
 path=ROOT/f'evaluation_features/K{K}_s{seed}.npy'
 data=np.load(path,mmap_mode='r')
 assert data.shape==(131072,298)
 return data

def _write_cache(model,F,q,metadata):
 path=cache_path(model)
 if path.exists() and path.with_suffix('.json').exists():return
 mean=q@F;sd=np.sqrt(q@((F-mean)**2));sd=np.where(sd>1e-14,sd,1.);Z=(F-mean)/sd
 ev,V=np.linalg.eigh(Z.T@(q[:,None]*Z));keep=ev>1e-12*max(ev.max(),1.)
 transform=(V[:,keep]/np.sqrt(ev[keep]))/sd[:,None];A=(F-mean)@transform
 assert abs(A.T@(q[:,None]*A)-np.eye(A.shape[1])).max()<1e-7
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.stem+'_tmp.npz')
 np.savez_compressed(tmp,F=F,A=A,q=q,mean=mean,transform=transform);tmp.replace(path)
 save(path.with_suffix('.json'),dict(**metadata,model=model,D=1,N=12,M=298,training_atoms=len(q),
  numerical_feature_rank=int(keep.sum()),feature_sha256=sha(path),haar_source_sha256=sha(__file__),
  whitening='Same centered/scaled covariance eigendecomposition and threshold as qgen.features.prepare_features.'))

def build_seed(seed):
 if seed not in SEEDS:raise ValueError(seed)
 complete=ROOT/f'circuits/seed{seed}_complete.json'
 if complete.exists():return json.loads(complete.read_text())
 from .features import latent_grid
 started=time.time();gates=haar_brickwork(16,seed)
 (ROOT/'circuits').mkdir(parents=True,exist_ok=True);(ROOT/'evaluation_features').mkdir(parents=True,exist_ok=True)
 with np.load(ensure_quadrature()) as data:zq,w=data['z'],data['weights']
 qs={K:np.zeros(4096) for K in DEPTHS}
 for sl,K,p in checkpoint_probabilities(zq,gates):qs[K]+=w[sl]@p
 for K in DEPTHS:qs[K]/=qs[K].sum()
 print(f'HAAR seed{seed}: weights ready ({time.time()-started:.1f}s)',flush=True)
 Ftr={K:np.empty((32768,298)) for K in DEPTHS}
 for sl,K,p in checkpoint_probabilities(latent_grid(1,32768),gates):Ftr[K][sl]=p@parities()
 for K in DEPTHS:
  number=sum(len(b) for b in brickwork_bonds(12,K));prefix=gates[:number]
  gate_path=ROOT/f'circuits/K{K}_s{seed}.npz'
  np.savez_compressed(gate_path,unitaries=np.asarray([U for U,_ in prefix]),bonds=np.asarray([b for _,b in prefix]))
  spread=float(Ftr[K].std(axis=0).mean());tv=float(abs(qs[K]-1/4096).sum()/2)
  meta=dict(family='haar',K=K,seed=seed,gate_rng_seed=10000+seed,gate_file=str(gate_path.relative_to(CODE)),
   gate_sha256=sha(gate_path),feature_spread=spread,prior_TV_uniform=tv,weight_quadrature_nodes=8192,
   weight_quadrature_sha256=sha(ensure_quadrature()),environment=environment())
  _write_cache(f'expectation_haar_K{K}_s{seed}',Ftr[K],np.full(32768,1/32768),meta)
  _write_cache(f'shot_haar_K{K}_s{seed}',parities(),qs[K],meta)
  save(ROOT/f'circuits/K{K}_s{seed}.json',meta)
 del Ftr
 print(f'HAAR seed{seed}: training caches ready ({time.time()-started:.1f}s)',flush=True)
 arrays={K:np.lib.format.open_memmap(ROOT/f'evaluation_features/K{K}_s{seed}.tmp.npy',mode='w+',dtype='float64',shape=(131072,298)) for K in DEPTHS}
 last=time.time()
 for sl,K,p in checkpoint_probabilities(latent_grid(1,131072,heldout=True),gates):
  arrays[K][sl]=p@parities()
  if time.time()-last>40:
   print(f'HAAR seed{seed}: evaluation points {sl.stop}/131072 ({time.time()-started:.1f}s)',flush=True);last=time.time()
 for K,array in arrays.items():array.flush()
 del arrays
 for K in DEPTHS:
  tmp=ROOT/f'evaluation_features/K{K}_s{seed}.tmp.npy';path=ROOT/f'evaluation_features/K{K}_s{seed}.npy';tmp.replace(path)
  meta_path=ROOT/f'circuits/K{K}_s{seed}.json';meta=json.loads(meta_path.read_text())
  meta.update(evaluation_points=131072,evaluation_feature_file=str(path.relative_to(CODE)),evaluation_feature_sha256=sha(path));save(meta_path,meta)
 record=dict(seed=seed,status='complete',seconds=time.time()-started,environment=environment())
 save(complete,record);print(f'HAAR seed{seed}: all depths complete ({record["seconds"]:.1f}s)',flush=True);return record

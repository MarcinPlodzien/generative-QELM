"""Independent dense integration validates exact cluster weights for both encodings."""
import os,sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from qgen.io import CODE,save
from qgen.data import settings,result_folder
from qgen.features import circuit,exact_prior,bits_and_parities,expectation_features
from qgen.circuits import CZ,H
from validate_gallery import dense_probabilities

def main():
 assert len([(n,m) for n,m in settings() if '_haar_' not in m])==166
 assert sum(m=='shot_cluster' for _,m in settings())==8
 try:result_folder('narrow_broad','shot_cluster')
 except ValueError:pass
 else:raise AssertionError('narrow_broad has no cluster single-shot setting')
 gates=circuit('shot_cluster');expected=[(CZ,(i,i+1)) for i in range(11)]+[(H,(i,)) for i in range(12)]
 assert len(gates)==23 and all(b==c and np.array_equal(U,V) for (U,b),(V,c) in zip(gates,expected))
 rows=[];max_feature_error=0.
 _,parities,_=bits_and_parities(12)
 for d in [1,2]:
  n=96 if d==1 else 24
  nodes,w=np.polynomial.legendre.leggauss(n);nodes=(nodes+1)/2;w=w/2
  if d==1:z=nodes[:,None];weights=w
  else:
   z=np.stack(np.meshgrid(nodes,nodes,indexing='ij'),axis=-1).reshape(-1,2);weights=np.outer(w,w).ravel()
  dense_q=weights@dense_probabilities('shot_cluster',d,z,N=4)
  error=float(abs(dense_q-exact_prior('shot_cluster',d,N=4)).max());assert error<1e-12,error
  z12=np.random.default_rng(804+d).uniform(size=(3,d))
  features=dense_probabilities('shot_cluster',d,z12,N=12)@parities
  max_feature_error=max(max_feature_error,float(abs(features-expectation_features('expectation_cluster',d,z12)).max()))
  q=exact_prior('shot_cluster',d)
  assert q.shape==(4096,) and np.isfinite(q).all() and q.min()>=0 and abs(q.sum()-1)<1e-14
  rows.append(dict(D=d,N=12,atoms=4096,TV_to_uniform=float(abs(q-1/4096).sum()/2),minimum_weight=float(q.min()),
   maximum_weight=float(q.max()),mass=float(q.sum()),independent_N4_dense_integral_error=error,
   weights_array_sha256=hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()))
 assert max_feature_error<1e-12,max_feature_error
 result=dict(status='PASS',circuit='nearest-neighbour CZ chain followed by H on all qubits',gate_order_checked=True,
  full_N12_Pauli_vs_dense_error=max_feature_error,priors=rows)
 save(CODE/'evaluation/cluster_single_shot_validation.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()

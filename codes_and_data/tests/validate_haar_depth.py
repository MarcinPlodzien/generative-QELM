"""Dense simulator checks against independent state vectors and exact Clifford propagation."""
import os,sys,json
from pathlib import Path
sys.dont_write_bytecode=True
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from qgen.io import CODE,save
from qgen import haar
from qgen.data import settings,result_folder
from qgen.features import circuit,expectation_features,exact_prior
from validate_gallery import dense_probabilities

def main():
 assert len(settings())==226
 for name,model in [('two_rings','shot_haar_K1_s0'),('unequal_mixture','expectation_haar_K1_s0')]:
  try:result_folder(name,model)
  except ValueError:pass
  else:raise AssertionError((name,model))
 rng=np.random.default_rng(295);error=0.;norm_error=0.
 z=rng.uniform(size=(7,1))
 for K in haar.DEPTHS:
  model=f'expectation_haar_K{K}_s0';gates=haar.haar_brickwork(K,0,N=4)
  for U,_ in gates:assert abs(U.conj().T@U-np.eye(4)).max()<2e-14
  p=haar.probabilities(z,1,gates,N=4)
  independent=dense_probabilities(model,1,z,N=4)
  error=max(error,float(abs(p-independent).max()));norm_error=max(norm_error,float(abs(p.sum(1)-1).max()))
 assert error<1e-12 and norm_error<1e-12
 clifford_feature_error=0.
 z12=rng.uniform(size=(5,1))
 for model in ['expectation_identity','expectation_cluster','expectation_random0','expectation_random1','expectation_random2']:
  F=haar.features(z12,1,circuit(model))
  clifford_feature_error=max(clifford_feature_error,float(abs(F-expectation_features(model,1,z12)).max()))
 assert clifford_feature_error<1e-12
 # Check shared prefix execution gives the same features as separate circuits.
 prefix_error=0.;points={}
 for sl,K,p in haar.checkpoint_probabilities(z,haar.haar_brickwork(16,1,N=4),N=4,chunk=3):
  points.setdefault(K,np.empty((len(z),haar.parities(4).shape[1])))[sl]=p@haar.parities(4)
 for K in haar.DEPTHS:prefix_error=max(prefix_error,float(abs(points[K]-haar.features(z,1,haar.haar_brickwork(K,1,N=4),N=4,chunk=3)).max()))
 assert prefix_error<1e-14
 print('Dense simulator, full N12 Clifford features and shared prefixes PASS; preparing8192-node quadrature.',flush=True)
 with np.load(haar.ensure_quadrature()) as data:zq,w=data['z'],data['weights']
 harmonic_error=0.
 for start in range(1,4096,128):
  k=np.arange(start,min(start+128,4096));angles=np.pi*zq*k[None,:]
  harmonic_error=max(harmonic_error,float(abs(w@np.cos(angles)).max()),float(abs(w@np.sin(angles)-(1-(-1.)**k)/(np.pi*k)).max()))
 assert harmonic_error<2e-12,harmonic_error
 prior_error=0.
 for model in ['shot_identity','shot_cluster','shot_random1']:
  gates=circuit(model);q=np.zeros(4096)
  for i in range(0,len(zq),2048):q+=w[i:i+2048]@haar.probabilities(zq[i:i+2048],1,gates)
  q/=q.sum();prior_error=max(prior_error,float(abs(q-exact_prior(model,1)).max()))
 assert prior_error<1e-12,prior_error
 result=dict(status='PASS',haar_N4_dense_probability_error=error,normalization_error=norm_error,
  full_N12_Clifford_feature_error=clifford_feature_error,shared_prefix_error=prefix_error,
  quadrature_nodes=8192,max_trigonometric_frequency=4095,quadrature_harmonic_integral_error=harmonic_error,
  N12_Clifford_exact_prior_error=prior_error,environment=haar.environment())
 save(CODE/'evaluation/haar_depth_validation.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()

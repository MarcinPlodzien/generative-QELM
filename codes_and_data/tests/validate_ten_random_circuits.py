"""Validate the random circuits, dense features, settings and product counts."""
import os,sys,json
from pathlib import Path
sys.dont_write_bytecode=True
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from qgen.io import CODE,save,sha
from qgen.data import settings,result_folder
from qgen.features import circuit,bits_and_parities,expectation_features
from qgen.circuits import clifford_group_2q
from qgen.pauliprop import propagate
from validate_gallery import dense_probabilities

def main():
 assert len(settings('1d'))==123 and len([s for s in settings() if s[1]!='shot_cluster' and '_haar_' not in s[1]])==158
 for name,model in [('two_rings','expectation_random3'),('narrow_broad','shot_random3'),('unequal_mixture_s015','shot_random3')]:
  try:result_folder(name,model)
  except ValueError:pass
  else:raise AssertionError((name,model))
 group=clifford_group_2q();assert len(group)==11520
 _,parities,ro=bits_and_parities(12)
 bonds=[(i,i+1) for layer in range(4) for i in range(layer%2,11,2)]
 assert len(bonds)==22
 rows=[];error=0.
 for seed in range(10):
  model=f'expectation_random{seed}';gates=circuit(model)
  rng=np.random.default_rng(seed);indices=[int(rng.integers(11520)) for _ in bonds]
  assert [b for _,b in gates]==bonds
  assert all(np.array_equal(U,group[index]) for (U,_),index in zip(gates,indices))
  ps=propagate(ro,gates);canonical=np.where(ps.letters==2,1,ps.letters)
  products=len(np.unique(canonical,axis=0))
  if seed==2:assert products==284
  if seed>=3:
   for d in [1,2]:
    z=np.random.default_rng(2000+seed+d).uniform(size=(2,d))
    dense=dense_probabilities(model,d,z,N=12)@parities
    error=max(error,float(abs(dense-expectation_features(model,d,z)).max()))
  rows.append(dict(seed=seed,circuit_label=seed+1,N=12,L=4,M=298,
   distinct_products_ignoring_sign=products,two_qubit_group_indices=indices,bonds=bonds))
 assert error<1e-10,error
 result=dict(status='PASS',full_N12_Pauli_vs_dense_error=error,stage_1d_settings=123,final_settings=158,
  forbidden_settings_rejected=True,group_sha256=sha(CODE/'cache/clifford_group_2q.pkl'),
  definition='Canonical propagated products with Y identified with X at phi=pi/4, ignoring sign; count is shared by the D1/D2 encodings.',circuits=rows)
 save(CODE/'evaluation/ten_random_circuit_validation.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='circuits'},indent=2))
 print('Distinct products:',[(r['circuit_label'],r['distinct_products_ignoring_sign']) for r in rows],flush=True)

if __name__=='__main__':main()

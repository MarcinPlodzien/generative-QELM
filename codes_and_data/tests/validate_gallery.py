"""Independent transport, gradients, quantum features and uniform-weight checks.

All references are computed here independently.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm,gamma
from qgen.data import target
from qgen.io import CODE,save
from qgen.features import circuit,bits_and_parities,expectation_features,exact_prior
from qgen.encoding import Encoding
from qgen.losses import DistributionLoss,transport_1d

def dense_probabilities(model,D,z,N=4):
 enc=Encoding(N,D);gates=circuit(model,N);rows=[]
 for latent in z:
  state=np.array([1.+0j]);angles=latent[enc.variable()]*enc.frequencies()
  for theta in angles:state=np.kron(state,[np.cos(theta/2),np.exp(1j*np.pi/4)*np.sin(theta/2)])
  for U,qubits in gates:
   order=list(qubits)+[i for i in range(N) if i not in qubits]
   amplitudes=state.reshape([2]*N).transpose(order).reshape(2**len(qubits),-1)
   state=(U@amplitudes).reshape([2]*N).transpose(np.argsort(order)).ravel()
  rows.append(abs(state)**2)
 return np.asarray(rows)

def main():
 rng=np.random.default_rng(7);a=np.sort(rng.normal(size=7));b=np.sort(rng.normal(size=9))
 q=rng.uniform(size=7);q/=q.sum();p=rng.uniform(size=9);p/=p.sum()
 constraints=np.zeros((16,63))
 for i in range(7):constraints[i,i*9:(i+1)*9]=1
 for j in range(9):constraints[7+j,j::9]=1
 lp=linprog(((a[:,None]-b)**2).ravel(),A_eq=constraints,b_eq=np.r_[q,p],bounds=(0,None),method='highs')
 assert lp.success
 lp_error=abs(transport_1d(a,q,b,p)[0]-lp.fun);assert lp_error<1e-10
 y=rng.normal(size=(13,2));target_y=rng.normal(size=(17,2));w=np.full(13,1/13)
 loss=DistributionLoss(target_y,np.full(17,1/17),8);value,gradient=loss(y,w);eps=1e-6;gradient_error=0.
 for i,j in [(0,0),(3,1),(9,0)]:
  delta=np.zeros_like(y);delta[i,j]=eps
  numerical=(loss(y+delta,w)[0]-loss(y-delta,w)[0])/(2*eps)
  gradient_error=max(gradient_error,abs(numerical-gradient[i,j]))
 assert gradient_error<1e-8
 _,parities,_=bits_and_parities(4);feature_error=0.
 for D in [1,2]:
  z=rng.uniform(size=(7,D))
  for model in ['expectation_identity','expectation_cluster']+[f'expectation_random{s}' for s in range(10)]:
   dense=dense_probabilities(model,D,z)@parities
   feature_error=max(feature_error,float(abs(dense-expectation_features(model,D,z,N=4)).max()))
 assert feature_error<1e-10
 # An independent Gauss-Legendre integral of actual measurement probabilities.
 nodes,weights=np.polynomial.legendre.leggauss(96);nodes=(nodes+1)/2;weights=weights/2
 integrated=dense_probabilities('expectation_identity',1,nodes[:,None]).T@weights
 uniform_error=float(abs(integrated-1/16).max());assert uniform_error<1e-12
 for D in [1,2]:assert np.array_equal(exact_prior('shot_identity',D),np.full(4096,1/4096))
 prior_error=0.
 for s in range(10):
  integrated=dense_probabilities(f'expectation_random{s}',1,nodes[:,None]).T@weights
  prior_error=max(prior_error,float(abs(integrated-exact_prior(f'shot_random{s}',1,N=4)).max()))
 assert prior_error<1e-10
 # Independent PDF construction checks the new fixed target and component masses.
 t=target('narrow_broad');j=np.arange(4096);u=j/4095
 gaussian=norm.pdf(u,loc=.30,scale=16/4095);gaussian/=gaussian.sum()
 skewed=gamma.pdf(u,a=3,loc=.45,scale=.10);skewed/=skewed.sum()
 expected=.3*gaussian+.7*skewed
 target_error=float(abs(expected-t['p']).max());assert target_error<1e-15
 assert np.array_equal(t['raw'][:,0],j) and abs(t['y'][:,0]-(10*u-5)).max()<2e-15
 assert abs((.3*gaussian).sum()-.3)<1e-15 and abs((.7*skewed).sum()-.7)<1e-15
 assert abs(u[np.argmax(skewed)]-.65)<=1/4095
 window=(j>=1181)&(j<=1276);window_mass=float(t['p'][window].sum())
 assert .299<window_mass<.3 and np.all(skewed[window]==0)
 t2=target('unequal_mixture');y2=np.linspace(-5,5,4096)
 g1=norm.pdf(y2,loc=-3,scale=.1);g1/=g1.sum()
 g2=norm.pdf(y2,loc=1,scale=1.2);g2/=g2.sum()
 expected2=.2*g1+.8*g2;unequal_error=float(abs(expected2-t2['p']).max())
 assert unequal_error<1e-15 and np.array_equal(t2['y'][:,0],y2)
 assert abs((.2*g1).sum()-.2)<1e-15 and abs((.8*g2).sum()-.8)<1e-15
 mask=(y2>=-3.3)&(y2<=-2.7);unequal_mass=float(t2['p'][mask].sum())
 assert np.array_equal(np.flatnonzero(mask),np.arange(697,942))
 assert abs(unequal_mass-expected2[mask].sum())<1e-15
 result=dict(status='PASS',unequal_mixture_independent_pdf_error=unequal_error,unequal_mixture_window_target_mass=unequal_mass,narrow_broad_independent_pdf_error=target_error,narrow_window_target_mass=window_mass,transport_linear_program_error=lp_error,sliced_gradient_error=gradient_error,
  Pauli_vs_dense_error=feature_error,identity_dense_integral_error=uniform_error,
  identity_N12_D1_D2_weights_exactly_uniform=True,random_prior_vs_dense_integral_error=prior_error)
 save(CODE/'evaluation/validation.json',result);print(result)

if __name__=='__main__':main()

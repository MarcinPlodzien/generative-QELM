"""Deterministic construction of the additional targets."""
import hashlib,json,math
import numpy as np
from .io import CODE,save,sha

def build_narrow_broad(reuse_only=False):
 """Freeze separately normalized Gaussian and shifted-Gamma node densities."""
 folder=CODE/'targets/narrow_broad';path=folder/'target.npz'
 if path.exists():
  if not (folder/'definition.json').exists():raise FileNotFoundError(folder/'definition.json')
  return path
 if reuse_only:raise FileNotFoundError(path)
 config_path=CODE/'configs/narrow_broad.json';c=json.loads(config_path.read_text())
 j=np.arange(c['nodes'],dtype=float);t=j/(len(j)-1)
 g=np.exp(-.5*((j-c['gaussian_mean_t']*(len(j)-1))/c['gaussian_sigma_index'])**2)
 u=np.maximum(t-c['gamma_shift_t'],0.)
 h=u**(c['gamma_shape']-1)*np.exp(-u/c['gamma_scale_t'])/(math.gamma(c['gamma_shape'])*c['gamma_scale_t']**c['gamma_shape'])
 g/=g.sum();h/=h.sum();p=c['component_weights'][0]*g+c['component_weights'][1]*h;p/=p.sum()
 coordinates=j[:,None];folder.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(path,coordinates=coordinates,probabilities=p)
 digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
 window=abs(j-c['gaussian_mean_t']*(len(j)-1))<=c['narrow_window_sigma']*c['gaussian_sigma_index']
 definition=dict(target='narrow_broad',N=c['N'],node_count=len(j),coordinate_offset=[2047.5],coordinate_scale=[409.5],
  convention='normalized density at fixed nodes; nearest-node assignment, midpoint ties upward',
  original_training_target_sha256=sha(path),current_target_sha256=sha(path),
  coordinates_sha256=digest(coordinates),probabilities_sha256=digest(p),
  definition={**c,'config_sha256':sha(config_path),
   'formula':'p_j = 0.3*g(t_j)/sum_k g(t_k) + 0.7*h(t_j)/sum_k h(t_k); g(t)=exp(-(t-0.30)^2/(2*(16/4095)^2)); h(t)=(t-0.45)^2*exp(-(t-0.45)/0.10)/(2*0.10^3) for t>0.45, zero otherwise',
   'gamma_mode_t':c['gamma_shift_t']+(c['gamma_shape']-1)*c['gamma_scale_t'],
   'component_node_masses':[float(g.sum()),float(h.sum())],
   'narrow_window_index_inclusive':[int(j[window][0]),int(j[window][-1])],
   'narrow_window_target_mass':float(p[window].sum())})
 save(folder/'definition.json',definition)
 return path

def build_unequal_mixture(reuse_only=False):
 """Build the narrow/broad Gaussian mixture on [-5,5]."""
 folder=CODE/'targets/unequal_mixture';path=folder/'target.npz'
 if path.exists():
  if not (folder/'definition.json').exists():raise FileNotFoundError(folder/'definition.json')
  return path
 if reuse_only:raise FileNotFoundError(path)
 config_path=CODE/'configs/unequal_mixture.json';c=json.loads(config_path.read_text())
 y=np.linspace(*c['bounds'],c['nodes']);components=[]
 for mean,sigma in zip(c['means'],c['stds']):
  g=np.exp(-.5*((y-mean)/sigma)**2);components.append(g/g.sum())
 p=np.asarray(c['component_weights'])@np.asarray(components);p/=p.sum()
 coordinates=y[:,None];folder.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(path,coordinates=coordinates,probabilities=p)
 digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
 window=abs(y-c['means'][0])<=c['narrow_window_sigma']*c['stds'][0]
 definition=dict(target='unequal_mixture',N=c['N'],node_count=len(y),coordinate_offset=[0.],coordinate_scale=[1.],
  convention='normalized density at fixed nodes; nearest-node assignment, midpoint ties upward',
  original_training_target_sha256=sha(path),current_target_sha256=sha(path),
  coordinates_sha256=digest(coordinates),probabilities_sha256=digest(p),
  definition={**c,'config_sha256':sha(config_path),
   'formula':'p_j = 0.2*g_1(y_j)/sum_k g_1(y_k) + 0.8*g_2(y_j)/sum_k g_2(y_k); g_1(y)=exp(-(y+3)^2/(2*0.1^2)); g_2(y)=exp(-(y-1)^2/(2*1.2^2))',
   'component_node_masses':[float(g.sum()) for g in components],
   'narrow_window_bounds':[-3.3,-2.7],
   'narrow_window_index_inclusive':[int(np.flatnonzero(window)[0]),int(np.flatnonzero(window)[-1])],
   'narrow_window_component_masses':[float(w*g[window].sum()) for w,g in zip(c['component_weights'],components)],
   'narrow_window_target_mass':float(p[window].sum())})
 save(folder/'definition.json',definition)
 return path
